# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 16:52:12 2025

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
ICD 医保2.0 ↔ 台湾 ICD10-CM 对齐
—— 静态规则 + 语义向量召回 + Qwen LLM 复核
特性：
- 断点续跑：自动读取已有 CSV，跳过已完成的 cn_code
- 增量保存：每处理一条即追加写入 CSV（utf-8-sig，Excel 直接打开不乱码）
- 模型缓存：Embedding / LLM 只初始化一次
- 批量高效：向量批量编码；LLM 批量判定；GPU 自动利用
- 列名鲁棒：自动探测两表中“编码列/名称列”（包含换行等变体）
- 候选池收窄：首字母 + 三位数字 code_key 约束
"""

import os
import re
import json
import time
import math
import gc
import sys
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
)

# ============== 路径与参数 ==============
# 输入表（你已上传到当前目录或同一路径）
CN_XLSX = "ICD-10医保2.0版.xlsx"
TW_XLSX = "TW_ICD10CM_2023_cn.xlsx"

# 输出 CSV（断点续跑将自动续写）
OUT_CSV  = "output_icd_pairs.csv"

# 模型本地路径（你之前已下载缓存）
EMB_LOCAL = r"C:/models/bge-base-zh-v1.5"
LLM_LOCAL = r"C:/models/Qwen2.5-3B-Instruct"

# 召回/判定参数
TOPK_EACH_KEY   = 8            # 每个 code_key 语义向量召回 Top-K 候选（越小越快）
LLM_BATCH       = 32           # LLM 判定批大小（视显存可调：16/32/48）
LLM_MAX_NEW_TOK = 64           # 生成上限
SAVE_FLUSH_EVERY = 1           # 每多少行 flush 一次（1=每行写入）

# 随机种子（便于复现）
np.random.seed(123)
torch.manual_seed(123)

# ============== 全局缓存（避免重复初始化） ==============
_EMB_TOKENIZER = None
_EMB_MODEL     = None
_LLM_TOKENIZER = None
_LLM_MODEL     = None

# ============== 工具函数 ==============
def println(msg: str):
    ts = time.strftime("[%H:%M:%S]")
    print(f"{ts} {msg}", flush=True)

def ensure_exists(path: str, kind: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{kind} 不存在：{path}")

def pick_col(cols: List[str], candidates: List[str]) -> str:
    # 允许候选里出现换行和空白差异
    norm = {c: re.sub(r"\s+", "", c) for c in cols}
    for cand in candidates:
        # cand 中可能有 \n 变体
        cand_norm = re.sub(r"\s+", "", cand)
        for c, cn in norm.items():
            if cn == cand_norm:
                return c
    raise ValueError(f"在列 {list(cols)} 中未找到候选之一：{candidates}")

def norm_code_key(icd_code: str) -> str:
    """
    规则：取首字母 + 紧随的三位数字。如 A123、B005、J010 等。
    若解析失败，返回空字符串。
    """
    if not isinstance(icd_code, str):
        return ""
    s = icd_code.strip().upper()
    m = re.match(r"([A-Z])(\d{2,3})", s.replace(".", ""))
    if not m:
        return ""
    letter = m.group(1)
    digits = m.group(2)[:3] if len(m.group(2)) >= 3 else m.group(2).ljust(3, "0")
    return f"{letter}{digits}"

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: [N, d], b: [M, d] -> [N, M]
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T

def to_device(model):
    # 设备选择 & 半精度
    if torch.cuda.is_available():
        model.to("cuda")
        return "cuda"
    elif torch.backends.mps.is_available():  # mac
        model.to("mps")
        return "mps"
    else:
        return "cpu"

# ============== 模型加载（只加载一次） ==============
def get_emb_model():
    global _EMB_MODEL, _EMB_TOKENIZER
    if _EMB_MODEL is not None and _EMB_TOKENIZER is not None:
        return _EMB_TOKENIZER, _EMB_MODEL

    println("加载向量模型 …")
    # 采用 transformers 直接做 CLS 池化，避免额外依赖
    tok = AutoTokenizer.from_pretrained(EMB_LOCAL, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(EMB_LOCAL, trust_remote_code=True)
    device = to_device(mdl)
    _EMB_TOKENIZER, _EMB_MODEL = tok, mdl.eval()
    println(f"[EMB] 已就绪，device={device}")
    return _EMB_TOKENIZER, _EMB_MODEL

def mean_pooling(last_hidden_state, attention_mask):
    # 用 mask 做平均池化
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts

def embed_texts(texts: List[str], batch_size: int = 128) -> np.ndarray:
    tok, mdl = get_emb_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vecs = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inp = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            inp = {k: v.to(device) for k, v in inp.items()}
            out = mdl(**inp)
            embs = mean_pooling(out.last_hidden_state, inp["attention_mask"])
            # BGE 模型通常还会做 normalize
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            vecs.append(embs.cpu().numpy())
    return np.vstack(vecs) if vecs else np.zeros((0, 768), dtype=np.float32)

def get_llm():
    global _LLM_MODEL, _LLM_TOKENIZER
    if _LLM_MODEL is not None and _LLM_TOKENIZER is not None:
        return _LLM_TOKENIZER, _LLM_MODEL

    println("加载 Qwen 模型 …")
    tok = AutoTokenizer.from_pretrained(LLM_LOCAL, trust_remote_code=True)
    # Qwen2.5 是 decoder-only，padding 要设为 left
    if hasattr(tok, "padding_side"):
        tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        LLM_LOCAL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    ).eval()

    _LLM_TOKENIZER, _LLM_MODEL = tok, mdl
    println("[LLM] 已就绪")
    return _LLM_TOKENIZER, _LLM_MODEL

# ============== LLM 判定 ==============
SYS_PROMPT = (
    "你是医学编码对齐助手。任务：判断大陆医保诊断与台湾ICD-10-CM条目是否为同一含义。\n"
    "仅按如下 JSON 作答：{\"is_match\": 0或1, \"score\": 0.0~1.0, \"reason\": \"简要中文理由\"}\n"
    "要求：score 越高表示越像同义；如理解不充分也要给出合理分数。不得输出除 JSON 外的多余文本。"
)

def build_chat(cn_name: str, tw_name: str) -> List[Dict[str, str]]:
    user = (
        f"大陆医保诊断：{cn_name}\n"
        f"台湾 ICD-10-CM 条目：{tw_name}\n"
        f"请给出 JSON 判定。"
    )
    return [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": user}
    ]

def llm_judge_batch(pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    tok, mdl = get_llm()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    chats = [build_chat(cn, tw) for cn, tw in pairs]
    texts = tok.apply_chat_template(chats, tokenize=False, add_generation_prompt=True)

    enc = tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        out = mdl.generate(
            **enc,
            max_new_tokens=LLM_MAX_NEW_TOK,
            do_sample=False
        )

    # ✨兼容 transformers 新旧版本
    if isinstance(out, torch.Tensor):
        sequences = out
    else:
        sequences = out.sequences

    decoded = tok.batch_decode(sequences, skip_special_tokens=True)

    results = []
    for s in decoded:
        js = {"is_match": 0, "score": 0.0, "reason": "", "raw": s}
        matches = re.findall(r"\{.*?\}", s, flags=re.S)
        if matches:
            try:
                obj = json.loads(matches[-1])
                js["is_match"] = int(obj.get("is_match", 0))
                js["score"] = float(obj.get("score", 0.0))
                js["reason"] = obj.get("reason", "")
            except:
                js["reason"] = "无法解析 JSON"
        else:
            js["reason"] = "无 JSON 输出"
        results.append(js)

    return results


# ============== 数据加载 ==============
def load_cn_tw() -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_exists(CN_XLSX, "医保表")
    ensure_exists(TW_XLSX, "台湾表")
    println("读取两张表 …")
    cn = pd.read_excel(CN_XLSX)
    tw = pd.read_excel(TW_XLSX)

    # 台湾表列名候选
    tw_code_candidates = [
        "2023年版\nICD-10-CM", "2023年版ICD-10-CM",
        "ICD-10-CM", "ICD-10-CM编码", "ICD10-CM", "编码", "代碼", "代码"
    ]
    tw_name_candidates = [
        "2023 CM中文名稱", "CM中文名稱", "中文名稱", "名称", "中文名称"
    ]
    tw_code_col = pick_col(list(tw.columns), tw_code_candidates)
    tw_name_col = pick_col(list(tw.columns), tw_name_candidates)

    # 医保表列名候选
    cn_code_candidates = [
        "诊断代码","疾病编码","ICD-10编码","ICD10编码","ICD编码","编码","ICD-10码","ICD-10"
    ]
    cn_name_candidates = ["诊断名称","疾病名称","名称"]
    cn_code_col = pick_col(list(cn.columns), cn_code_candidates)
    cn_name_col = pick_col(list(cn.columns), cn_name_candidates)

    # 精简两表只保留必要列
    cn = cn[[cn_code_col, cn_name_col]].rename(columns={cn_code_col: "cn_code", cn_name_col: "cn_name"})
    tw = tw[[tw_code_col, tw_name_col]].rename(columns={tw_code_col: "tw_code", tw_name_col: "tw_name"})

    # 去空
    cn = cn.dropna(subset=["cn_code", "cn_name"]).reset_index(drop=True)
    tw = tw.dropna(subset=["tw_code", "tw_name"]).reset_index(drop=True)

    println(f"✅ 台湾表：{len(tw)} 行，医保表：{len(cn)} 行")
    return cn, tw

# ============== 候选池构造（按 code_key 先收窄） ==============
def build_tw_bank(tw: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    构建 {code_key: {"tw_codes":[], "tw_names":[], "emb": np.ndarray}}
    """
    println("为每个 code_key 预编码台湾名称向量 …")
    tw["code_key"] = tw["tw_code"].astype(str).map(norm_code_key)
    grp = tw.groupby("code_key", dropna=False)

    bank = {}
    for key, g in grp:
        if not key:
            continue
        names = g["tw_name"].astype(str).tolist()
        codes = g["tw_code"].astype(str).tolist()
        emb   = embed_texts(names, batch_size=256)
        bank[key] = {"tw_names": names, "tw_codes": codes, "emb": emb}
    println(f"[TW] 可用 code_key 组：{len(bank)}")
    return bank

# ============== 断点续跑：已完成集合 ==============
def load_done_set() -> set:
    if not os.path.exists(OUT_CSV):
        return set()
    try:
        df = pd.read_csv(OUT_CSV, usecols=["cn_code"])
        done = set(df["cn_code"].astype(str).tolist())
        println(f"[断点续跑] 已完成：{len(done)} 个 cn_code，将跳过")
        return done
    except Exception:
        println("[断点续跑] 现有 CSV 无法读取，将从头开始（文件可能损坏或空）")
        return set()

def save_row_append(row: Dict[str, Any], header_if_new=True):
    new_file = not os.path.exists(OUT_CSV)
    df = pd.DataFrame([row])
    df.to_csv(
        OUT_CSV,
        mode='a',
        header=(new_file and header_if_new),
        index=False,
        encoding="utf-8-sig"
    )

# ============== 主流程 ==============
def main():
    # 读取两表
    cn, tw = load_cn_tw()

    # 断点续跑：过滤已完成 cn_code
    done_codes = load_done_set()
    if done_codes:
        cn = cn[~cn["cn_code"].astype(str).isin(done_codes)].reset_index(drop=True)

    println(f"待处理医保诊断：{len(cn)}")

    # 构建台湾候选库（按 code_key）
    tw_bank = build_tw_bank(tw)

    # 预留：先把所有 cn 根据 code_key 分组，降低重复 Encode 次数
    cn["code_key"] = cn["cn_code"].astype(str).map(norm_code_key)
    groups = cn.groupby("code_key", dropna=False)

    # 逐组处理（每个 code_key 独立召回 + 判定）
    processed = 0
    since_flush = 0

    for code_key, g in groups:
        if not code_key or code_key not in tw_bank:
            # 没有匹配的 key，直接写出“缺少候选”的占位
            for _, row in g.iterrows():
                out = {
                    "cn_code": row["cn_code"],
                    "cn_name": row["cn_name"],
                    "tw_code": "",
                    "tw_name": "",
                    "code_key": code_key if code_key else "",
                    "sim": 0.0,
                    "llm_score": 0.0,
                    "is_match": 0,
                    "reason": "无同 key 候选",
                }
                save_row_append(out)
                processed += 1
                since_flush += 1
                if since_flush >= SAVE_FLUSH_EVERY:
                    sys.stdout.flush()
                    since_flush = 0
            continue

        bank = tw_bank[code_key]
        tw_emb = bank["emb"]                 # [M, d]
        tw_names = bank["tw_names"]
        tw_codes = bank["tw_codes"]

        # 对这一组 cn 做向量召回
        cn_names = g["cn_name"].astype(str).tolist()
        cn_codes = g["cn_code"].astype(str).tolist()

        cn_emb = embed_texts(cn_names, batch_size=256)   # [N, d]
        sims = cosine_sim(cn_emb, tw_emb)                # [N, M]

        # 每个 cn 取 TopK_EACH_KEY
        topk_idx = np.argsort(-sims, axis=1)[:, :min(TOPK_EACH_KEY, sims.shape[1])]

        # 构造 LLM 批判定输入
        llm_pairs = []
        pair_meta = []  # 记录 (i_cn, j_tw, cn_code, tw_code, sim)
        for i, row in enumerate(cn_names):
            for j in topk_idx[i]:
                llm_pairs.append((cn_names[i], tw_names[j]))
                pair_meta.append((i, j, cn_codes[i], tw_codes[j], float(sims[i, j])))

        # 批次送 LLM（分块，避免显存爆）
        judged_all: List[Dict[str, Any]] = []
        for k in range(0, len(llm_pairs), LLM_BATCH):
            sub = llm_pairs[k:k+LLM_BATCH]
            judged = llm_judge_batch(sub)
            judged_all.extend(judged)

        # 汇总每个 cn 的最优 tw
        # 结合向量相似度与 LLM 评分（可调融合权重）
        ALPHA = 0.5  # 0~1：越大越靠向 LLM
        combined = [{} for _ in range(len(cn_names))]  # 每个 cn 选一个 best

        for m, jdict in zip(pair_meta, judged_all):
            i, j, cn_code, tw_code, sim = m
            score = jdict.get("score", 0.0)
            fused = ALPHA * score + (1 - ALPHA) * max(sim, 0.0)
            rec = {
                "cn_code": cn_code,
                "cn_name": cn_names[i],
                "tw_code": tw_code,
                "tw_name": tw_names[j],
                "code_key": code_key,
                "sim": float(sim),
                "llm_score": float(score),
                "is_match": int(jdict.get("is_match", 0)),
                "reason": jdict.get("reason", ""),
                "fused": float(fused),
            }
            # 取 fused 最大
            if not combined[i] or rec["fused"] > combined[i]["fused"]:
                combined[i] = rec

        # 写出这一组的每一条
        for rec in combined:
            if not rec:  # unlikely
                continue
            # 断点续跑：再次保护，避免重复
            # （若 OUT_CSV 存在相同 cn_code，默认以“先到先得”为准，不再覆盖）
            save_row_append({
                k: rec[k] for k in [
                    "cn_code","cn_name","tw_code","tw_name",
                    "code_key","sim","llm_score","is_match","reason","fused"
                ]
            })
            processed += 1
            since_flush += 1
            if processed % 200 == 0:
                println(f"进度：已处理 {processed}/{len(cn)}")
            if since_flush >= SAVE_FLUSH_EVERY:
                sys.stdout.flush()
                since_flush = 0

        # 释放显存
        del cn_emb, sims, topk_idx, judged_all
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    println(f"✅ 全部完成：共写出 {processed} 行到 {OUT_CSV}")

if __name__ == "__main__":
    main()

