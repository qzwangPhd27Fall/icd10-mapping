
# -*- coding: utf-8 -*-
"""
ICD医保2.0 ↔ 台湾ICD10-CM 对齐（极速版）
- 语义召回：BAAI/bge-base-zh-v1.5（向量召回 TopK=5，阈值=0.65）
- 规则约束：同一候选池内仅比较 code_key（首字母 + 3位数字）一致的台湾编码
- LLM 精判：Qwen2.5-3B-Instruct（FP16，左填充，批量=8）
- 断点续跑：自动保存中间结果到 output_progress.parquet
- 结果导出：CSV + Excel

环境假设：
- torch>=2.6（你已 2.9.0+cu128 ✅）
- transformers≈4.57.1（已适配，不用最新特性）
- pandas/openpyxl/huggingface_hub/safetensors 已装
"""

import os
import re
import json
import math
import time
import gc
import warnings
from typing import List, Tuple, Dict, Any

import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM,
    AutoTokenizer as LLMTokenizer,
)
from huggingface_hub import snapshot_download

warnings.filterwarnings("ignore", category=FutureWarning)

# ========= 可调参数（已按“最快 & 最合适”调好） =========
CN_XLSX = r"C:\Users\user\Desktop\icd10\ICD-10医保2.0版.xlsx"      # 医保2.0表
TW_XLSX = r"C:\Users\user\Desktop\icd10\TW_ICD10CM_2023_cn.xlsx"  # 台湾 ICD-10-CM 中文表

# 列名适配（支持自动探测，没找到会报错）
CN_NAME_COL_CAND = ["诊断名称", "名称", "疾病名称"]
TW_NAME_COL_CAND = ["2023 CM中文名稱", "中文名稱", "名称"]
TW_CODE_COL_CAND = ["2023年版\nICD-10-CM", "ICD10-CM", "ICD-10-CM", "编码", "代码"]

# 模型与缓存
EMB_REPO  = "BAAI/bge-base-zh-v1.5"
EMB_LOCAL = r"C:/models/bge-base-zh-v1.5"
LLM_REPO  = "Qwen/Qwen2.5-3B-Instruct"
LLM_LOCAL = r"C:/models/Qwen2.5-3B-Instruct"

# 召回与判定
TOPK          = 5        # 候选TopK（越小越快）
SIM_THRESHOLD = 0.65     # 语义相似度阈值（越低召回越多，判定更慢）
EMB_BATCH     = 256      # 向量批量
LLM_BATCH     = 8        # LLM批量（已验证 3B 在 4070 Laptop 可承受）
MAX_NEW_TOKENS = 16      # LLM输出长度
PROGRESS_PARQUET = "output_progress.parquet"

# 输出
OUT_CANDIDATES_CSV = "ICD对齐_候选TopK.csv"
OUT_MATCHES_CSV    = "ICD对齐_最终匹配.csv"
OUT_MATCHES_XLSX   = "ICD对齐_最终匹配.xlsx"

# ====================================================

def tprint(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def ensure_model(repo: str, local_dir: str, needed_files: List[str] = None):
    os.makedirs(local_dir, exist_ok=True)
    missing = []
    if needed_files is not None:
        for fn in needed_files:
            if not os.path.exists(os.path.join(local_dir, fn)):
                missing.append(fn)
    if not missing:
        tprint(f"✅ 模型已存在：{local_dir}")
        return

    tprint(f"📥 从 HuggingFace 下载 {repo} -> {local_dir}")
    snapshot_download(
        repo_id=repo,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        revision="main",
        resume_download=True
    )

def find_first_exist(df: pd.DataFrame, cands: List[str]) -> str:
    cols = list(df.columns)
    for c in cands:
        if c in cols:
            return c
    # 容错：允许通过包含关系模糊匹配
    for c in cols:
        for key in cands:
            if key.replace("\n", "").lower() in str(c).replace("\n","").lower():
                return c
    raise ValueError(f"找不到匹配的列名，候选：{cands}，现有列：{list(df.columns)[:10]}...")

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    # 全半角简单处理
    s = s.replace("（", "(").replace("）", ")").replace("，", ",").replace("：", ":")
    return s

def extract_code_key(code: str) -> str:
    """
    提取 “首字母 + 三位数字” 作为 code_key，例如 A120, B003
    """
    if not isinstance(code, str):
        code = "" if pd.isna(code) else str(code)
    code = code.strip().upper()
    m = re.search(r"([A-Z])\s*[^0-9]*([0-9]{1,3})", code)
    if not m:
        return ""
    head = m.group(1)
    digits = m.group(2).zfill(3)
    return f"{head}{digits}"

def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden_state: [B, T, H], attention_mask: [B, T]
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # [B, T, 1]
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-6)
    return summed / denom

class HFEmbedder:
    def __init__(self, local_dir: str):
        self.tok = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(local_dir, trust_remote_code=True)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.half().to("cuda")
        else:
            self.model = self.model.float()
        # 归一化器
        self.normalize = True

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        embs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i+batch_size]
            inputs = self.tok(
                chunk,
                padding=True, truncation=True, max_length=128,
                return_tensors="pt"
            )
            if torch.cuda.is_available():
                inputs = {k:v.to("cuda") for k,v in inputs.items()}
            out = self.model(**inputs)
            pool = mean_pool(out.last_hidden_state, inputs["attention_mask"])
            if self.normalize:
                pool = torch.nn.functional.normalize(pool, p=2, dim=1)
            embs.append(pool.detach().cpu().numpy())
        return np.vstack(embs)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: [M, D], b: [N, D] (已归一化)
    # 返回 [M, N]
    return a @ b.T

def build_llm(local_dir: str):
    # 1) tokenizer：左填充 + 补齐 pad_token
    tok = LLMTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    tok.padding_side = "left"  # 关键：decoder-only 必须左填充
    if tok.pad_token_id is None:
        # 多数 Qwen 系列没有 pad_token，用 eos 代替没问题
        tok.pad_token = tok.eos_token

    # 2) model：设置 pad_token_id，并高效加载
    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).eval()

    # 与 tokenizer 对齐
    model.config.pad_token_id = tok.pad_token_id
    return tok, model

def format_prompt2(cn_name: str, tw_name: str) -> str:
    """
    简短、稳定、严格 JSON 输出的判定提示
    """
    return (
        "你是一名医学编码对齐专家。现在要判断两个中文疾病名称是否表达同一疾病。\n"
        "只输出严格 JSON，不要多余文本。\n\n"
        f"医保名称：{cn_name}\n"
        f"台湾名称：{tw_name}\n\n"
        "请输出：{\"match\": 1或0, \"reason\": \"最简要原因\"}"
    )

def llm_judge_batch2(tok, model, pairs: List[Tuple[str,str]], batch_size: int = 8) -> List[Dict[str, Any]]:
    """
    批量判定：返回 [{'match':0/1, 'reason': '...'}, ...]
    """
    results: List[Dict[str,Any]] = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        prompts = [format_prompt2(cn, tw) for cn, tw in batch]

        enc = tok(
            prompts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            enc = {k:v.to("cuda") for k,v in enc.items()}

        with torch.no_grad():
            gen_out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                eos_token_id=tok.eos_token_id,
            )

        # 逐条解析
        for j in range(len(batch)):
            # 取新增片段
            input_len = enc["input_ids"][j].size(0)
            out_ids = gen_out[j][input_len:]
            text = tok.decode(out_ids, skip_special_tokens=True).strip()
            # 容错：尝试抓取JSON
            m = re.search(r"\{.*\}", text, flags=re.S)
            if m:
                text = m.group(0)
            try:
                obj = json.loads(text)
                match = int(obj.get("match", 0))
                reason = str(obj.get("reason", ""))
            except Exception:
                # 无法解析则判0
                match, reason = 0, f"parse_failed: {text[:100]}"
            results.append({"match": int(bool(match)), "reason": reason})
        # 清理
        del enc, gen_out
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return results

def main():
    # ========== 1) 读取数据 ==========
    tprint("读取两张表 …")
    cn = pd.read_excel(CN_XLSX)
    tw = pd.read_excel(TW_XLSX)

    cn_name_col = find_first_exist(cn, CN_NAME_COL_CAND)
    tw_name_col = find_first_exist(tw, TW_NAME_COL_CAND)
    tw_code_col = find_first_exist(tw, TW_CODE_COL_CAND)

    cn[cn_name_col] = cn[cn_name_col].map(normalize_text)
    tw[tw_name_col] = tw[tw_name_col].map(normalize_text)
    tw[tw_code_col] = tw[tw_code_col].astype(str).map(lambda s: s.strip())

    # 台湾侧生成 code_key，并剔除无 key 的行
    tw["code_key"] = tw[tw_code_col].map(extract_code_key)
    tw = tw[tw["code_key"] != ""].copy()

    # 为了约束：我们将医保每条记录与“相同 code_key 的台湾候选”比对。
    # 由于医保表缺少直接 code，我们用“语义召回→候选TW→取其 code_key 的众数”策略：
    # 具体：先对医保名称做向量检索，取TopK的TW，统计其 code_key 频次最高的一个作为该医保条目的目标 code_key，
    # 然后在该 code_key 分组内再做一次 TopK+LLM 精判（更快、噪声更低）。

    # ========== 2) 准备向量模型 ==========
    ensure_model(EMB_REPO, EMB_LOCAL, needed_files=["config.json"])
    tprint("加载向量模型 …")
    emb = HFEmbedder(EMB_LOCAL)

    # 预编码台湾名称向量（一次完成、缓存内存）
    tprint("为每个 code_key 预编码台湾名称向量 …")
    tw_names = tw[tw_name_col].tolist()
    tw_emb = emb.encode(tw_names, batch_size=EMB_BATCH)  # [N, D]
    # 建立 code_key -> 索引列表
    codekey_to_indices: Dict[str, List[int]] = {}
    for idx, ck in enumerate(tw["code_key"].tolist()):
        codekey_to_indices.setdefault(ck, []).append(idx)

    # ========== 3) 召回：为每条医保名称先做全库TopK，估算目标 code_key ==========
    tprint("医保诊断逐个召回 TopK …")
    cn_names = cn[cn_name_col].tolist()
    cn_emb = emb.encode(cn_names, batch_size=EMB_BATCH)  # [M, D]

    # 直接全量对比，为每条取全库TopK
    sims = cosine_sim(cn_emb, tw_emb)  # [M, N]
    topk_idx = np.argpartition(-sims, TOPK, axis=1)[:, :TOPK]   # 未排序
    # 排序
    sorted_topk = np.take_along_axis(topk_idx, np.argsort(-np.take_along_axis(sims, topk_idx, axis=1), axis=1), axis=1)

    # 估算每条医保记录的“目标 code_key”
    cn_target_codekey: List[str] = []
    cn_topk_pools: List[List[int]] = []
    for i in range(len(cn_names)):
        idxs = sorted_topk[i].tolist()
        cn_topk_pools.append(idxs)
        # 先按阈值过滤
        idxs = [j for j in idxs if sims[i, j] >= SIM_THRESHOLD]
        if not idxs:
            # 如果没有过阈值的，取最高的一个
            idxs = [sorted_topk[i, 0]]
        # 统计 code_key 众数
        keys = [tw["code_key"].iat[j] for j in idxs]
        # 众数
        key = pd.Series(keys).mode().iat[0]
        cn_target_codekey.append(key)

    # 将候选池限制在“该 code_key 分组”内再做一次 TopK（更快 & 更干净）
    refined_pairs: List[Tuple[int, int, float]] = []  # (cn_idx, tw_idx, sim)
    for i, ck in enumerate(cn_target_codekey):
        idxs = codekey_to_indices.get(ck, [])
        if not idxs:
            continue
        sims_sub = sims[i, idxs]  # [S]
        if len(idxs) > TOPK:
            sub_topk = np.argpartition(-sims_sub, TOPK)[:TOPK]
            sub_sorted = sub_topk[np.argsort(-sims_sub[sub_topk])]
            chosen = [idxs[k] for k in sub_sorted]
        else:
            chosen = sorted(range(len(idxs)), key=lambda k: -sims_sub[k])[:TOPK]
            chosen = [idxs[k] for k in chosen]
        for j in chosen:
            refined_pairs.append((i, j, float(sims[i, j])))

    # 导出候选池（可选）
    cand_rows = []
    for i, j, sc in refined_pairs:
        cand_rows.append({
            "cn_row": i,
            "医保名称": cn_names[i],
            "目标code_key": cn_target_codekey[i],
            "tw_row": j,
            "台湾名称": tw_names[j],
            "台湾编码": tw[tw_code_col].iat[j],
            "候选相似度": round(sc, 4)
        })
    cand_df = pd.DataFrame(cand_rows)
    cand_df.to_csv(OUT_CANDIDATES_CSV, index=False, encoding="utf-8-sig")

    # ========== 4) 准备 LLM ==========
    tprint("加载 Qwen2.5-3B-Instruct …")
    # 确保 tokenizer 文件齐全（有些镜像缺 tokenizer.json / special_tokens_map.json）
    need_tok_files = ["tokenizer.json", "special_tokens_map.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
    missing = [f for f in need_tok_files if not os.path.exists(os.path.join(LLM_LOCAL, f))]
    if missing:
        tprint(f"[LLM] ⚠ 检测到模型缺少 {missing}，自动从 HuggingFace 下载补全 …")
        snapshot_download(
            repo_id=LLM_REPO,
            local_dir=LLM_LOCAL,
            local_dir_use_symlinks=False,
            revision="main",
            resume_download=True
        )
    tok, model = build_llm(LLM_LOCAL)

    # ========== 5) LLM 批量精判（断点续跑） ==========
    tprint("对每个医保诊断的TopN候选进行 LLM 判定 …")

    # 读取历史进度
    judged_map: Dict[Tuple[int,int], Dict[str,Any]] = {}
    if os.path.exists(PROGRESS_PARQUET):
        try:
            prev = pd.read_parquet(PROGRESS_PARQUET)
            for _, r in prev.iterrows():
                judged_map[(int(r["cn_row"]), int(r["tw_row"]))] = {
                    "match": int(r["match"]),
                    "reason": r.get("reason", "")
                }
            tprint(f"🔁 检测到历史进度：{len(judged_map)} 条")
        except Exception:
            pass

    # 需要判定的 pair 列表
    need_pairs: List[Tuple[int,int]] = []
    for i, j, _ in refined_pairs:
        if (i, j) not in judged_map:
            need_pairs.append((i, j))

    # 批量跑
    new_records = []
    for k in range(0, len(need_pairs), LLM_BATCH):
        batch_pairs = need_pairs[k:k+LLM_BATCH]
        texts = [(cn_names[i], tw_names[j]) for (i, j) in batch_pairs]
        batch_res = llm_judge_batch2(tok, model, texts, batch_size=len(batch_pairs))
        for (i, j), res in zip(batch_pairs, batch_res):
            judged_map[(i, j)] = res
            new_records.append({
                "cn_row": i,
                "tw_row": j,
                "match": int(res["match"]),
                "reason": res.get("reason","")
            })
        # 及时落盘断点
        if new_records:
            prog_df = pd.DataFrame([
                {
                    "cn_row": i, "tw_row": j,
                    "match": judged_map[(i,j)]["match"],
                    "reason": judged_map[(i,j)].get("reason","")
                }
                for (i,j) in judged_map.keys()
            ])
            prog_df.to_parquet(PROGRESS_PARQUET, index=False)
        # 显存回收
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ========== 6) 汇总最佳一对一匹配 ==========
    # 规则：对每条医保记录，在其候选内按 (match desc, 候选相似度 desc) 取Top1
    cand_df["match"] = cand_df.apply(lambda r: judged_map.get((int(r["cn_row"]), int(r["tw_row"])), {"match":0})["match"], axis=1)
    pick_df = (
        cand_df.sort_values(["cn_row", "match", "候选相似度"], ascending=[True, False, False])
               .groupby("cn_row", as_index=False).head(1)
    )

    # 导出最终匹配
    out_cols = ["cn_row", "医保名称", "目标code_key", "tw_row", "台湾名称", "台湾编码", "候选相似度", "match"]
    pick_df[out_cols].to_csv(OUT_MATCHES_CSV, index=False, encoding="utf-8-sig")
    try:
        pick_df[out_cols].to_excel(OUT_MATCHES_XLSX, index=False)
    except Exception:
        pass

    tprint(f"✅ 完成！候选池: {len(cand_df)}，最终匹配: {len(pick_df)}")
    tprint(f"已保存：{OUT_CANDIDATES_CSV} / {OUT_MATCHES_CSV} / {OUT_MATCHES_XLSX}")
    tprint(f"断点文件：{PROGRESS_PARQUET}")

if __name__ == "__main__":
    main()
