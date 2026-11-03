# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 23:28:46 2025

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
医保2.0（中文） ↔ 台湾中文版 ICD-10-CM（中文）
【DeepSeek 强化版】：在 Top-5 候选基础上，用 DeepSeek 做“标准化 + 受约束重排”，纠正外因码等错配。

特点
- 先做：规范化精确匹配 → TF-IDF（字符2-5gram，分块Top-K，省内存）→ 产生 Top-5 候选
- 再做：DeepSeek 两步
    ① 标准化医保诊断（结构化 JSON：主疾病 + 修饰词：病原体/急慢性/外因等），带缓存
    ② 受约束重排：仅在该行 Top-5 中二次选择；若 injury_external_cause≠“是”，禁止选 V/W/X/Y/T/Z 章
- 输出：
    1) 医保表 + 最终 DeepSeek 选中的 Top-1（mapped_icd10cm, mapped_name_tw, reason）
    2) Top-5 候选（便于人工核对）

依赖（建议先在 icdmap 环境中安装）：
    pip install pandas openpyxl scikit-learn transformers accelerate sentencepiece

如需走镜像/鉴权（可选）：
    setx HF_ENDPOINT https://hf-mirror.com
    huggingface-cli login   （或设置环境变量 HF_TOKEN）

作者提示：
- 模型默认：deepseek-ai/deepseek-llm-7b-chat（公开可用）。若显存紧张，可改小模型（如 6.7B）。
- 4070 Super（12GB）可运行，BATCH_LLM 调小一些（例如 4~8）。
"""

import os
import re
import sys
import json
import time
import math
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch, os

# ========= 路径（按你的实际机器路径，如需修改就改这里） =========
P_MB = Path(r"C:\Users\user\Desktop\icd10\ICD-10医保2.0版.xlsx")   # 医保2.0
P_TW = Path(r"C:\Users\user\Desktop\icd10\TW_ICD10CM_2023_cn.xlsx")  # 台湾中文版 ICD-10-CM（请确认文件名）
OUT_TOPK = Path(r"C:\Users\user\Desktop\icd10\医保2.0_对齐台湾ICD10CM_Top5_DeepSeek.csv")
OUT_MAIN = Path(r"C:\Users\user\Desktop\icd10\医保2.0_对齐台湾ICD10CM_Top1_DeepSeek.xlsx")
LOG_PATH = Path(r"C:\Users\user\Desktop\icd10\map_mb2_twicd_deepseek.log")
CACHE_STD = Path(r"C:\Users\user\Desktop\icd10\cache_standardize_deepseek.json")  # 诊断标准化缓存

# ========= DeepSeek 模型配置 =========
HF_TOKEN = os.environ.get("HF_TOKEN", None)  # 可为空
LLM_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"

DEVICE_MAP = "auto"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
BATCH_LLM = 8            # LLM 批大小（太大会吃显存）
MAX_NEW_TOKENS_STD = 128 # 标准化输出 token 数上限
MAX_NEW_TOKENS_RER = 192 # 重排输出 token 数上限

# ========= TF-IDF & 匹配参数（内存友好） =========
TOPK = 5
CHUNK = 1000
VOCAB_MAX = 200_000
MIN_DF = 2

# ========= 章节/外因辅助 =========
EXTERNAL_CHAPTERS = set(list("VWXYZT"))  # V/W/X/Y 外因，Z 因素/接触，T 损伤/中毒（非疾病本体）
def get_chapter_letter(code: str) -> str:
    if not code: return ""
    return str(code).strip().upper()[:1]

# ========= 日志 =========
def log(msg: str):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    text = f"[{stamp}] {msg}"
    print(text, flush=True)
    try:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception:
        pass

# ========= 列识别 & 规范化 =========
def pick_name_col(df: pd.DataFrame) -> str:
    prefers = ["诊断名称","中文名称","名称","疾病名称","病名","诊断","zh_name","name","desc","描述"]
    lower = {c.lower(): c for c in df.columns}
    for p in prefers:
        if p in df.columns: return p
        if p.lower() in lower: return lower[p.lower()]
    # fallback：中文占比最高
    def zh_ratio(s):
        if pd.isna(s): return 0.0
        s = str(s)
        zh = sum('\u4e00' <= ch <= '\u9fff' for ch in s)
        return zh/max(1, len(s))
    ratios = {c: df[c].astype(str).head(500).map(zh_ratio).mean() for c in df.columns}
    return max(ratios, key=ratios.get)

def pick_code_col(df: pd.DataFrame) -> str:
    prefers = ["ICD10","ICD-10","ICD10编码","ICD编码","编码","代碼","代码","編碼","code","icd_code"]
    lower = {c.lower(): c for c in df.columns}
    for p in prefers:
        if p in df.columns: return p
        if p.lower() in lower: return lower[p.lower()]
    return df.columns[0]

def norm_zh(s: str) -> str:
    import numpy as np
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s).strip()
    s = s.replace("\u3000", " ")
    s = re.sub(r"[（）\(\)]", "()", s)
    s = s.replace("－","-").replace("—","-").replace("～","~").replace("·","·")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff\-\.\(\)·\s]", "", s)
    return s

def exact_map(mb_names: pd.Series, tw_df: pd.DataFrame, tw_name_col: str):
    tw_df = tw_df.copy()
    tw_df["_name_norm"] = tw_df[tw_name_col].astype(str).map(norm_zh)
    g = tw_df.groupby("_name_norm")
    tw_map = {k: v.index.tolist() for k, v in g}
    idx_map = []
    for i, name in mb_names.items():
        key = norm_zh(name)
        hits = tw_map.get(key, [])
        idx_map.append(hits)
    return idx_map

# ========= DeepSeek 构建 =========
def build_deepseek():
    log(f"加载 DeepSeek 模型：{LLM_NAME}")
    local_dir = r"C:\models\deepseek-coder-6.7b-instruct"
    os.makedirs(local_dir, exist_ok=True)

    # 自动断点续传下载
    snapshot_download(
        repo_id="deepseek-ai/deepseek-coder-6.7b-instruct",
        local_dir=local_dir,
        resume_download=True,
        local_dir_use_symlinks=False
    )

    tok = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        device_map="auto",
        offload_folder=r"C:\Users\user\Desktop\icd10\.offload",
        low_cpu_mem_usage=True,
        use_safetensors=True,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )

    return pipeline("text-generation", model=model, tokenizer=tok, device_map="auto")


# ========= 标准化 Prompt / 重排 Prompt =========
STD_SYS = (
    "你是医学编码助理。请将输入的中文诊断名称标准化为 ICD-10-CM 友好的要素："
    "提取主疾病名（head）及修饰（modifiers：病原体/病因、解剖部位、严重度/分期、急/慢性、侧别、妊娠相关、外因/损伤相关、后遗症/随访）。"
    "不做推断，不添加输入中没有的信息。仅返回严格 JSON（不包含多余文字）。"
)

STD_USER_TMPL = (
    "诊断名称：{name}\n"
    "请输出：\n"
    "{{\n"
    '  "head": "<主疾病名>",\n'
    '  "modifiers": {{\n'
    '    "etiology": "<病原体或病因，若无填空>",\n'
    '    "site": "<解剖部位，若无填空>",\n'
    '    "stage_severity": "<分期/严重度，若无填空>",\n'
    '    "acute_chronic": "<急性/慢性/未提及>",\n'
    '    "laterality": "<左/右/双/未提及>",\n'
    '    "pregnancy_related": "<是/否>",\n'
    '    "injury_external_cause": "<是/否>",\n'
    '    "sequela_followup": "<是/否>"\n'
    "  }}\n"
    "}}\n"
)

RERANK_SYS = (
    "你是 ICD-10-CM 对齐助手。根据“标准化诊断要素”和“候选列表（仅限候选中选择）”，返回最合适的一项。"
    "严格遵循：1) 只可在候选中选择；若均不匹配，返回 NONE；"
    "2) 若标准化要素未明确 injury_external_cause=是，则避免选择 V/W/X/Y/T/Z 章；"
    "3) 传染病（A00-B99）优先匹配感染性疾病；外因码不能作为疾病本体；"
    "4) 优先与 head 完全同义/近义且一致的项，再看修饰；"
    "仅返回严格 JSON：{\"code\":\"...\",\"name\":\"...\",\"reason\":\"...\"} 或 {\"code\":\"NONE\",\"name\":\"\",\"reason\":\"...\"}"
)

RERANK_USER_TMPL = (
    "标准化要素：\n{std_json}\n\n"
    "候选（仅从以下选择其一或NONE）：\n"
    "{lines}\n\n"
    "请输出严格 JSON，不要多余文字。"
)

def extract_last_json(text: str):
    """
    从模型输出中尽力取最后一段 JSON。保障有逗号/引号等小问题时仍能解析。
    """
    text = text.strip()
    # 找到最后一个 '{' 到最后一个 '}' 之间的内容
    l = text.rfind("{")
    r = text.rfind("}")
    if l != -1 and r != -1 and r > l:
        frag = text[l:r+1]
        try:
            return json.loads(frag)
        except Exception:
            # 宽松修正：替换中文引号等
            frag2 = frag.replace("“","\"").replace("”","\"").replace("’","'").replace("‘","'")
            # 去掉尾随逗号
            frag2 = re.sub(r",\s*}", "}", frag2)
            frag2 = re.sub(r",\s*]", "]", frag2)
            try:
                return json.loads(frag2)
            except Exception:
                return None
    return None

# ========= DeepSeek 调用（批处理） =========
def standardize_batch(gen, names):
    prompts = []
    for n in names:
        user = STD_USER_TMPL.format(name=n)
        full = f"<|system|>\n{STD_SYS}\n<|user|>\n{user}\n<|assistant|>\n"
        prompts.append(full)
    outs = gen(prompts, max_new_tokens=MAX_NEW_TOKENS_STD, do_sample=False)
    results = []
    for o in outs:
        txt = o["generated_text"]
        js = extract_last_json(txt)
        if not js:
            js = {"head": n, "modifiers": {
                "etiology":"", "site":"", "stage_severity":"", "acute_chronic":"未提及",
                "laterality":"未提及", "pregnancy_related":"否", "injury_external_cause":"否", "sequela_followup":"否"
            }}
        results.append(js)
    return results

def rerank_batch(gen, std_json_list, candidate_lines_list):
    prompts = []
    for std_json, lines in zip(std_json_list, candidate_lines_list):
        user = RERANK_USER_TMPL.format(std_json=json.dumps(std_json, ensure_ascii=False, indent=2), lines=lines)
        full = f"<|system|>\n{RERANK_SYS}\n<|user|>\n{user}\n<|assistant|>\n"
        prompts.append(full)
    outs = gen(prompts, max_new_tokens=MAX_NEW_TOKENS_RER, do_sample=False)
    results = []
    for o in outs:
        txt = o["generated_text"]
        js = extract_last_json(txt)
        if not js:
            js = {"code":"NONE","name":"","reason":"parse_error"}
        results.append(js)
    return results

# ========= 主流程 =========
def main():
    log("启动：医保2.0 ↔ 台湾ICD-10-CM（DeepSeek 重排版）")
    log(f"Python: {sys.executable}")
    log(f"医保表存在: {P_MB.exists()}  台湾表存在: {P_TW.exists()}")
    if not P_MB.exists() or not P_TW.exists():
        log("❌ 输入文件不存在，请检查路径"); sys.exit(2)

    # 读取
    mb = pd.read_excel(P_MB, dtype=str)
    tw = pd.read_excel(P_TW, dtype=str)

    mb_name_col = pick_name_col(mb)
    tw_name_col = pick_name_col(tw)
    tw_code_col = pick_code_col(tw)
    log(f"✓ 医保名称列：{mb_name_col}")
    log(f"✓ 台湾名称列：{tw_name_col}，编码列：{tw_code_col}")

    mb_names = mb[mb_name_col].fillna("").astype(str).reset_index(drop=True)
    tw_names = tw[tw_name_col].fillna("").astype(str).reset_index(drop=True)

    # 1) 规范化 + 精确匹配
    log("▶ 精确匹配（规范化后）...")
    exact_hits = exact_map(mb_names, tw, tw_name_col)
    mapped_idx = {}
    unresolved = []
    for i, hits in enumerate(exact_hits):
        if len(hits) == 1:
            mapped_idx[i] = hits[0]
        else:
            unresolved.append(i)
    log(f"  · 精确唯一命中：{len(mapped_idx)} 条；待模糊：{len(unresolved)} / {len(mb_names)}")

    # 2) TF-IDF 模糊匹配（仅对未决）
    rows = []
    if unresolved:
        log("▶ TF-IDF 建模（字符2-5gram，省内存参数）...")
        mb_unres_list = mb_names.iloc[unresolved].tolist()
        vect = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2,5),
            min_df=MIN_DF,
            max_features=VOCAB_MAX
        )
        tfidf_mb = vect.fit_transform(mb_unres_list)
        tfidf_tw = vect.transform(tw_names.tolist())

        log(f"▶ 分块相似度 Top-{TOPK}（CHUNK={CHUNK})...")
        for start in range(0, tfidf_mb.shape[0], CHUNK):
            end = min(start + CHUNK, tfidf_mb.shape[0])
            log(f"  · 处理 {start} ~ {end-1} / {tfidf_mb.shape[0]-1}")
            X = tfidf_mb[start:end]
            sims_sparse = cosine_similarity(X, tfidf_tw, dense_output=False)
            for local_row in range(sims_sparse.shape[0]):
                row = sims_sparse.getrow(local_row)
                idxs, vals = row.indices, row.data
                global_mb_row = unresolved[start + local_row]
                if len(vals) == 0:
                    rows.append({
                        "mb_row": global_mb_row, "医保名称": mb_names.iloc[global_mb_row],
                        "rank": 1, "tw_icd_code": "", "tw_icd_name": "", "sim_score": 0.0,
                        "match_type": "tfidf_none"
                    })
                    continue
                if len(vals) > TOPK:
                    topk_part = np.argpartition(-vals, TOPK-1)[:TOPK]
                    order = topk_part[np.argsort(-vals[topk_part])]
                else:
                    order = np.argsort(-vals)
                for r, pos in enumerate(order, start=1):
                    j = idxs[pos]
                    rows.append({
                        "mb_row": global_mb_row, "医保名称": mb_names.iloc[global_mb_row],
                        "rank": r,
                        "tw_icd_code": tw.iloc[j][tw_code_col],
                        "tw_icd_name": tw.iloc[j][tw_name_col],
                        "sim_score": float(vals[pos]),
                        "match_type": "tfidf_topk"
                    })

    # 把“精确唯一命中”也加入 Top-K（rank=1）
    for mb_row, tw_row in mapped_idx.items():
        rows.append({
            "mb_row": mb_row, "医保名称": mb_names.iloc[mb_row],
            "rank": 1,
            "tw_icd_code": tw.iloc[tw_row][tw_code_col],
            "tw_icd_name": tw.iloc[tw_row][tw_name_col],
            "sim_score": 1.0,
            "match_type": "exact"
        })

    topk_df = pd.DataFrame(rows)
    if topk_df.empty:
        log("❗ 没有候选产生，请检查输入表或列识别。"); sys.exit(1)

    # 保存原始 Top-5（供审阅）
    topk_export = (topk_df.sort_values(["mb_row","rank"])
                             .rename(columns={"tw_icd_code":"icd10cm_code",
                                              "tw_icd_name":"icd10cm_name_tw"}))
    OUT_TOPK.parent.mkdir(parents=True, exist_ok=True)
    topk_export.to_csv(OUT_TOPK, index=False, encoding="utf-8-sig")
    log(f"✓ 已写出原始 Top-5 候选：{OUT_TOPK}")

    # 3) DeepSeek：标准化 + 受约束重排
    # 3.1 载入/构建 LLM
    log("▶ 加载 DeepSeek 模型（用于标准化 + 重排）...")
    gen = build_deepseek()

    # 3.2 标准化（带缓存）
    uniq_names = list(dict.fromkeys(mb_names.tolist()))
    cache = {}
    if CACHE_STD.exists():
        try:
            cache = json.loads(CACHE_STD.read_text(encoding="utf-8"))
            log(f"  · 载入标准化缓存：{len(cache)} 条")
        except Exception:
            log("  · 缓存读取失败，忽略。")

    need = [n for n in uniq_names if n not in cache]
    if need:
        log(f"▶ 标准化 {len(need)} 条名称（BATCH_LLM={BATCH_LLM})...")
        for i in range(0, len(need), BATCH_LLM):
            batch = need[i:i+BATCH_LLM]
            stds = standardize_batch(gen, batch)
            for k, v in zip(batch, stds):
                cache[k] = v
            CACHE_STD.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            log(f"  · 标准化进度：{i+len(batch)}/{len(need)}")
    else:
        log("  · 全部命中缓存，无需标准化调用。")

    std_list = [cache.get(n, {"head": n, "modifiers": {"injury_external_cause":"否"}}) for n in mb_names.tolist()]

    # 3.3 为每行构造受约束候选行文本，并根据外因标志进行预过滤
    final_rows = []
    grouped = topk_df.groupby("mb_row")
    mb_total = len(mb_names)

    log("▶ 受约束重排（仅在各自的Top-5里挑选；必要时可剔除外因章）...")
    # 分批调用模型重排
    mb_indices = []
    std_batch = []
    cand_lines_batch = []

    def flush_rerank_batch():
        if not std_batch: return []
        outs = rerank_batch(gen, std_batch, cand_lines_batch)
        result_objs = outs
        out_rows = []
        for mb_row, std_obj, lines_text, choice in zip(mb_indices, std_batch, cand_lines_batch, result_objs):
            code = str(choice.get("code","")).strip().upper()
            name = str(choice.get("name","")).strip()
            reason = str(choice.get("reason","")).strip()
            # 如果返回 NONE 或空，做兜底：取非外因章的 rank 最小候选，否则取 rank=1
            if not code or code == "NONE":
                cands = grouped.get_group(mb_row).sort_values("rank")
                # 先尝试非外因章
                pick = None
                for _, r in cands.iterrows():
                    if get_chapter_letter(r["tw_icd_code"]) not in EXTERNAL_CHAPTERS:
                        pick = r; break
                if pick is None and len(cands)>0:
                    pick = cands.iloc[0]
                if pick is not None:
                    out_rows.append({
                        "mb_row": mb_row,
                        "mapped_icd10cm": pick["tw_icd_code"],
                        "mapped_name_tw": pick["tw_icd_name"],
                        "mapped_sim": pick["sim_score"],
                        "reason": f"fallback:{reason or 'none'}"
                    })
                else:
                    out_rows.append({
                        "mb_row": mb_row,
                        "mapped_icd10cm": "",
                        "mapped_name_tw": "",
                        "mapped_sim": 0.0,
                        "reason": f"none:{reason or 'none'}"
                    })
            else:
                # 以模型选择为准；若不在候选内，仍回退
                cands = grouped.get_group(mb_row)
                hit = cands[cands["tw_icd_code"].astype(str).str.upper() == code]
                if hit.empty:
                    # 回退策略同上
                    cands2 = cands.sort_values("rank")
                    pick = None
                    for _, r in cands2.iterrows():
                        if get_chapter_letter(r["tw_icd_code"]) not in EXTERNAL_CHAPTERS:
                            pick = r; break
                    if pick is None and len(cands2)>0:
                        pick = cands2.iloc[0]
                    if pick is not None:
                        out_rows.append({
                            "mb_row": mb_row,
                            "mapped_icd10cm": pick["tw_icd_code"],
                            "mapped_name_tw": pick["tw_icd_name"],
                            "mapped_sim": pick["sim_score"],
                            "reason": f"model_pick_not_in_top5:{code}; fallback"
                        })
                    else:
                        out_rows.append({
                            "mb_row": mb_row,
                            "mapped_icd10cm": "",
                            "mapped_name_tw": "",
                            "mapped_sim": 0.0,
                            "reason": f"model_pick_not_in_top5:{code}; none"
                        })
                else:
                    pick = hit.sort_values("rank").iloc[0]
                    out_rows.append({
                        "mb_row": mb_row,
                        "mapped_icd10cm": pick["tw_icd_code"],
                        "mapped_name_tw": pick["tw_icd_name"],
                        "mapped_sim": pick["sim_score"],
                        "reason": reason or "model_pick"
                    })
        return out_rows

    for i in range(mb_total):
        std_obj = std_list[i]
        mods = std_obj.get("modifiers", {}) if isinstance(std_obj, dict) else {}
        injury_flag = str(mods.get("injury_external_cause","")).strip()
        # 本行的候选
        if i not in grouped.groups:
            # 没有候选，空占位
            mb_indices.append(i)
            std_batch.append(std_obj)
            cand_lines_batch.append("（无候选）")
        else:
            cands = grouped.get_group(i).sort_values("rank")
            # 若未明确 injury_external_cause=是，先过滤 V/W/X/Y/T/Z 章；若全被滤光，再恢复原始
            filtered = []
            if injury_flag != "是":
                for _, r in cands.iterrows():
                    if get_chapter_letter(r["tw_icd_code"]) not in EXTERNAL_CHAPTERS:
                        filtered.append(r)
                if len(filtered) == 0:
                    filtered = [r for _, r in cands.iterrows()]
            else:
                filtered = [r for _, r in cands.iterrows()]
            # 构造候选文本
            lines = []
            for idx, r in enumerate(filtered[:TOPK], start=1):
                ch = get_chapter_letter(r["tw_icd_code"])
                lines.append(f"{idx}) {r['tw_icd_code']} | {r['tw_icd_name']} | 章:{ch}")
            if not lines:
                lines = ["1)  |  | 章: "]
            mb_indices.append(i)
            std_batch.append(std_obj)
            cand_lines_batch.append("\n".join(lines))

        # 批量触发
        if len(std_batch) >= BATCH_LLM:
            final_rows += flush_rerank_batch()
            mb_indices, std_batch, cand_lines_batch = [], [], []

        # 打印进度
        if (i+1) % 1000 == 0:
            log(f"  · 重排进度：{i+1}/{mb_total}")

    # 余量批次
    final_rows += flush_rerank_batch()

    final_df = pd.DataFrame(final_rows)
    if final_df.empty:
        log("❗ 重排未得到任何结果"); sys.exit(1)

    # 合并回医保表
    mb_out = mb.copy().reset_index(drop=True)
    mb_out["mb_row"] = mb_out.index
    mb_out = mb_out.merge(final_df, how="left", on="mb_row")

    # 写文件
    OUT_MAIN.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_MAIN, engine="openpyxl") as w:
        mb_out.to_excel(w, sheet_name="映射结果_Top1_DeepSeek", index=False)

    log("✅ 完成：")
    log(f"  Top-5 候选（原始TF-IDF）：{OUT_TOPK}")
    log(f"  最终Top-1（DeepSeek重排）：{OUT_MAIN}")
    log("提示：若显存不足，调小 BATCH_LLM；若 TF-IDF 匹配偏差较大，可适当提高 MIN_DF、降低 VOCAB_MAX。")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        log(f"❌ 运行出错：{e}")
        sys.exit(1)
