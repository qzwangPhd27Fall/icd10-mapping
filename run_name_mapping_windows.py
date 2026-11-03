# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 02:01:34 2025

@author: Administrator

"""

import pandas as pd
import numpy as np
import re
import sys
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- 用户文件路径 ----------
P_MB = Path(r"C:\Users\user\Desktop\icd10\ICD-10医保2.0版.xlsx")
P_CCIR = Path(r"C:\Users\user\Desktop\icd10\CCIR_v2025-1.csv")

OUT_MAIN = Path(r"C:\Users\user\Desktop\icd10\医保2.0_name_map_with_CCIR.xlsx")
OUT_TOPK = Path(r"C:\Users\user\Desktop\icd10\医保2.0_name_map_topk.csv")

TOPK = 5
CHUNK = 800#每批处理医保名称条数，可按内存调大/调小
VOCAB_MAX = 120_00 # TF-IDF 最大特征数
MIN_DF = 3       # 过滤只出现一次的 n-gram，降低噪音与内存

# ---------- 辅助函数 ----------
def pick_name_col(df: pd.DataFrame) -> str:
    prefers = ["名称","诊断名称","中文名称","疾病名称","病名","诊断","name","diag_name","zh_name"]
    lower = {c.lower(): c for c in df.columns}
    for p in prefers:
        if p in df.columns: return p
        if p.lower() in lower: return lower[p.lower()]
    def zh_ratio(s):
        if pd.isna(s): return 0.0
        s = str(s); zh = sum('\u4e00' <= ch <= '\u9fff' for ch in s)
        return zh / max(1, len(s))
    ratios = {c: df[c].astype(str).head(500).map(zh_ratio).mean() for c in df.columns}
    return max(ratios, key=ratios.get)

def load_ccir_table(p_csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(p_csv, header=None, dtype=str, encoding="utf-8", engine="python")
    start_idx = None
    for i in range(len(raw)):
        v = str(raw.iloc[i,0])
        if re.match(r"^'?[A-Z][0-9A-Z]{2,6}'?$", v):
            start_idx = i; break
    if start_idx is None:
        print("❌ 无法定位 CCIR 数据起始行，请检查 CSV。", file=sys.stderr)
        sys.exit(2)
    ccir = raw.iloc[start_idx:].copy()
    ccir.columns = ["code_raw","desc_en","ccir_flag"]
    ccir["ccir_flag"] = ccir["ccir_flag"].astype(str).str.strip().str.replace(r"[^019]","", regex=True)
    def norm_code(x):
        if pd.isna(x): return None
        s = str(x).strip().upper().strip("'\"")
        s = re.sub(r"[^A-Z0-9\.]", "", s)
        return s.replace(".", "") or None
    ccir["_code_norm"] = ccir["code_raw"].map(norm_code)
    ccir = ccir.dropna(subset=["_code_norm"]).drop_duplicates("_code_norm")
    return ccir

def ccir_label(v: str) -> str:
    return {"1":"Chronic/慢性","0":"Non-chronic/非慢性","9":"No determination/无法判定"}.get(str(v).strip(), "")

# ---------- 主流程 ----------
def main():
    print("▶ 读取医保与 CCIR 文件...")
    mb = pd.read_excel(P_MB, dtype=str)
    ccir = load_ccir_table(P_CCIR)

    name_col = pick_name_col(mb)
    print(f"✓ 识别医保名称列：{name_col}")

    mb_names = mb[name_col].fillna("").astype(str).tolist()
    ccir_texts = ccir["desc_en"].fillna("").astype(str).tolist()

    print("▶ 构建 TF-IDF（字符 2-5gram，max_features/min_df 控制规模）...")
    vect = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2,5),
        min_df=MIN_DF,
        max_features=VOCAB_MAX
    )
    tfidf_mb_full = vect.fit_transform(mb_names)     # CSR
    tfidf_ccir = vect.transform(ccir_texts)          # CSR

    print("▶ 分块计算稀疏相似度并提取每行 Top-{}...".format(TOPK))
    ccir_codes = ccir["_code_norm"].tolist()
    ccir_descs = ccir["desc_en"].tolist()
    ccir_flags = ccir["ccir_flag"].tolist()

    rows = []
    n = tfidf_mb_full.shape[0]
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        print(f"  · 处理 {start} ~ {end-1} / {n-1}")
        X = tfidf_mb_full[start:end]    # CSR (CHUNK x D)

        # 稀疏余弦相似度（不输出 dense），形状 (chunk, N_ccir)
        sims_sparse = cosine_similarity(X, tfidf_ccir, dense_output=False)  # returns CSR

        # 对每一行从稀疏向量里取 Top-K
        for local_row in range(sims_sparse.shape[0]):
            row = sims_sparse.getrow(local_row)
            if row.nnz == 0:
                # 没有任何重叠 n-gram，相似度都为 0：给一个空候选（可选：跳过）
                continue
            # 取该行非零元素的 indices 和 data
            idxs = row.indices
            vals = row.data
            if len(vals) > TOPK:
                topk_part = np.argpartition(-vals, TOPK-1)[:TOPK]
                order = topk_part[np.argsort(-vals[topk_part])]
            else:
                order = np.argsort(-vals)
            for rank, pos in enumerate(order, start=1):
                rows.append({
                    "mb_row": start + local_row,
                    "医保名称": mb_names[start + local_row],
                    "rank": rank,
                    "icd10cm_code": ccir_codes[idxs[pos]],
                    "icd10cm_desc": ccir_descs[idxs[pos]],
                    "ccir_flag": ccir_flags[idxs[pos]],
                    "sim_score": float(vals[pos])
                })

    topk_df = pd.DataFrame(rows)
    # 有可能某些行完全没候选（全 0），这里可以补一个空记录方便对齐
    have = set(topk_df["mb_row"].unique())
    for i in range(n):
        if i not in have:
            topk_df = pd.concat([topk_df, pd.DataFrame([{
                "mb_row": i, "医保名称": mb_names[i], "rank": 1,
                "icd10cm_code": "", "icd10cm_desc": "", "ccir_flag": "", "sim_score": 0.0
            }])], ignore_index=True)

    # 只取每行 rank=1 做 Top-1 映射
    top1 = (topk_df.sort_values(["mb_row","rank"])
                    .drop_duplicates(subset=["mb_row"], keep="first")
                    .rename(columns={
                        "icd10cm_code":"mapped_icd10cm",
                        "icd10cm_desc":"mapped_desc_en",
                        "ccir_flag":"mapped_ccir_flag",
                        "sim_score":"mapped_sim"
                    }))

    # —— 原来这里是 join(on=mb_out.reset_index().index)；改成先显式添加行号再 merge —— 
    # 回填到医保表（按行号对齐）
    mb_out = mb.copy().reset_index(drop=True)
    mb_out["mb_row"] = mb_out.index  # 明确的行号键

    # 只取每行 rank=1 做 Top-1 映射
    top1 = (topk_df.sort_values(["mb_row","rank"])
                    .drop_duplicates(subset=["mb_row"], keep="first")
                    .rename(columns={
                        "icd10cm_code":"mapped_icd10cm",
                        "icd10cm_desc":"mapped_desc_en",
                        "ccir_flag":"mapped_ccir_flag",
                        "sim_score":"mapped_sim"
                    }))

    # 和 top1 用 mb_row 对齐
    mb_out = mb_out.merge(
        top1[["mb_row","mapped_icd10cm","mapped_desc_en","mapped_ccir_flag","mapped_sim"]],
        how="left",
        on="mb_row"
    )

    # 友好标签
    mb_out["CCIR_label"] = mb_out["mapped_ccir_flag"].map(ccir_label)

    # （可选）导出前去掉内部用的 mb_row
    # mb_out = mb_out.drop(columns=["mb_row"])
    print("▶ 写入结果：")
    OUT_MAIN.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_MAIN, engine="openpyxl") as w:
        mb_out.to_excel(w, sheet_name="映射结果_Top1", index=False)
    topk_df.to_csv(OUT_TOPK, index=False, encoding="utf-8-sig")

    print(f"✅ 完成：\n  {OUT_MAIN}\n  {OUT_TOPK}")
    print("提示：Top1 是自动选择；如有歧义，请打开 Top5 文件进行人工确认。")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ 运行出错：", e)
        sys.exit(1)

