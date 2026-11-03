# -*- coding: utf-8 -*-
"""

"""

import re
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ========== 请根据你的电脑修改这三条路径 ==========
P_MB = Path(r"C:\Users\user\Desktop\icd10\ICD-10医保2.0版.xlsx")          # 医保2.0 Excel
P_TW = Path(r"C:\Users\user\Desktop\icd10\TW_ICD10CM_2023_cn.xlsx")      # 台湾中文版 ICD-10-CM Excel
OUT_MAIN = Path(r"C:\Users\user\Desktop\icd10\医保2.0_对齐台湾ICD10CM_Top1.xlsx")
OUT_TOPK = Path(r"C:\Users\user\Desktop\icd10\医保2.0_对齐台湾ICD10CM_Top5.csv")

# ========== 内存友好参数（必要时可以收紧） ==========
TOPK = 5           # 每条医保名称输出前 K 个候选
CHUNK = 1000       # 分块大小（每次处理多少条医保名称）；内存紧张就调小，比如 500
VOCAB_MAX = 200_000  # TF-IDF 最大特征数；内存紧张可降到 120_000
MIN_DF = 2         # 过滤只出现一次的 n-gram；可提高到 3 进一步降噪与省内存

# ---------- 工具函数 ----------
def log(msg: str):
    print(msg, flush=True)

def pick_name_col(df: pd.DataFrame) -> str:
    """自动找中文名称列（两边都用）"""
    prefers = ["诊断名称","中文名称","名称","疾病名称","病名","诊断","zh_name","name","desc","描述"]
    lower = {c.lower(): c for c in df.columns}
    for p in prefers:
        if p in df.columns: return p
        if p.lower() in lower: return lower[p.lower()]
    # 退化：选中文占比最高的列
    def zh_ratio(s):
        if pd.isna(s): return 0.0
        s = str(s)
        zh = sum('\u4e00' <= ch <= '\u9fff' for ch in s)
        return zh/max(1, len(s))
    ratios = {c: df[c].astype(str).head(500).map(zh_ratio).mean() for c in df.columns}
    return max(ratios, key=ratios.get)

def pick_code_col(df: pd.DataFrame) -> str:
    """自动找编码列（台湾侧用作返回字段；医保侧不强制要求）"""
    prefers = ["ICD10","ICD-10","ICD10编码","ICD编码","编码","代碼","代码","編碼","code","icd_code"]
    lower = {c.lower(): c for c in df.columns}
    for p in prefers:
        if p in df.columns: return p
        if p.lower() in lower: return lower[p.lower()]
    # 若没有明显编码列，就用第一列兜底
    return df.columns[0]

def norm_zh(s: str) -> str:
    """中文名称规范化：去空白、去全角空格、统一括号与连字符、去多余符号"""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s).strip()
    s = s.replace("\u3000", " ")  # 全角空格
    s = re.sub(r"[（）\(\)]", "()", s)  # 括号统一成半角
    s = s.replace("－","-").replace("—","-").replace("～","~").replace("·","·")
    s = re.sub(r"\s+", " ", s)
    # 去掉不影响语义的符号（保留中文、数字、字母、常见连接符、括号）
    s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff\-\.\(\)·\s]", "", s)
    return s

def exact_map(mb_names: pd.Series, tw_df: pd.DataFrame, tw_name_col: str):
    """做精确匹配（规范化后完全相同）"""
    # 台湾侧建立 norm 名 → 多条记录的列表（可能一名多码）
    tw_df = tw_df.copy()
    tw_df["_name_norm"] = tw_df[tw_name_col].astype(str).map(norm_zh)
    g = tw_df.groupby("_name_norm")
    tw_map = {k: v.index.tolist() for k, v in g}
    # 医保侧逐条查
    idx_map = []
    for i, name in mb_names.items():
        key = norm_zh(name)
        hits = tw_map.get(key, [])
        idx_map.append(hits)
    return idx_map  # list of list[int]

# ---------- 主流程 ----------
def main():
    log("▶ 读取文件...")
    if not P_MB.exists():
        log(f"❌ 医保文件不存在：{P_MB}"); sys.exit(2)
    if not P_TW.exists():
        log(f"❌ 台湾文件不存在：{P_TW}"); sys.exit(2)

    mb = pd.read_excel(P_MB, dtype=str)
    tw = pd.read_excel(P_TW, dtype=str)

    mb_name_col = pick_name_col(mb)
    tw_name_col = pick_name_col(tw)
    tw_code_col = pick_code_col(tw)
    log(f"✓ 医保名称列：{mb_name_col}")
    log(f"✓ 台湾名称列：{tw_name_col}，台湾编码列：{tw_code_col}")

    mb_names = mb[mb_name_col].fillna("").astype(str)
    tw_names = tw[tw_name_col].fillna("").astype(str)

    # ========== 第一阶段：规范化 + 精确匹配 ==========
    log("▶ 规范化精确匹配...")
    exact_hits = exact_map(mb_names, tw, tw_name_col)

    # 先把“有唯一命中”的直接定为 Top-1
    mapped_idx = {}   # mb_row -> tw_row_index
    unresolved = []   # 需要进入模糊匹配的行
    for i, hits in enumerate(exact_hits):
        if len(hits) == 1:
            mapped_idx[i] = hits[0]
        elif len(hits) > 1:
            # 名称相同但多条台湾记录：放入未决，交给 TF-IDF 再 disambiguate
            unresolved.append(i)
        else:
            unresolved.append(i)

    log(f"  · 精确唯一命中：{len(mapped_idx)} 条")
    log(f"  · 待模糊匹配：{len(unresolved)} 条 / 总计 {len(mb_names)}")

    # ========== 第二阶段：TF-IDF 字符 n-gram 模糊匹配（仅对未决部分） ==========
    rows = []  # Top-K 输出
    if unresolved:
        log("▶ 构建 TF-IDF（字符 2-5gram，控制特征规模）...")
        # 只针对待匹配的医保子集建模（更省内存），但字典要能兼容台湾侧文本
        mb_unres_list = mb_names.iloc[unresolved].tolist()
        vect = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2,5),
            min_df=MIN_DF,
            max_features=VOCAB_MAX
        )
        tfidf_mb = vect.fit_transform(mb_unres_list)  # CSR shape: (U, D)
        tfidf_tw = vect.transform(tw_names.tolist())  # CSR shape: (T, D)

        log("▶ 分块计算稀疏相似度并提取每行 Top-{}...".format(TOPK))
        T = tfidf_tw.shape[0]
        # 为节省内存，按 CHUNK 切分医保未决子集
        for start in range(0, tfidf_mb.shape[0], CHUNK):
            end = min(start + CHUNK, tfidf_mb.shape[0])
            log(f"  · 处理未决 {start} ~ {end-1} / {tfidf_mb.shape[0]-1}")
            X = tfidf_mb[start:end]  # (chunk, D)
            sims_sparse = cosine_similarity(X, tfidf_tw, dense_output=False)  # CSR: (chunk, T)
            for local_row in range(sims_sparse.shape[0]):
                row = sims_sparse.getrow(local_row)
                idxs, vals = row.indices, row.data
                global_mb_row = unresolved[start + local_row]
                if len(vals) == 0:
                    # 没有重叠 n-gram，记一个空候选
                    rows.append({
                        "mb_row": global_mb_row,
                        "医保名称": mb_names.iloc[global_mb_row],
                        "rank": 1,
                        "tw_icd_code": "",
                        "tw_icd_name": "",
                        "sim_score": 0.0,
                        "match_type": "tfidf_none"
                    })
                    continue
                # 取 Top-K
                if len(vals) > TOPK:
                    topk_part = np.argpartition(-vals, TOPK-1)[:TOPK]
                    order = topk_part[np.argsort(-vals[topk_part])]
                else:
                    order = np.argsort(-vals)
                for r, pos in enumerate(order, start=1):
                    j = idxs[pos]
                    rows.append({
                        "mb_row": global_mb_row,
                        "医保名称": mb_names.iloc[global_mb_row],
                        "rank": r,
                        "tw_icd_code": tw.iloc[j][tw_code_col],
                        "tw_icd_name": tw.iloc[j][tw_name_col],
                        "sim_score": float(vals[pos]),
                        "match_type": "tfidf_topk"
                    })

    # 把精确唯一命中的也加入 Top-K（rank=1，标记 exact）
    for mb_row, tw_row in mapped_idx.items():
        rows.append({
            "mb_row": mb_row,
            "医保名称": mb_names.iloc[mb_row],
            "rank": 1,
            "tw_icd_code": tw.iloc[tw_row][tw_code_col],
            "tw_icd_name": tw.iloc[tw_row][tw_name_col],
            "sim_score": 1.0,
            "match_type": "exact"
        })

    # 汇总 Top-K
    topk_df = pd.DataFrame(rows)
    if topk_df.empty:
        log("❗ 没有生成任何候选，请检查输入文件与列名识别。")
        sys.exit(1)

    # 生成 Top-1（每个 mb_row 取 rank 最小且 sim_score 最大的一行）
    top1 = (topk_df.sort_values(["mb_row","rank","sim_score"], ascending=[True, True, False])
                    .drop_duplicates(subset=["mb_row"], keep="first")
                    .rename(columns={
                        "tw_icd_code":"mapped_icd10cm",
                        "tw_icd_name":"mapped_name_tw",
                        "sim_score":"mapped_sim"
                    }))

    # 回填到医保表
    log("▶ 合并回医保表...")
    mb_out = mb.copy().reset_index(drop=True)
    mb_out["mb_row"] = mb_out.index
    mb_out = mb_out.merge(
        top1[["mb_row","mapped_icd10cm","mapped_name_tw","mapped_sim"]],
        how="left",
        on="mb_row"
    )

    # 输出文件
    log("▶ 写入结果文件...")
    OUT_MAIN.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_MAIN, engine="openpyxl") as w:
        mb_out.to_excel(w, sheet_name="映射结果_Top1", index=False)
    # TopK 排序更利于人工核查
    topk_export = (topk_df.sort_values(["mb_row","rank"])
                             .rename(columns={"tw_icd_code":"icd10cm_code",
                                              "tw_icd_name":"icd10cm_name_tw"}))
    topk_export.to_csv(OUT_TOPK, index=False, encoding="utf-8-sig")

    log("✅ 完成：")
    log(f"  Top1：{OUT_MAIN}")
    log(f"  Top5：{OUT_TOPK}")
    log("提示：如内存仍紧张，可把 CHUNK 降到 500，或将 VOCAB_MAX 降到 120000、MIN_DF 提到 3。")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"❌ 运行出错：{e}")
        sys.exit(1)

