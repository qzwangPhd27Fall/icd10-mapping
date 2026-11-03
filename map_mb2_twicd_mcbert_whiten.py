# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 04:12:36 2025

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
医保2.0（中文） ↔ 台湾中文版 ICD-10-CM（中文）
BERT 语义对齐 + 白化（Whitening）+ 外因章约束 版本（safetensors 友好）

改进点：
  - TF-IDF 候选：分块 + 稀疏余弦 + 局部Top-K（避免巨大dense矩阵）
  - 句向量：BERT（默认 bert-base-chinese，use_safetensors=True，后续可替换为本地MacBERT_safetensors）
  - 白化：联合语料拟合 Σ，支持 Σ^{-1/2}（WHITEN_POWER=0.5）或更强
  - 重排：哈希索引 (name,code)->idx，避免 DataFrame 慢查询；每1000行打印进度
  - 约束：对 V/W/X/Y/Z/T 章进行惩罚（非硬过滤），降低非伤害诊断的外因误配
  - 产出：Top-5（含TF-IDF与白化相似度）CSV + Top-1 Excel；详尽日志
"""

import os
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import torch
from transformers import BertTokenizer, BertModel

# ========================= 可 配 置 =========================
# 输入/输出默认路径（可用命令行 --mb/--tw/--outdir 覆盖）
P_MB_DEFAULT = r"C:\Users\user\Desktop\icd10\ICD-10医保2.0版.xlsx"
P_TW_DEFAULT = r"C:\Users\user\Desktop\icd10\TW_ICD10CM_2023_cn.xlsx"
OUT_DIR_DEFAULT = r"C:\Users\user\Desktop\icd10"

# 模型（默认走 safetensors 友好模型；若换本地 MacBERT safetensors，运行时 --model 指到目录）
MODEL_NAME = "bert-base-chinese"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TF-IDF 候选参数
TOPK = 5                 # 最终输出的Top-K（也用于Top5表）
CAND_TOPK = 8            # TF-IDF 初筛候选数（>=TOPK，确保召回）
CHUNK = 2000             # 分块大小（根据内存可 1500~4000）
VOCAB_MAX = 150_000
MIN_DF = 3               # 高频降噪（2~3 均可）

# BERT 编码参数
MAX_LEN = 64
BATCH_SIZE = 64          # 显存紧张可 32

# 白化参数
WHITEN_POWER = 0.5       # 0.5=Σ^{-1/2}（常用）；1.0更激进
EPS = 1e-9

# 外因章惩罚
EXTERNAL_CHAPTERS = set(list("VWXYZT"))
PENALTY_EXT = 0.08

# 伤害/外因提示词（命中则不惩罚外因章）
INJURY_HINTS = [
    "损伤","创伤","骨折","烧伤","中毒","窒息","坠落","交通事故",
    "外因","异物","扭伤","挫伤","咬伤","刺伤","撞击","碾压","溺水",
    "火焰","化学品","毒物","刀割","枪伤","坠物","跌倒","自杀","攻击"
]

# 是否对台湾侧文本注入编码碎片（增强语义）
INJECT_CODE = True
INJECT_PARTS = True

# ========================= 日 志 =========================
def make_logger(log_path: Path):
    def _log(msg: str):
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        text = f"[{stamp}] {msg}"
        print(text, flush=True)
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(text + "\n")
        except Exception:
            pass
    return _log

# ========================= 工 具 函 数 =========================
def norm_zh(s: str) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s).strip()
    s = s.replace("\u3000", " ")
    s = re.sub(r"[（）\(\)]", "()", s)
    s = s.replace("－","-").replace("—","-").replace("～","~").replace("·","·")
    s = re.sub(r"\s+", " ", s)
    # 允许中英数/括号/点/加号/星号/空格/常见标点
    s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff\-\.\(\)·\s\+\*，,；;：:]", "", s)
    return s

def pick_name_col(df: pd.DataFrame) -> str:
    prefers = ["诊断名称","中文名称","名称","疾病名称","病名","诊断","zh_name","name","desc","描述"]
    for p in prefers:
        if p in df.columns: return p
    # 退化：中文占比最高的列
    def zh_ratio(s):
        s = str(s) if pd.notna(s) else ""
        zh = sum('\u4e00' <= ch <= '\u9fff' for ch in s)
        return zh / max(1, len(s))
    ratios = {c: df[c].astype(str).head(300).map(zh_ratio).mean() for c in df.columns}
    return max(ratios, key=ratios.get)

def pick_code_col(df: pd.DataFrame) -> str:
    prefers = ["ICD10","ICD-10","ICD10编码","ICD编码","编码","代碼","代码","編碼","code","icd_code"]
    for p in prefers:
        if p in df.columns: return p
    return df.columns[0]

def is_external_chapter(code: str) -> bool:
    if not code: return False
    return str(code).strip().upper()[:1] in EXTERNAL_CHAPTERS

def contains_injury_hint(text: str) -> bool:
    txt = text or ""
    return any(k in txt for k in INJURY_HINTS)

def inject_text(name: str, code: str) -> str:
    """对台湾侧文本注入编码碎片。"""
    if not INJECT_CODE or not code:
        return name
    code = str(code)
    parts = [code]
    if INJECT_PARTS:
        c = code.replace("．", ".").replace("＋", "+").replace("＊","*")
        c = c.replace("，", ",").replace("；",";").replace("：",":")
        c = re.sub(r"\s+", "", c)
        for seg in re.split(r"[\+\s]", c):
            seg = seg.strip()
            if not seg: continue
            parts.append(seg)
            m = re.match(r"([A-Za-z])(\d{2})(?:\.(\d+))?", seg)
            if m:
                ch = m.group(1); two = m.group(2); tail = m.group(3)
                parts.append(ch); parts.append(f"{ch}{two}")
                if tail: parts.append(f"{ch}{two}.{tail}")
    unique = list(dict.fromkeys([p for p in parts if p]))
    return f"{name}（编码提示：{'，'.join(unique)}）"

# ========================= BERT & 白 化 =========================
class SentenceEncoder:
    def __init__(self, model_name=MODEL_NAME, device=DEVICE, max_len=MAX_LEN, batch_size=BATCH_SIZE):
        # 强制使用 safetensors 以兼容 torch<2.6 的安全限制；若你装了 torch≥2.6 正式版，这样写也没问题
        self.tok = BertTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = BertModel.from_pretrained(model_name, use_safetensors=True).to(device)
        self.model.eval()
        self.device = device
        self.max_len = max_len
        self.batch = batch_size

    @torch.no_grad()
    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), self.batch):
            batch = texts[i:i+self.batch]
            inputs = self.tok(batch, padding=True, truncation=True,
                              max_length=self.max_len, return_tensors="pt").to(self.device)
            out = self.model(**inputs).last_hidden_state[:, 0, :]   # [CLS]
            vecs.append(out.cpu().numpy())
        if vecs:
            return np.vstack(vecs)
        d = self.model.config.hidden_size
        return np.zeros((0, d), dtype=np.float32)

def fit_whitening(X: np.ndarray, power: float = WHITEN_POWER, eps: float = EPS) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    cov = (Xc.T @ Xc) / max(1, Xc.shape[0])
    U, S, _ = np.linalg.svd(cov, full_matrices=False)
    S_inv = np.power(S + eps, -power)
    W = (U * S_inv) @ U.T
    return mu, W

def apply_whitening(X: np.ndarray, mu: np.ndarray, W: np.ndarray) -> np.ndarray:
    Xw = (X - mu) @ W
    n = np.linalg.norm(Xw, axis=1, keepdims=True) + 1e-12
    return Xw / n

# ========================= 主 过 程 =========================
def main():
    import argparse
    from sklearn.preprocessing import normalize

    parser = argparse.ArgumentParser()
    parser.add_argument("--mb", type=str, default=P_MB_DEFAULT, help="医保2.0 Excel 路径")
    parser.add_argument("--tw", type=str, default=P_TW_DEFAULT, help="台湾ICD-10-CM Excel 路径")
    parser.add_argument("--outdir", type=str, default=OUT_DIR_DEFAULT, help="输出目录")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="BERT模型名或本地目录（safetensors）")
    parser.add_argument("--topk", type=int, default=TOPK, help="最终Top-K输出（默认5）")
    parser.add_argument("--candk", type=int, default=CAND_TOPK, help="TF-IDF候选数量（默认8）")
    args = parser.parse_args()

    out_dir = Path(args.outdir); out_dir.mkdir(parents=True, exist_ok=True)
    LOG_PATH = out_dir / "map_mb2_twicd_mcbert_whiten.log"
    log = make_logger(LOG_PATH)

    P_MB = Path(args.mb); P_TW = Path(args.tw)
    OUT_TOPK = out_dir / "医保2.0_对齐台湾ICD10CM_Top5_MCBERT.csv"
    OUT_MAIN = out_dir / "医保2.0_对齐台湾ICD10CM_Top1_MCBERT.xlsx"

    log("启动：医保2.0 ↔ 台湾ICD-10-CM（BERT + Whitening）")
    log(f"Python: {sys.executable}")
    log(f"模型: {args.model} | 设备: {DEVICE}")
    log(f"输入存在：医保={P_MB.exists()}，台湾={P_TW.exists()}")
    if not P_MB.exists() or not P_TW.exists():
        log("❌ 输入文件不存在，请检查路径"); sys.exit(2)

    # 读取原表
    mb_df = pd.read_excel(P_MB, dtype=str)
    tw_df = pd.read_excel(P_TW, dtype=str)

    mb_name_col = pick_name_col(mb_df)
    tw_name_col = pick_name_col(tw_df)
    tw_code_col = pick_code_col(tw_df)
    log(f"✓ 医保名称列：{mb_name_col}")
    log(f"✓ 台湾名称列：{tw_name_col}，编码列：{tw_code_col}")

    mb_names_raw = mb_df[mb_name_col].fillna("").astype(str).map(norm_zh).tolist()
    tw_names_raw = tw_df[tw_name_col].fillna("").astype(str).map(norm_zh).tolist()
    tw_codes = tw_df[tw_code_col].fillna("").astype(str).tolist()

    # ========= 1) 精确匹配 =========
    tw_norm_to_idx = {}
    for j, nm in enumerate(tw_names_raw):
        tw_norm_to_idx.setdefault(nm, []).append(j)

    exact_rows = []
    unresolved_idx = []
    for i, nm in enumerate(mb_names_raw):
        hits = tw_norm_to_idx.get(nm, [])
        if len(hits) == 1:
            j = hits[0]
            exact_rows.append({
                "mb_row": i, "医保名称": nm, "rank": 1,
                "tw_icd_code": tw_codes[j], "tw_icd_name": tw_names_raw[j],
                "sim_tfidf": 1.0, "match_type": "exact"
            })
        else:
            unresolved_idx.append(i)
    log(f"▶ 精确唯一命中：{len(exact_rows)} 条；待模糊：{len(unresolved_idx)} / {len(mb_names_raw)}")

    # ========= 2) TF-IDF 候选（分块稀疏Top-K） =========
    rows = list(exact_rows)
    if unresolved_idx:
        log("▶ TF-IDF 候选生成（char 2~5-gram，分块相似度）")
        vect = TfidfVectorizer(
            analyzer="char", ngram_range=(2,5),
            min_df=MIN_DF, max_features=VOCAB_MAX, dtype=np.float32
        )
        X_tw = vect.fit_transform(tw_names_raw).astype(np.float32)
        X_mb = vect.transform([mb_names_raw[i] for i in unresolved_idx]).astype(np.float32)

        X_tw = normalize(X_tw, copy=False)
        X_mb = normalize(X_mb, copy=False)

        CAND = max(args.candk, args.topk)
        total = X_mb.shape[0]
        for start in range(0, total, CHUNK):
            end = min(start + CHUNK, total)
            log(f"  · 相似度分块 {start} ~ {end-1} / {total-1}")
            sims_sparse = cosine_similarity(X_mb[start:end], X_tw, dense_output=False)  # CSR

            for local_row in range(sims_sparse.shape[0]):
                row = sims_sparse.getrow(local_row)
                idxs, vals = row.indices, row.data
                global_mb_row = unresolved_idx[start + local_row]

                if len(vals) == 0:
                    rows.append({
                        "mb_row": global_mb_row, "医保名称": mb_names_raw[global_mb_row],
                        "rank": 1, "tw_icd_code": "", "tw_icd_name": "",
                        "sim_tfidf": 0.0, "match_type": "tfidf_none"
                    })
                    continue

                if len(vals) > CAND:
                    topk_part = np.argpartition(-vals, CAND-1)[:CAND]
                    order = topk_part[np.argsort(-vals[topk_part])]
                else:
                    order = np.argsort(-vals)

                for r, pos in enumerate(order, start=1):
                    j = idxs[pos]
                    rows.append({
                        "mb_row": global_mb_row, "医保名称": mb_names_raw[global_mb_row],
                        "rank": r,
                        "tw_icd_code": tw_codes[j], "tw_icd_name": tw_names_raw[j],
                        "sim_tfidf": float(vals[pos]), "match_type": "tfidf_topk"
                    })

    topk_df = pd.DataFrame(rows)
    if topk_df.empty:
        log("❗ 未产生任何候选，检查列识别/输入表"); sys.exit(1)

    topk_export = (topk_df.sort_values(["mb_row","rank"])
                             .rename(columns={"tw_icd_code":"icd10cm_code",
                                              "tw_icd_name":"icd10cm_name_tw"}))
    OUT_TOPK.parent.mkdir(parents=True, exist_ok=True)
    topk_export.to_csv(OUT_TOPK, index=False, encoding="utf-8-sig")
    log(f"✓ 已写出 TF-IDF 候选（最多 {args.candk}）：{OUT_TOPK}")

    # ========= 3) BERT 编码 + 白化拟合 =========
    log("▶ MC-BERT 编码 + 白化拟合")
    enc = SentenceEncoder(model_name=args.model, device=DEVICE)

    # 准备联合语料（台湾侧注入编码碎片增强语义；医保侧不注入）
    mb_corpus = mb_names_raw
    tw_corpus = [inject_text(nm, cd) for nm, cd in zip(tw_names_raw, tw_codes)]
    union_corpus = mb_corpus + tw_corpus

    Z_union = enc.encode(union_corpus)
    mu, W = fit_whitening(Z_union, power=WHITEN_POWER, eps=EPS)
    Z_mb = apply_whitening(Z_union[:len(mb_corpus)], mu, W)
    Z_tw = apply_whitening(Z_union[len(mb_corpus):], mu, W)  # 与 tw_df 行对齐

    # ========= 4) 候选内语义重排（白化 + 约束） =========
    log("▶ 候选内语义重排（白化 + 约束）")

    # (name,code)->idx 与 name->first 索引，加速定位
    name_code_to_idx: Dict[Tuple[str,str], int] = {}
    name_to_first_idx: Dict[str, int] = {}
    for j, (nm, cd) in enumerate(zip(tw_names_raw, tw_codes)):
        key = (str(nm), str(cd))
        if key not in name_code_to_idx:
            name_code_to_idx[key] = j
        if nm not in name_to_first_idx:
            name_to_first_idx[nm] = j

    final_rows = []
    topk_scored = []

    grouped = topk_export.groupby("mb_row", sort=True)
    total_groups = len(grouped)
    processed = 0

    for i, group in grouped:
        processed += 1
        if processed % 1000 == 0:
            log(f"  · 语义重排进度 {processed} / {total_groups}")

        i = int(i)
        query = mb_names_raw[i]
        qv = Z_mb[i:i+1]  # (1, d)

        # 限制到 Top-K
        cands = group.sort_values("rank").head(args.topk if args.topk > 0 else TOPK)

        # 收集候选索引
        tw_idxs, names, codes, tfidf_vals = [], [], [], []
        for _, r in cands.iterrows():
            name = str(r["icd10cm_name_tw"])
            code = str(r["icd10cm_code"])
            j = name_code_to_idx.get((name, code))
            if j is None:
                j = name_to_first_idx.get(name)
            if j is None:
                continue
            tw_idxs.append(j); names.append(name); codes.append(code)
            tfidf_vals.append(float(r["sim_tfidf"]) if "sim_tfidf" in r and pd.notna(r["sim_tfidf"]) else np.nan)

        if not tw_idxs:
            final_rows.append({
                "mb_row": i, "医保名称": query,
                "mapped_icd10cm": "", "mapped_name_tw": "",
                "mcbert_score": 0.0, "remark": "no_candidate"
            })
            continue

        C = Z_tw[tw_idxs, :]                   # (k, d)
        sims = (qv @ C.T).flatten()            # 余弦（白化后已L2）

        if not contains_injury_hint(query):
            for t, code in enumerate(codes):
                if is_external_chapter(code):
                    sims[t] -= PENALTY_EXT

        best = int(np.argmax(sims))
        final_rows.append({
            "mb_row": i,
            "医保名称": query,
            "mapped_icd10cm": codes[best],
            "mapped_name_tw": names[best],
            "mcbert_score": float(sims[best]),
            "remark": "ok"
        })

        # 同时保存每条候选的语义分数（Top-5 CSV 用）
        for rk, (n, c, s, tfv) in enumerate(zip(names, codes, sims, tfidf_vals), start=1):
            topk_scored.append({
                "mb_row": i,
                "医保名称": query,
                "rank": rk,
                "icd10cm_code": c,
                "icd10cm_name_tw": n,
                "sim_tfidf": tfv,
                "sim_mcbert_whiten": float(s)
            })

    # ========= 5) 写出结果 =========
    final_df = pd.DataFrame(final_rows).sort_values("mb_row")

    mb_out = mb_df.copy().reset_index(drop=True)
    mb_out["mb_row"] = mb_out.index
    mb_out = mb_out.merge(final_df, how="left", on="mb_row")

    topk_scored_df = pd.DataFrame(topk_scored).sort_values(["mb_row","rank"])
    topk_scored_df.to_csv(OUT_TOPK, index=False, encoding="utf-8-sig")
    log(f"✓ 已更新 Top-5 候选（含MC-BERT白化相似度）：{OUT_TOPK}")

    with pd.ExcelWriter(OUT_MAIN, engine="openpyxl") as w:
        mb_out.to_excel(w, sheet_name="映射结果_Top1_MCBERT", index=False)
    log(f"✅ 完成！Top-1 结果输出：{OUT_MAIN}")
    log("提示：若需更强约束，可将外因章从“惩罚”改为“硬过滤”（判断后 continue 跳过）。")

if __name__ == "__main__":
    main()

