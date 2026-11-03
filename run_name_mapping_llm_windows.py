# -*- coding: utf-8 -*-
"""
LLM 版（Windows 本地可跑）：中文医保“诊断名称” → ICD-10-CM（CCIR） + 慢性标记
- 翻译：Qwen2.5 指令模型（默认 1.8B，更省显存；可改 7B）
- 向量：BAAI/bge-m3（中英双语句向量）
- 计算：批处理 + 顶层 Top-K（不生成巨型矩阵），节省内存
- 输出：Top1 映射并合并 CCIR；Top5 候选用于人工复核

注意：
* 首次运行会从 Hugging Face 下载模型；若网络不便，可将模型名称改为本地路径或使用 ModelScope 同名模型。
* 显存吃紧可把 Qwen 改成 1.8B；7B 更准但更占显存。
"""

import os, re, sys, json, time, traceback
from pathlib import Path
import pandas as pd
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer, util

# ---------------- 用户文件路径（按你的实际路径） ----------------
P_MB   = Path(r"C:\Users\user\Desktop\icd10\ICD-10医保2.0版.xlsx")
P_CCIR = Path(r"C:\Users\user\Desktop\icd10\CCIR_v2025-1.csv")

OUT_MAIN = Path(r"C:\Users\user\Desktop\icd10\医保2.0_name_map_with_CCIR_LLM.xlsx")
OUT_TOPK = Path(r"C:\Users\user\Desktop\icd10\医保2.0_name_map_topk_LLM.csv")
LOG_PATH = Path(r"C:\Users\user\Desktop\icd10\icdmap_llm_run.log")
CACHE_TRANSL = Path(r"C:\Users\user\Desktop\icd10\cache_translate_qwen.json")

# ---------------- 模型选择（按需修改） ----------------
LLM_NAME = "deepseek-ai/deepseek-llm-7b-chat"  # 可改成 Qwen/Qwen2.5-7B-Instruct
EMB_NAME = os.environ.get("ICDMAP_EMB", "BAAI/bge-m3")

# ---------------- 超参数 ----------------
TOPK = 5
BATCH_TXT = 64        # 翻译/向量化的 batch size
BATCH_SIM = 256       # 做相似度的查询批大小
MAX_NEW_TOKENS = 64   # 翻译最大输出 tokens
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

def log(msg: str):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    text = f"[{stamp}] {msg}"
    print(text, flush=True)
    try:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception:
        pass

# ---------------- 解析 CCIR CSV ----------------
def load_ccir_table(p_csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(p_csv, header=None, dtype=str, encoding="utf-8", engine="python")
    start_idx = None
    for i in range(len(raw)):
        v = str(raw.iloc[i,0])
        if re.match(r"^'?[A-Z][0-9A-Z]{2,6}'?$", v):
            start_idx = i; break
    if start_idx is None:
        raise RuntimeError("无法定位 CCIR 数据起始行，请检查 CSV 内容/编码。")
    ccir = raw.iloc[start_idx:].copy()
    ccir.columns = ["code_raw","desc_en","ccir_flag"]
    ccir["ccir_flag"] = ccir["ccir_flag"].astype(str).str.strip().str.replace(r"[^019]", "", regex=True)
    def norm_code(x):
        if pd.isna(x): return None
        s = str(x).strip().upper().strip("'\"")
        s = re.sub(r"[^A-Z0-9\.]", "", s)
        return s.replace(".", "") or None
    ccir["_code_norm"] = ccir["code_raw"].map(norm_code)
    ccir = ccir.dropna(subset=["_code_norm"]).drop_duplicates("_code_norm")
    if ccir.empty:
        raise RuntimeError("CCIR 解析后为空，可能文件格式不对。")
    return ccir

def pick_name_col(df: pd.DataFrame) -> str:
    prefers = ["名称","诊断名称","中文名称","疾病名称","病名","诊断","name","diag_name","zh_name"]
    lower = {c.lower(): c for c in df.columns}
    for p in prefers:
        if p in df.columns: return p
        if p.lower() in lower: return lower[p.lower()]
    # fallback：中文比例最高
    def zh_ratio(s):
        if pd.isna(s): return 0.0
        s = str(s); zh = sum('\u4e00' <= ch <= '\u9fff' for ch in s)
        return zh / max(1, len(s))
    ratios = {c: df[c].astype(str).head(500).map(zh_ratio).mean() for c in df.columns}
    return max(ratios, key=ratios.get)

# ---------------- 构建翻译器（Qwen） ----------------
def build_translator():
    log(f"加载翻译模型：{LLM_NAME}")
    tok = AutoTokenizer.from_pretrained(LLM_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    return gen

def translate_batch(gen, zh_list):
    out = []
    for text in zh_list:
        prompt = f"Translate the following Chinese medical diagnosis into a concise English medical term (output English only): {text}"
        res = gen(prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)[0]["generated_text"]
        # 取最后一行；去除前缀
        lines = [l.strip() for l in res.splitlines() if l.strip()]
        ans = lines[-1]
        ans = re.sub(r"^(英文|翻译|English|Answer)[:：]\s*", "", ans, flags=re.I)
        out.append(ans)
    return out

# ---------------- 主程序 ----------------
def main():
    log("启动脚本：run_name_mapping_llm_windows.py")
    log(f"Python: {sys.executable}")
    log(f"工作目录: {Path.cwd()}")
    log(f"医保表存在: {P_MB.exists()}  路径: {P_MB}")
    log(f"CCIR表存在: {P_CCIR.exists()}  路径: {P_CCIR}")

    log("读取数据...")
    mb = pd.read_excel(P_MB, dtype=str)
    ccir = load_ccir_table(P_CCIR)
    name_col = pick_name_col(mb)
    log(f"识别医保名称列：{name_col}")

    zh_names = mb[name_col].fillna("").astype(str).tolist()
    log(f"名称条目数：{len(zh_names)}")

    # 翻译缓存
    cache = {}
    if CACHE_TRANSL.exists():
        try:
            cache = json.loads(CACHE_TRANSL.read_text(encoding="utf-8"))
            log(f"加载翻译缓存：{len(cache)} 条")
        except Exception:
            log("翻译缓存读取失败，忽略。")

    # 需要翻译的唯一短语
    uniq = list(dict.fromkeys(zh_names))
    need = [t for t in uniq if t not in cache]

    # 翻译
    if need:
        gen = build_translator()
        log(f"开始翻译 {len(need)} 条唯一中文诊断（批量 {BATCH_TXT}）...")
        for i in range(0, len(need), BATCH_TXT):
            batch = need[i:i+BATCH_TXT]
            trans = translate_batch(gen, batch)
            for k, v in zip(batch, trans):
                cache[k] = v
            # 及时落盘缓存
            CACHE_TRANSL.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            log(f"  · 翻译进度：{i+len(batch)}/{len(need)}")
    else:
        log("无需翻译，全部命中缓存。")

    en_terms = [cache.get(t, "") for t in zh_names]
    # 构造检索查询（中英拼接）
    queries = [f"{z} || {e}" if e else z for z, e in zip(zh_names, en_terms)]

    # 载入向量模型并编码
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"加载句向量模型：{EMB_NAME}  设备：{device}")
    embedder = SentenceTransformer(EMB_NAME, device=device)

    log("编码 CCIR 描述向量...")
    d_texts = ccir["desc_en"].fillna("").astype(str).tolist()
    d_vecs = embedder.encode(d_texts, normalize_embeddings=True, batch_size=BATCH_TXT, convert_to_tensor=True)

    log("编码 查询向量（中文+译文）...")
    q_vecs = embedder.encode(queries, normalize_embeddings=True, batch_size=BATCH_TXT, convert_to_tensor=True)

    # Top-K 相似度（分批以省内存）
    log(f"计算 Top-{TOPK} 相似度（批大小 {BATCH_SIM}）...")
    ccir_codes = ccir["_code_norm"].tolist()
    ccir_descs = ccir["desc_en"].tolist()
    ccir_flags = ccir["ccir_flag"].tolist()

    rows = []
    N = q_vecs.shape[0]
    for s in range(0, N, BATCH_SIM):
        e = min(s+BATCH_SIM, N)
        sim = util.cos_sim(q_vecs[s:e], d_vecs)  # (b, D)
        # 在 GPU/CPU 上 topk
        scores, idxs = torch.topk(sim, k=TOPK, dim=1)
        scores = scores.detach().cpu().numpy()
        idxs = idxs.detach().cpu().numpy()
        for i in range(e - s):
            row_id = s + i
            for r in range(TOPK):
                j = int(idxs[i, r])
                rows.append({
                    "mb_row": row_id,
                    "医保名称": zh_names[row_id],
                    "译文": en_terms[row_id],
                    "rank": r+1,
                    "icd10cm_code": ccir_codes[j],
                    "icd10cm_desc": ccir_descs[j],
                    "ccir_flag": ccir_flags[j],
                    "sim_score": float(scores[i, r])
                })
        log(f"  · 相似度批次完成：{e}/{N}")

    topk_df = pd.DataFrame(rows)
    log(f"TopK 候选行数：{len(topk_df)}")

    # 取 Top-1 映射
    top1 = (topk_df.sort_values(["mb_row","rank"])
                    .drop_duplicates(subset=["mb_row"], keep="first")
                    .rename(columns={
                        "icd10cm_code":"mapped_icd10cm",
                        "icd10cm_desc":"mapped_desc_en",
                        "ccir_flag":"mapped_ccir_flag",
                        "sim_score":"mapped_sim"
                    }))

    # 合并回医保表
    mb_out = mb.copy().reset_index(drop=True)
    mb_out["mb_row"] = mb_out.index
    mb_out = mb_out.merge(
        top1[["mb_row","mapped_icd10cm","mapped_desc_en","mapped_ccir_flag","mapped_sim"]],
        how="left",
        on="mb_row"
    )
    def ccir_label(v: str) -> str:
        return {"1":"Chronic/慢性","0":"Non-chronic/非慢性","9":"No determination/无法判定"}.get(str(v).strip(), "")
    mb_out["CCIR_label"] = mb_out["mapped_ccir_flag"].map(ccir_label)

    # 写文件
    log("写入结果文件...")
    OUT_MAIN.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_MAIN, engine="openpyxl") as w:
        mb_out.to_excel(w, sheet_name="映射结果_Top1", index=False)
    topk_df.to_csv(OUT_TOPK, index=False, encoding="utf-8-sig")
    log(f"完成：{OUT_MAIN}  和  {OUT_TOPK}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write("\n--- EXCEPTION ---\n")
            traceback.print_exc(file=f)
        print(f"❌ 运行出错：{e}", flush=True)
        sys.exit(1)
