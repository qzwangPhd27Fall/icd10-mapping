# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 16:04:07 2025

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
ICD-10 医保2.0 ↔ 台湾 ICD10-CM 对齐系统 (支持断点续跑 + 进度条 + Excel 增量保存)

作者：ChatGPT（为用户定制）
运行环境：Python 3.12 + PyTorch 2.9 + CUDA 12.8
"""

import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# ================= 参数设置 =================

CN_FILE = "ICD-10医保2.0版.xlsx"
TW_FILE = "TW_ICD10CM_2023_cn.xlsx"

OUTPUT_PROGRESS = "output_progress.xlsx"
OUTPUT_FINAL = "final_mapping.xlsx"

EMB_MODEL_DIR = "C:/models/bge-base-zh-v1.5"
LLM_MODEL_DIR = "C:/models/Qwen2.5-3B-Instruct"

TOPK = 5          # BGE向量召回的候选数
BATCH_SIZE = 100  # 每100条医保诊断：跑LLM + 保存 + 刷新进度

# =============== 工具函数 ===============

def now():
    return datetime.now().strftime("%H:%M:%S")

def load_data():
    print(f"[{now()}] 读取两张表 …")
    cn = pd.read_excel(CN_FILE)
    tw = pd.read_excel(TW_FILE)
    
    print(f"[{now()}] ✅ 台湾表：{len(tw)} 行，医保表：{len(cn)} 行")

    # 取医保/台湾的编码和名称列
    cn_code_col = "诊断代码"
    cn_name_col = "诊断名称"
    tw_code_col = "2023年版\nICD-10-CM"
    tw_name_col = "2023 CM中文名稱"

    cn = cn[[cn_code_col, cn_name_col]].dropna().unique()
    tw = tw[[tw_code_col, tw_name_col]].dropna().unique()

    cn.rename(columns={cn_code_col: "cn_code", cn_name_col: "cn_name"}, inplace=True)
    tw.rename(columns={tw_code_col: "tw_code", tw_name_col: "tw_name"}, inplace=True)

    cn["code_key"] = cn["cn_code"].str[:4]
    tw["code_key"] = tw["tw_code"].str[:4]

    return cn, tw

# =============== 加载Embedding模型 ===============

class Embedder:
    def __init__(self, model_dir):
        from sentence_transformers import SentenceTransformer
        print(f"[{now()}] 加载向量模型 …")
        self.model = SentenceTransformer(model_dir, device="cuda")

    def encode(self, texts):
        return self.model.encode(texts, convert_to_tensor=True, device="cuda", batch_size=64)

# =============== 加载LLM模型 ===============

def build_llm():
    print(f"[{now()}] 加载 Qwen2.5-3B …")
    tok = AutoTokenizer.from_pretrained(LLM_MODEL_DIR, trust_remote_code=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_DIR,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto"
    )
    return tok, model


# ================= 规范化 & 工具 =================
import re
import torch
from torch.nn import functional as F

def norm_text(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = x.strip().replace("\u3000", " ")
    # 全角转半角
    res = []
    for ch in x:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            res.append(chr(code - 0xFEE0))
        elif code == 0x3000:
            res.append(" ")
        else:
            res.append(ch)
    x = "".join(res)
    # 去多余空白
    x = re.sub(r"\s+", " ", x)
    return x

_code_key_pat = re.compile(r"([A-Za-z])[^0-9]*([0-9])[^0-9]*([0-9])[^0-9]*([0-9])")

def norm_code_key(code: str) -> str:
    """
    提取“首字母 + 后续三位数字”的 key，用于首字符编码约束。
    示例：I21.0 -> I210； E11 -> E110； A09.X -> A090
    """
    if not isinstance(code, str):
        return ""
    s = re.sub(r"[^A-Za-z0-9]", "", code.upper())
    m = _code_key_pat.search(s)
    if not m:
        return ""
    return f"{m.group(1)}{m.group(2)}{m.group(3)}{m.group(4)}"  # 例如 I210

def build_prompt(cn_name: str, tw_name: str) -> str:
    """
    极简判定提示：只要回答 YES 或 NO
    """
    return (
        "你是医学编码比对助手。判断两条中文诊断名称是否表示“同一疾病/病种”，"
        "若等价或高度同义，输出：YES；否则输出：NO。\n\n"
        f"医保诊断：{cn_name}\n"
        f"台湾ICD名称：{tw_name}\n"
        "只输出 YES 或 NO。"
    )

@torch.no_grad()
def llm_batch_yesno(tokenizer, model, pairs, max_new_tokens=2):
    """
    pairs: List[(cn_name, tw_name)]
    返回：List[int]，YES=1, NO=0
    """
    prompts = [build_prompt(c, t) for c, t in pairs]
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=384
    )
    # 左填充已在 tokenizer 初始化设置
    enc = {k: v.to(model.device) for k, v in enc.items()}
    gen_out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        eos_token_id=tokenizer.eos_token_id,
    )
    # 解码“新增的”tokens
    # 获取每条样本的新增片段（右侧）
    results = []
    input_lens = enc["input_ids"].shape[1]
    for i in range(gen_out.size(0)):
        new_tokens = gen_out[i, input_lens:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().upper()
        if "YES" in text and "NO" in text:
            # 优先取首个明确词
            pos_yes = text.find("YES")
            pos_no = text.find("NO")
            results.append(1 if pos_yes != -1 and (pos_yes < pos_no or pos_no == -1) else 0)
        elif "YES" in text:
            results.append(1)
        elif "NO" in text:
            results.append(0)
        else:
            # 模糊输出，保守判 NO
            results.append(0)
    return results

def ensure_xlsx_utf8(writer):
    # openpyxl 默认即为合适格式，这里留作占位；主要保证 index=False
    pass

# ================= 候选召回（向量 + 规则） =================
def cosine_topk(query_vec, cand_mat, k=TOPK):
    # query_vec: (D,), cand_mat: (N, D)
    q = F.normalize(query_vec.unsqueeze(0), dim=1)     # (1, D)
    c = F.normalize(cand_mat, dim=1)                   # (N, D)
    sim = (q @ c.T).squeeze(0)                         # (N,)
    val, idx = torch.topk(sim, k=min(k, sim.numel()))
    return idx.cpu().tolist(), val.cpu().tolist()

def load_data():
    print(f"[{now()}] 读取两张表 …")
    cn = pd.read_excel(CN_FILE)
    tw = pd.read_excel(TW_FILE)

    # 识别列名（已根据你提供的截图）
    cn_code_col = "诊断代码"
    cn_name_col = "诊断名称"
    tw_code_col = "2023年版\nICD-10-CM"
    tw_name_col = "2023 CM中文名稱"

    need_cols_cn = [cn_code_col, cn_name_col]
    need_cols_tw = [tw_code_col, tw_name_col]
    for c in need_cols_cn:
        if c not in cn.columns:
            raise ValueError(f"医保表缺列：{c}；实际列：{list(cn.columns)}")
    for c in need_cols_tw:
        if c not in tw.columns:
            raise ValueError(f"台湾表缺列：{c}；实际列：{list(tw.columns)}")

    cn = cn[need_cols_cn].dropna().drop_duplicates()
    tw = tw[need_cols_tw].dropna().drop_duplicates()

    # 规范化
    cn["cn_name"] = cn[cn_name_col].map(norm_text)
    cn["cn_code"] = cn[cn_code_col].astype(str).str.upper()
    tw["tw_name"] = tw[tw_name_col].map(norm_text)
    tw["tw_code"] = tw[tw_code_col].astype(str).str.upper()

    # 首字符 + 三位数字 约束 key
    cn["code_key"] = cn["cn_code"].map(norm_code_key)
    tw["code_key"] = tw["tw_code"].map(norm_code_key)

    # 过滤无 key 的行
    cn = cn[cn["code_key"] != ""].reset_index(drop=True)
    tw = tw[tw["code_key"] != ""].reset_index(drop=True)

    print(f"[{now()}] ✅ 台湾表：{len(tw)} 行，医保表：{len(cn)} 行")
    return cn[["cn_code", "cn_name", "code_key"]], tw[["tw_code", "tw_name", "code_key"]]

class Embedder:
    def __init__(self, model_dir):
        from sentence_transformers import SentenceTransformer
        print(f"[{now()}] 加载向量模型 …")
        self.model = SentenceTransformer(model_dir, device="cuda")

    def encode(self, texts, batch_size=128):
        return self.model.encode(
            texts,
            convert_to_tensor=True,
            device="cuda",
            batch_size=batch_size,
            show_progress_bar=False
        )

def build_llm():
    print(f"[{now()}] 加载 Qwen2.5-3B-Instruct …")
    tok = AutoTokenizer.from_pretrained(
        LLM_MODEL_DIR, trust_remote_code=True, padding_side="left"
    )
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_DIR,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto"
    )
    return tok, model

# =============== 断点续跑 ===============
def load_progress():
    if os.path.exists(OUTPUT_PROGRESS):
        try:
            dfp = pd.read_excel(OUTPUT_PROGRESS, engine="openpyxl")
            # 确保列存在
            must = ["cn_code", "cn_name", "tw_code", "tw_name", "sim", "judge"]
            for c in must:
                if c not in dfp.columns:
                    raise ValueError("progress表结构不完整")
            print(f"[{now()}] ♻️ 发现断点文件：{OUTPUT_PROGRESS}，已有 {len(dfp)} 行")
            return dfp
        except Exception:
            print(f"[{now()}] ⚠ progress 文件损坏或结构不符，将重新开始")
            return pd.DataFrame(columns=["cn_code","cn_name","tw_code","tw_name","sim","judge"])
    else:
        return pd.DataFrame(columns=["cn_code","cn_name","tw_code","tw_name","sim","judge"])

def append_progress(rows):
    # rows: List[dict]
    mode = "a" if os.path.exists(OUTPUT_PROGRESS) else "w"
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(OUTPUT_PROGRESS, engine="openpyxl", mode=mode, if_sheet_exists="overlay") as w:
        # 写入一个 Sheet（单表），采用追加行（保留表头一次）
        if mode == "w":
            df.to_excel(w, index=False, sheet_name="progress")
        else:
            # 读现有，再写回（openpyxl 追加行相对麻烦，这里合并写回更稳）
            old = pd.read_excel(OUTPUT_PROGRESS, engine="openpyxl")
            out = pd.concat([old, df], ignore_index=True)
            out.to_excel(w, index=False, sheet_name="progress")

def save_final(df_all):
    with pd.ExcelWriter(OUTPUT_FINAL, engine="openpyxl", mode="w") as w:
        df_all.to_excel(w, index=False, sheet_name="mapping")
    print(f"[{now()}] ✅ 已导出最终结果：{OUTPUT_FINAL}")

# =============== 主流程 ===============
def main():
    cn, tw = load_data()

    # 载入 progress
    progress = load_progress()
    done_keys = set(zip(progress["cn_code"], progress["tw_code"]))  # 已保存的 pair
    done_cn = set(progress["cn_code"])  # 已覆盖的医保诊断（简单近似）

    # 建立 TW 名称向量缓存：按 code_key 聚类，减少编码量
    emb = Embedder(EMB_MODEL_DIR)
    tw_groups = {}
    tw_groups_vec = {}
    print(f"[{now()}] 为每个 code_key 预编码台湾名称向量 …")
    for k, g in tw.groupby("code_key", sort=False):
        names = g["tw_name"].tolist()
        vecs = emb.encode(names)  # Tensor (N, D)
        tw_groups[k] = g.reset_index(drop=True)
        tw_groups_vec[k] = vecs

    # 过滤已完成的 cn
    cn_todo = cn[~cn["cn_code"].isin(done_cn)].reset_index(drop=True)
    total = len(cn)
    todo = len(cn_todo)
    print(f"[{now()}] 待处理医保诊断：{todo} / {total}")

    # 加载 LLM（一次）
    tok, llm = build_llm()

    rows_to_save = []
    last_save_time = time.time()
    start_time = time.time()

    pbar = tqdm(total=todo, desc="LLM 判定中", unit="cn")
    processed = 0

    for _, row in cn_todo.iterrows():
        cn_code = row["cn_code"]
        cn_name = row["cn_name"]
        key = row["code_key"]

        # 规则预筛：必须在同一 code_key 的台湾候选池
        if key not in tw_groups:
            # 无候选，写空
            rows_to_save.append({
                "cn_code": cn_code, "cn_name": cn_name,
                "tw_code": "", "tw_name": "",
                "sim": np.nan, "judge": 0
            })
            processed += 1
            pbar.update(1)
        else:
            # 向量召回 TopK
            tw_g = tw_groups[key]
            tw_vec = tw_groups_vec[key]
            qv = emb.encode([cn_name])[0]  # (D,)
            idxs, sims = cosine_topk(qv, tw_vec, k=TOPK)

            # 对候选做 LLM 判定
            pairs = [(cn_name, tw_g.loc[i, "tw_name"]) for i in idxs]
            judge_yes = llm_batch_yesno(tok, llm, pairs, max_new_tokens=2)  # List[int]

            # 取第一个 YES，否则取相似度最高的 NO 作为兜底
            chosen = None
            for rank, yes in enumerate(judge_yes):
                i = idxs[rank]
                if yes == 1:
                    chosen = (tw_g.loc[i, "tw_code"], tw_g.loc[i, "tw_name"], sims[rank], 1)
                    break
            if chosen is None:
                i0 = idxs[0]
                chosen = (tw_g.loc[i0, "tw_code"], tw_g.loc[i0, "tw_name"], sims[0], 0)

            tw_code, tw_name, sim, judge = chosen
            rows_to_save.append({
                "cn_code": cn_code, "cn_name": cn_name,
                "tw_code": tw_code, "tw_name": tw_name,
                "sim": float(sim), "judge": int(judge)
            })
            processed += 1
            pbar.update(1)

        # 每 BATCH_SIZE 条增量保存一次（Excel 断点续跑）
        if processed % BATCH_SIZE == 0 or (time.time() - last_save_time) > 180:
            append_progress(rows_to_save)
            rows_to_save.clear()
            last_save_time = time.time()

            # 进度与 ETA
            elapsed = time.time() - start_time
            done_now = processed
            per_item = elapsed / max(done_now, 1)
            remain = todo - done_now
            eta = remain * per_item
            pbar.set_postfix({
                "done": f"{done_now}/{todo}",
                "elapsed_min": f"{elapsed/60:.1f}",
                "eta_min": f"{eta/60:.1f}"
            })

    pbar.close()

    # 收尾保存
    if rows_to_save:
        append_progress(rows_to_save)
        rows_to_save.clear()

    # 汇总最终结果（progress 全量去重后的最新结果）
    final_df = pd.read_excel(OUTPUT_PROGRESS, engine="openpyxl")
    # 对同一个 cn_code，优先保留 judge=1 的记录；否则保留 sim 最大的记录
    final_df["sim"] = pd.to_numeric(final_df["sim"], errors="coerce")
    final_df["judge"] = pd.to_numeric(final_df["judge"], errors="coerce").fillna(0).astype(int)

    final_df.sort_values(["cn_code", "judge", "sim"], ascending=[True, False, False], inplace=True)
    final_best = final_df.groupby("cn_code", as_index=False).first()

    save_final(final_best)
    print(f"[{now()}] 🎉 全部完成！")

# 入口
if __name__ == "__main__":
    main()











