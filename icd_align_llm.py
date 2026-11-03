# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 23:29:09 2025

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
ICD 对齐：编码前缀硬约束 + Qwen 语义重排（自动下载模型）
适配：Windows / Conda / torch 2.8
输出：1) 最终匹配(含LLM分数与理由) 2) 仅医保未匹配 3) 仅台湾未匹配
"""

import os, re, time, json, math, warnings, itertools
import pandas as pd

# ============ 路径与参数（按需修改） ============
CN_PATH   = r"C:\Users\user\Desktop\icd10\ICD-10医保2.0版.xlsx"
TW_PATH   = r"C:\Users\user\Desktop\icd10\TW_ICD10CM_2023_cn.xlsx"
OUT_DIR   = r"C:\Users\user\Desktop\icd10"
os.makedirs(OUT_DIR, exist_ok=True)

# 列名（与文件匹配）
CN_CODE_COL = "诊断代码"
CN_NAME_COL = "诊断名称"
TW_CODE_COL = "2023年版\nICD-10-CM"
TW_NAME_COL = "2023 CM中文名稱"
TW_USE_FILTER = True    # 台湾表仅保留 USE==1

# 选用 Qwen（推荐 3B）
MODEL_REPO = "Qwen/Qwen2.5-3B-Instruct"
LOCAL_DIR  = r"C:/models/Qwen2.5-3B-Instruct"  # 会自动下载到这里（可改）

# LLM 推理参数
MAX_NEW_TOKENS = 128
TEMPERATURE    = 0.2
TOP_P          = 0.9
BATCH_SIZE     = 16          # 每批评估多少对 (cn_name, tw_name)
TOPK_PER_KEY   = 5           # 每个 code_key 进入 LLM 的台湾候选上限（若>5先做轻量词匹配预筛）

# 轻量预筛（不装 rapidfuzz 也能跑）
ENABLE_LIGHT_PREFILTER = True
# ==============================================

def ts(): return time.strftime("%Y%m%d_%H%M%S")
def log(msg): print(time.strftime("[%H:%M:%S] "), msg)

def normalize_code_key(code: str) -> str:
    """
    硬约束匹配键：<首字母><三位数字>；如 A15.0 -> A150，I10 -> I100，B0.1 -> B010
    """
    if code is None: return ""
    s = str(code).strip().upper()
    if not s: return ""
    s = re.sub(r"[^A-Z0-9]", "", s)
    m = re.match(r"^([A-Z])([0-9]+)", s)
    if not m: return ""
    letter, digits = m.group(1), m.group(2)[:3]
    digits = digits.ljust(3, "0")
    return f"{letter}{digits}"

def code_prefix_2or3(code: str) -> str:
    """
    可选的更严格前缀（字母+2~3位数字），用于信息展示/诊断。
    """
    if code is None: return ""
    s = re.sub(r"[^A-Z0-9]", "", str(code).upper())
    m = re.match(r"([A-Z]\d{2,3})", s)
    return m.group(1) if m else ""

def safe_write_csv(df: pd.DataFrame, path: str, tries: int = 4, interval: float = 0.6):
    for _ in range(tries):
        try:
            if os.path.exists(path):
                try: os.remove(path)
                except: pass
            df.to_csv(path, index=False, encoding="utf-8-sig")
            return path
        except PermissionError:
            time.sleep(interval)
    base, ext = os.path.splitext(path)
    alt = f"{base}_new{ext}"
    df.to_csv(alt, index=False, encoding="utf-8-sig")
    return alt

# =============== 读取数据 ===============
log("读取医保与台湾 ICD 表 ...")
cn = pd.read_excel(CN_PATH, sheet_name=0)
tw = pd.read_excel(TW_PATH, sheet_name=0)
if TW_USE_FILTER and "USE" in tw.columns:
    tw = tw[tw["USE"] == 1].copy()

# 生成硬约束匹配键
cn["code_key"] = cn[CN_CODE_COL].map(normalize_code_key)
tw["code_key"] = tw[TW_CODE_COL].map(normalize_code_key)
cn["code_prefix"] = cn[CN_CODE_COL].map(code_prefix_2or3)
tw["code_prefix"] = tw[TW_CODE_COL].map(code_prefix_2or3)

cn_valid = cn[cn["code_key"] != ""].copy()
tw_valid = tw[tw["code_key"] != ""].copy()

# =============== 做硬约束内连接（候选池） ===============
log("按 code_key 做内连接，构造候选池 ...")
pool = cn_valid.merge(
    tw_valid[["code_key", TW_CODE_COL, TW_NAME_COL, "code_prefix"]],
    on="code_key", how="inner", suffixes=("_CN", "_TW")
)
if pool.empty:
    log("按硬约束无交集，导出未匹配清单后退出。")
    ts_now = ts()
    p1 = safe_write_csv(cn_valid[[CN_CODE_COL, CN_NAME_COL, "code_key", "code_prefix"]],
                        os.path.join(OUT_DIR, f"ICD对齐_仅医保未匹配_{ts_now}.csv"))
    p2 = safe_write_csv(tw_valid[[TW_CODE_COL, TW_NAME_COL, "code_key", "code_prefix"]],
                        os.path.join(OUT_DIR, f"ICD对齐_仅台湾未匹配_{ts_now}.csv"))
    log(f"未匹配输出：\n  {p1}\n  {p2}")
    raise SystemExit(0)

# =============== 候选轻量预筛（提高LLM效率） ===============
def light_score(a: str, b: str) -> float:
    """
    轻量级中文名相似度：基于 token 集合的 Jaccard（极快、零依赖）
    """
    a = re.sub(r"\s+", "", str(a))
    b = re.sub(r"\s+", "", str(b))
    if not a or not b: return 0.0
    sa = set(list(a))
    sb = set(list(b))
    inter = len(sa & sb)
    union = len(sa | sb) or 1
    return inter / union

def prefilter_group(df_group: pd.DataFrame) -> pd.DataFrame:
    """
    同一 code_key 内，如果候选太多，先用轻量相似度筛至 TOPK_PER_KEY。
    """
    if not ENABLE_LIGHT_PREFILTER: 
        return df_group
    if len(df_group) <= TOPK_PER_KEY:
        return df_group
    g = df_group.copy()
    g["light_sim"] = g.apply(lambda r: light_score(r[CN_NAME_COL], r[TW_NAME_COL]), axis=1)
    g = g.sort_values("light_sim", ascending=False).head(TOPK_PER_KEY)
    return g.drop(columns=["light_sim"])

log("候选轻量预筛（如组内>TOPK，先筛到 TOPK） ...")
pool = pool.groupby(["code_key", CN_CODE_COL, CN_NAME_COL], group_keys=False).apply(prefilter_group)

# =============== 自动下载并加载 Qwen ===============
warnings.filterwarnings("ignore")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

def hf_sanitize_session():
    """尽量避免代理/中文头导致的请求异常（可选）"""
    try:
        from huggingface_hub.utils._http import get_session
        s = get_session()
        s.trust_env = False
        s.proxies.clear()
        s.headers.clear()
        s.headers.update({"user-agent":"icdmap/1.0","accept":"*/*"})
    except Exception:
        pass

def download_model_if_needed(repo_id: str, local_dir: str):
    if not os.path.exists(local_dir) or len(os.listdir(local_dir)) == 0:
        log(f"下载模型：{repo_id} -> {local_dir}")
        hf_sanitize_session()
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns="**"
        )
        log("模型下载完成。")
    else:
        log(f"模型已存在：{local_dir}")

def build_llm(local_dir: str):
    download_model_if_needed(MODEL_REPO, local_dir)
    log("加载 Qwen 模型 ...")
    tok = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model.eval()
    return tok, model

tok, model = build_llm(LOCAL_DIR)

# =============== LLM 评分（组内受约束重排） ===============
SYS_PROMPT = (
    "你是医学编码匹配专家。现在给你两个中文医学诊断名称（一个来自中国医保2.0目录，一个来自台湾翻译的美国ICD-10-CM）。"
    "请判断它们是否表示相同的临床概念，并仅以JSON返回："
    '{"match": "是/否", "score": 0-100的整数, "reason": "简短中文理由"}。'
    "请勿输出除JSON以外的任何文字。"
)

def make_messages(cn_name: str, tw_name: str):
    user = f"医保诊断名称：{cn_name}\n台湾ICD名称：{tw_name}\n请仅输出JSON："
    return [{"role":"system","content":SYS_PROMPT},{"role":"user","content":user}]

def parse_score(text: str):
    """
    解析模型回答，容错处理（尽量从JSON或数字中取 score）
    """
    text = (text or "").strip()
    # 优先找 JSON
    mjson = re.search(r"\{.*\}", text, flags=re.S)
    if mjson:
        try:
            obj = json.loads(mjson.group(0))
            score = int(obj.get("score", 0))
            match = str(obj.get("match","否"))
            reason = str(obj.get("reason",""))
            score = max(0, min(100, score))
            return score, match, reason
        except Exception:
            pass
    # 次选：找数字
    mnum = re.search(r"(\d{1,3})", text)
    score = max(0, min(100, int(mnum.group(1)))) if mnum else 0
    return score, "否", ""

@torch.inference_mode()
def score_pairs(pairs):
    """
    pairs: list[(cn_name, tw_name)]
    return: list[ (score:int, match:str, reason:str, raw:str) ]
    """
    results = []
    for i in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[i:i+BATCH_SIZE]
        messages = [make_messages(cn, tw) for (cn, tw) in batch]
        # 有些 tokenizer 不支持 padding=True 的 chat_template，做两级方案
        try:
            inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", padding=True).to(model.device)
            out = model.generate(
                inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                eos_token_id=tok.eos_token_id
            )
            gen = out[:, inputs.shape[1]:] if out.shape[1] > inputs.shape[1] else out
            texts = tok.batch_decode(gen, skip_special_tokens=True)
        except TypeError:
            texts = []
            for msg in messages:
                inp = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt").to(model.device)
                o = model.generate(
                    **inp,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    eos_token_id=tok.eos_token_id
                )
                txt = tok.decode(o[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
                texts.append(txt)

        for txt in texts:
            sc, mt, rs = parse_score(txt)
            results.append((sc, mt, rs, txt))
    return results

# 逐组（同一医保代码+名称）进行 LLM 受约束重排
log("开始 LLM 受约束重排（同一 code_key 组内） ...")
rows = []
group_cols = ["code_key", CN_CODE_COL, CN_NAME_COL]
for (ck, ccode, cname), g in pool.groupby(group_cols, sort=False):
    twnames = g[TW_NAME_COL].astype(str).tolist()
    twcodes = g[TW_CODE_COL].astype(str).tolist()
    pairs   = list(zip([cname]*len(twnames), twnames))
    scored  = score_pairs(pairs)  # [(score, match, reason, raw), ...]
    # 合并结果
    g2 = g.copy()
    g2["llm_score"]  = [s[0] for s in scored]
    g2["llm_match"]  = [s[1] for s in scored]
    g2["llm_reason"] = [s[2] for s in scored]
    g2["llm_raw"]    = [s[3] for s in scored]
    # 选每个(医保代码+名称)的最佳台湾候选
    best = g2.sort_values(["llm_score"], ascending=False).groupby(level=0).head(1)
    rows.append(best)

ranked = pd.concat(rows, ignore_index=True)

# 组织输出
result = ranked[[
    "code_key",
    CN_CODE_COL, CN_NAME_COL, "code_prefix",
    TW_CODE_COL, TW_NAME_COL, # 台湾侧列
    "llm_score","llm_match","llm_reason"
]].rename(columns={
    "code_key":"匹配键(首字母+后三位)",
    CN_CODE_COL:"医保_诊断代码",
    CN_NAME_COL:"医保_诊断名称",
    "code_prefix":"医保_编码前缀(2-3位)",
    TW_CODE_COL:"台湾_2023ICD10CM",
    TW_NAME_COL:"台湾_中文名称",
    "llm_score":"LLM分(0-100)",
    "llm_match":"LLM判断(是/否)",
    "llm_reason":"LLM理由"
})

# 未匹配（按 code_key 不在交集）
unmatched_cn = cn_valid[~cn_valid["code_key"].isin(tw_valid["code_key"])][["code_key", "code_prefix", CN_CODE_COL, CN_NAME_COL]] \
               .rename(columns={"code_key":"匹配键(首字母+后三位)","code_prefix":"医保_编码前缀(2-3位)", CN_CODE_COL:"医保_诊断代码", CN_NAME_COL:"医保_诊断名称"})
unmatched_tw = tw_valid[~tw_valid["code_key"].isin(cn_valid["code_key"])][["code_key", "code_prefix", TW_CODE_COL, TW_NAME_COL]] \
               .rename(columns={"code_key":"匹配键(首字母+后三位)","code_prefix":"台湾_编码前缀(2-3位)", TW_CODE_COL:"台湾_2023ICD10CM", TW_NAME_COL:"台湾_中文名称"})

# 写文件
ts_now = ts()
p1 = safe_write_csv(result,      os.path.join(OUT_DIR, f"ICD对齐_编码前缀+Qwen重排_{ts_now}.csv"))
p2 = safe_write_csv(unmatched_cn,os.path.join(OUT_DIR, f"ICD对齐_仅医保未匹配_{ts_now}.csv"))
p3 = safe_write_csv(unmatched_tw,os.path.join(OUT_DIR, f"ICD对齐_仅台湾未匹配_{ts_now}.csv"))

log("完成 ✅")
log(f"结果：{p1}\n未匹配-医保：{p2}\n未匹配-台湾：{p3}")
