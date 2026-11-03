# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 23:56:39 2025

@author: Administrator
"""

import pandas as pd
import numpy as np

# === 1. 载入数据 ===
path = r"C:\Users\user\Desktop\icd10\output_icd_pairs_cleaned.xlsx"
df = pd.read_csv(path, dtype=str)

# 确保 sim 为浮点
if "sim" in df.columns:
    df["sim"] = pd.to_numeric(df["sim"], errors="coerce")

# === 2. 删除父层 ICD ===
if "tw_code" in df.columns:
    before = len(df)
    df = df[~df["tw_code"].str.match(r"^[A-Z]\d{2,3}$", na=False)]
    after = len(df)
    print(f"🧹 删除父层编码 {before - after} 条，保留 {after} 条。")

# === 3. 统计 reason ===
if "reason" in df.columns:
    fail_df = df[df["reason"].astype(str).str.contains("无法解析", na=False)]
    fail_ratio = len(fail_df) / len(df)
    print(f"❌ 无法解析条目：{len(fail_df)} 条，占比 {fail_ratio:.2%}")

    # 导出无法解析表格
    fail_out = r"C:\Users\user\Desktop\icd10\output_icd_pairs_unmatched.xlsx"
    fail_df.to_excel(fail_out, index=False)
    print(f"📤 无法解析结果已保存：{fail_out}")
else:
    print("⚠ 未检测到 reason 列，跳过无法解析统计。")

# === 4. 相似度置信度分析 ===
if "sim" in df.columns:
    valid = df[~df["reason"].astype(str).str.contains("无法解析", na=False)]
    bins = np.arange(0.5, 1.05, 0.05)
    sim_stats = []
    for b in bins:
        subset = valid[valid["sim"] >= b]
        if len(subset) == 0:
            continue
        success_ratio = len(subset) / len(valid)
        sim_stats.append((round(b, 2), round(success_ratio, 4)))

    sim_df = pd.DataFrame(sim_stats, columns=["阈值", "成功率"])
    best_row = sim_df.iloc[(sim_df["成功率"].diff().abs().fillna(0)).idxmax()]
    print("\n📊 相似度阈值分析：")
    print(sim_df)
    print(f"\n✅ 推荐阈值：{best_row['阈值']}（成功率提升点）")

    sim_out = r"C:\Users\user\Desktop\icd10\output_icd_sim_analysis.xlsx"
    sim_df.to_excel(sim_out, index=False)
    print(f"📤 置信度分析结果已保存：{sim_out}")
else:
    print("⚠ 未检测到 sim 列，跳过置信度分析。")

# === 5. 保存清洗后的最终文件 ===
clean_out = r"C:\Users\user\Desktop\icd10\output_icd_pairs_cleaned.xlsx"
df.to_excel(clean_out, index=False)
print(f"\n✅ 清理+分析完成，结果已保存：{clean_out}")
