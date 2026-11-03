# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 23:43:42 2025

@author: Administrator
"""

from modelscope.hub.snapshot_download import snapshot_download

# ✅ 1. 下载中文 BGE embedding 模型（替代 m3e-base）
print("Downloading embedding model: AI-ModelScope/bge-base-zh ...")
snapshot_download('AI-ModelScope/bge-base-zh', cache_dir='C:/models/bge-base-zh', revision=None)

# ✅ 2. （可选）下载 Qwen2.5-3B-Instruct 作为 LLM 对齐模型
print("Downloading LLM: qwen/Qwen2.5-3B-Instruct ...")
snapshot_download('qwen/Qwen2.5-3B-Instruct', cache_dir='C:/models/Qwen2.5-3B-Instruct', revision=None)

print("✅ All models downloaded successfully!")

