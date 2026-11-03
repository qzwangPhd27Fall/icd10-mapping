# Cross-Version ICD-10 Disease-Code Mapping

Code for aligning disease-code dictionaries across ICD-10 versions: mapping the **China ICD-10 Medical Insurance v2.0** codes to **Taiwan ICD-10-CM (2023)**, covering 33,000+ diagnosis entries.

## Approach

A three-stage mapping pipeline combining semantic retrieval and LLM judgment:

1. **Code-key constraint** — exact code-string matching as a hard constraint;
2. **Semantic vector recall** — MC-BERT embeddings (with whitening) to retrieve candidate matches by name similarity;
3. **LLM judgment** — Qwen / DeepSeek models score and finalize candidate pairs.

## Scripts

| Script | Role |
|---|---|
| `map_mb2_twicd.py` | MC-BERT embedding mapping (medical insurance v2.0 → Taiwan ICD-10-CM) |
| `map_mb2_twicd_mcbert_whiten.py` | MC-BERT + whitening variant |
| `map_mb2_twicd_deepseek.py` | DeepSeek-LLM-judged mapping |
| `icd_align_llm.py` / `icd_align_llm_qwen*.py` | LLM alignment pipelines (Qwen) |
| `run_name_mapping_windows.py` | Name-based mapping runner |
| `download_models.py` | Model download helper |
| `run_icdmap.bat` | One-click batch entry point |

## Data note

The input code dictionaries (ICD-10 Medical Insurance v2.0, Taiwan ICD-10-CM 2023) are not included in this repository.
