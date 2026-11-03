@echo off
REM 批处理：运行医保 ICD-10 → CCIR 映射

REM 1. 激活 conda 环境（改成你建的环境名，默认是 icdmap）
call conda activate icdmap

REM 2. 确认依赖安装（只会在第一次跑时用到）
pip install -q pandas openpyxl scikit-learn

REM 3. 切换到脚本目录
cd /d C:\Users\user\Desktop\icd10

REM 4. 运行脚本
echo =====================================
echo 正在运行 ICD 映射脚本...
echo =====================================
python -u run_name_mapping_windows.py

echo.
echo =====================================
echo 任务已完成，请检查输出文件：
echo   医保2.0_name_map_with_CCIR.xlsx
echo   医保2.0_name_map_topk.csv
echo =====================================
pause
