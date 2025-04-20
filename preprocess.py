import pandas as pd
import numpy as np
import os
import time
import gc
from tqdm import tqdm
from datetime import datetime

# ========= 配置路径 =========
input_folder = '../data/30G_data'
output_folder = '../data/processed_30G_data'
os.makedirs(output_folder, exist_ok=True)

# ========= 初始化全局统计 =========
total_stats = {
    'original_rows': 0,
    'deduplicated_rows': 0,
    'missing_dropped': 0,
    'outliers_removed': 0,
    'final_rows': 0,
}
step_times = {}
total_start = time.time()


# ========= 函数：处理单个 parquet 文件 =========
def preprocess_parquet(file_path, output_path, index, total_files):
    local_stats = {}
    start_time = time.time()
    file_name = os.path.basename(file_path)

    print(f"\n[{index}/{total_files}] 开始处理文件：{file_name}")

    # Step 1: 读取 parquet
    print("正在读取数据...")
    df = pd.read_parquet(file_path)
    local_stats['original'] = len(df)
    print(f"读取完成，记录数：{len(df)}")

    # Step 2: 去重
    print("正在去重...")
    before = len(df)
    df.drop_duplicates(subset=['id'], keep='first', inplace=True)
    removed = before - len(df)
    local_stats['deduplicated'] = removed
    print(f"去重完成，去除 {removed} 条")

    # Step 3: 删除缺失值
    print("正在删除缺失值...")
    before = len(df)
    df.dropna(inplace=True)
    removed = before - len(df)
    local_stats['missing_dropped'] = removed
    print(f"删除缺失值记录 {removed} 条")

    # Step 4: 异常值删除（Z-score）
    print("正在检测并删除异常值...")
    before = len(df)
    for col in ['age', 'income']:
        z = (df[col] - df[col].mean()) / df[col].std()
        df = df[np.abs(z) <= 3]
    removed = before - len(df)
    local_stats['outliers_removed'] = removed
    print(f"删除异常值记录 {removed} 条")

    # Step 5: 时间字段转换
    print("正在转换时间字段...")
    df['last_login'] = pd.to_datetime(df['last_login'], errors='coerce')
    df['registration_date'] = pd.to_datetime(df['registration_date'], errors='coerce')
    print("时间字段转换完成")

    # Step 6: 保存 CSV
    print("正在保存为 CSV 文件...")
    df.to_csv(output_path, index=False)
    local_stats['final'] = len(df)
    print(f"保存完成，剩余记录数：{len(df)}")

    del df
    gc.collect()

    step_times[file_name] = time.time() - start_time
    print(f"文件处理完成，用时：{step_times[file_name]:.2f} 秒")
    return local_stats


# ========= 主处理流程 =========
parquet_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.parquet')])
print(f"共检测到 {len(parquet_files)} 个 parquet 文件")

for idx, file in enumerate(parquet_files, 1):
    file_path = os.path.join(input_folder, file)
    output_csv = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_processed.csv")

    stats = preprocess_parquet(file_path, output_csv, idx, len(parquet_files))

    total_stats['original_rows'] += stats['original']
    total_stats['deduplicated_rows'] += stats['deduplicated']
    total_stats['missing_dropped'] += stats['missing_dropped']
    total_stats['outliers_removed'] += stats['outliers_removed']
    total_stats['final_rows'] += stats['final']

# ========= 总结统计输出 =========
total_time = time.time() - total_start

print("\n处理完成，总体数据统计如下：")
print(f"原始总数据量: {total_stats['original_rows']}")
print(f"去重记录总数: {total_stats['deduplicated_rows']}")
print(f"删除缺失值记录总数: {total_stats['missing_dropped']}")
print(f"删除异常值记录总数: {total_stats['outliers_removed']}")
print(f"最终保留总记录数: {total_stats['final_rows']}")
print(f"\n总耗时: {total_time:.2f} 秒")

print("\n每个文件的耗时（秒）：")
for file, sec in step_times.items():
    print(f" - {file:<30}: {sec:.2f} 秒")