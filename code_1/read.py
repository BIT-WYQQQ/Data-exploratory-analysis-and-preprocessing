import pandas as pd
import os

# 设置文件夹路径，假设所有 .parquet 文件都在这个文件夹中
folder_path = '../data/10G_data'

# 获取所有 .parquet 文件
parquet_files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]

# 解析并读取每个 .parquet 文件
for file in parquet_files:
    file_path = os.path.join(folder_path, file)
    # 读取 Parquet 文件
    df = pd.read_parquet(file_path)

    # 打印 DataFrame 的基本信息
    print(f"File: {file}")
    print("DataFrame Info:")
    print(df.info())

    # 打印 DataFrame 的前几行数据
    print("First 5 rows of the DataFrame:")
    print(df.head())
    print("\n")
