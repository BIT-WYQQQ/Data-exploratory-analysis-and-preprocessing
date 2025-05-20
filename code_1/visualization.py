import os
import pandas as pd
import time
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from datetime import datetime
from tqdm import tqdm

# ========= 路径设置 =========
csv_folder = '../data/processed_10G_data'
save_folder = '../data/figs_10G_data'
os.makedirs(save_folder, exist_ok=True)

csv_files = [f for f in os.listdir(csv_folder) if f.endswith('_processed.csv')]

# ========= 初始化统计器 =========
gender_counter = Counter()
country_counter = Counter()
reg_date_counter = defaultdict(int)
sampled_age = []
sampled_income = []

# ========= 扫描所有 CSV 文件 =========
total_start = time.time()
step1_start = total_start
print("正在逐文件收集统计信息...")

for file in tqdm(csv_files, desc="读取文件"):
    file_path = os.path.join(csv_folder, file)
    try:
        chunk = pd.read_csv(file_path, usecols=[
            'gender', 'country', 'age', 'income', 'registration_date'
        ])

        # 分类统计
        gender_counter.update(chunk['gender'].dropna())
        country_counter.update(chunk['country'].dropna())

        # 数值采样
        sampled_age.extend(chunk['age'].dropna().tolist())
        sampled_income.extend(chunk['income'].dropna().tolist())

        # 时间统计
        dates = pd.to_datetime(chunk['registration_date'], errors='coerce').dt.date
        for date in dates.dropna():
            reg_date_counter[date] += 1

        del chunk

    except Exception as e:
        print(f"跳过文件 {file}，错误：{e}")
        continue
step1_time = time.time() - step1_start
print(f"已完成信息统计，用时：{step1_time:.2f} 秒")

# ========= 设置绘图风格与中文字体 =========
print("开始绘制图像")
step2_start = time.time()
sns.set_theme(style="whitegrid")
matplotlib.rc("font", family='SimHei')

# ========= 1. Gender 饼图 =========
plt.figure(figsize=(6, 6))
labels, sizes = zip(*gender_counter.items())
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("性别分布饼状图")
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(save_folder, "gender_distribution.png"))
print("性别分布饼状图已绘制完成")
# plt.show()

# ========= 2. Country 饼图 =========
plt.figure(figsize=(6, 6))
top_items = country_counter.most_common(5)
top_labels = [k for k, _ in top_items]
top_sizes = [v for _, v in top_items]
other_total = sum(country_counter.values()) - sum(top_sizes)
labels = top_labels + ['其他']
sizes = top_sizes + [other_total]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("国家分布饼状图 (Top 5 + 其他)")
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(save_folder, "country_distribution.png"))
print("国家分布饼状图已绘制完成")
# plt.show()

# ========= 3. age 直方图 =========
plt.figure(figsize=(6, 4))
sns.histplot(sampled_age, kde=True, bins=30)
plt.title("年龄分布直方图")
plt.xlabel("年龄")
plt.ylabel("用户数量")
plt.tight_layout()
plt.savefig(os.path.join(save_folder, "age_distribution.png"))
print("年龄分布直方图已绘制完成")
# plt.show()

# ========= 4. income 直方图 =========
plt.figure(figsize=(6, 4))
sns.histplot(sampled_income, kde=True, bins=30)
plt.title("收入分布直方图")
plt.xlabel("收入")
plt.ylabel("用户数量")
plt.tight_layout()
plt.savefig(os.path.join(save_folder, "income_distribution.png"))
print("收入分布直方图已绘制完成")
# plt.show()

# ========= 5. reg_date折线图 =========
plt.figure(figsize=(10, 5))
reg_series = pd.Series(reg_date_counter).sort_index()
reg_series.plot(kind='line')
plt.title("用户注册日期折线图")
plt.xlabel("日期")
plt.ylabel("注册数量")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_folder, "registration_trend.png"))
print("用户注册日期折线图已绘制完成")
step2_time = time.time() - step2_start
print(f"已完成绘图，用时{step2_time:.2f} 秒")
total_time = time.time() - total_start
print(f"总耗时{total_time:.2f} 秒")
# plt.show()