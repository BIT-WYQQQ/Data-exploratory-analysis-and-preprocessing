import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import time
from collections import Counter

start_time = time.time()
# 设置中文字体和输出目录
sns.set_theme(style='whitegrid')
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# === Step 1: 数据加载与时间字段解析 ===
print("正在加载 structured_transactions.csv，并进行时间字段解析")
df = pd.read_csv("../data/processed_10G_data/structured_transactions.csv")
df["main_categories"] = df["main_categories"].apply(ast.literal_eval)
df["purchase_date"] = pd.to_datetime(df["purchase_date"])
df["year"] = df["purchase_date"].dt.year
df["month"] = df["purchase_date"].dt.month
df["quarter"] = df["purchase_date"].dt.quarter
df["weekday"] = df["purchase_date"].dt.dayofweek  # 0=周一

# === Step 2: 季节性购物行为分析 ===
print("正在进行季节性购物行为分析")
df["month"].value_counts().sort_index().plot(kind='bar', figsize=(8, 4))
plt.title("每月购物行为统计")
plt.xlabel("月份")
plt.ylabel("购物记录数")
plt.tight_layout()
plt.savefig("../data/figs_10G_data/purchase_by_month.png")
plt.close()

df["quarter"].value_counts().sort_index().plot(kind='bar', color='orange', figsize=(6, 4))
plt.title("每季度购物行为统计")
plt.xlabel("季度")
plt.ylabel("购物记录数")
plt.tight_layout()
plt.savefig("../data/figs_10G_data/purchase_by_quarter.png")
plt.close()

df["weekday"].value_counts().sort_index().plot(kind='bar', color='green', figsize=(6, 4))
plt.title("每周购物行为统计")
plt.xlabel("星期")
plt.ylabel("购物记录数")
plt.tight_layout()
plt.savefig("../data/figs_10G_data/purchase_by_weekday.png")
plt.close()

# === Step 3: 商品类别-时间频率变化分析（按月） ===
print("正在进行商品类别-时间频率变化分析")
category_month_rows = []
for _, row in df.iterrows():
    for cat in row["main_categories"]:
        category_month_rows.append({"category": cat, "month": row["month"]})

cat_month_df = pd.DataFrame(category_month_rows)
pivot = cat_month_df.value_counts(["month", "category"]).unstack().fillna(0)
pivot.plot(kind="bar", stacked=True, colormap="tab20", figsize=(12, 6))
plt.title("每月各商品类别的购买频次")
plt.xlabel("月份")
plt.ylabel("记录数")
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
plt.tight_layout()
plt.savefig("../data/figs_10G_data/category_by_month_stackedbar.png")
plt.close()

# === Step 4: 用户购买顺序模式（A类→B类）分析 ===
print("正在进行用户购买顺序模式分析")
# 若无 user_id 列则自动生成
if "user_id" not in df.columns:
    df["user_id"] = df.index
# 对每个用户构建购买序列
df_sorted = df.sort_values(by=["user_id", "purchase_date"])
user_sequences = {}
for user, group in df_sorted.groupby("user_id"):
    sequence = []
    for _, row in group.iterrows():
        sequence.extend(sorted(set(row["main_categories"])))  # 保证一致性
    user_sequences[user] = sequence
# 提取相邻类别转移对
transitions = []
for seq in user_sequences.values():
    for i in range(len(seq) - 1):
        transitions.append((seq[i], seq[i+1]))
# 统计转移频率
trans_count = Counter(transitions)
trans_df = pd.DataFrame(trans_count.items(), columns=["Transition", "Count"]).sort_values(by="Count", ascending=False)
trans_df[["From", "To"]] = pd.DataFrame(trans_df["Transition"].tolist(), index=trans_df.index)
trans_df.drop(columns="Transition", inplace=True)
# 保存与可视化
trans_df.to_csv("../data/processed_10G_data/category_transitions.csv", index=False)
plt.figure(figsize=(10, 6))
sns.barplot(
    data=trans_df.head(15),
    x="Count",
    y=trans_df.head(15).apply(lambda x: f"{x['From']}→{x['To']}", axis=1)
)
plt.title("Top 15 类别顺序转移模式")
plt.xlabel("转移次数")
plt.ylabel("转移路径")
plt.tight_layout()
plt.savefig("../data/figs_10G_data/top15_category_transitions.png")
plt.close()

end_time = time.time()
total_time = end_time - start_time
print(f"总耗时{total_time:.2f} 秒")

