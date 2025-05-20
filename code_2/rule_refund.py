import pandas as pd
import ast
import time
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from adjustText import adjust_text

start_time = time.time()
# 设置中文与输出路径
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# === Step 1: 读取数据并筛选退款订单 ===
print("正在加载 structured_transactions.csv，并筛选退款订单")
df = pd.read_csv("../data/processed_10G_data/structured_transactions.csv")
df["main_categories"] = df["main_categories"].apply(ast.literal_eval)
df_refund = df[df["payment_status"].isin(["已退款", "部分退款"])].copy()

# === Step 2: 构造商品组合事务列表 ===
print("正在进行退款规则挖掘")
transactions = df_refund["main_categories"].tolist()
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
basket_df = pd.DataFrame(te_array, columns=te.columns_)

# === Step 3: Apriori 挖掘规则 ===
frequent_itemsets = apriori(basket_df, min_support=0.005, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)
if rules.empty:
    print("没有满足置信度规则的结果，尝试使用confidence > 0.2")
    rules_all = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
if rules.empty:
    print("没有满足置信度规则的结果，尝试使用confidence > 0.05")
    rules_all = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.05)
rules = rules.round(3)

# === Step 4: 保存规则表格 ===
rules.to_csv("../data/processed_10G_data/refund_category_rules.csv", index=False)
print("已保存全部规则，即将进行可视化")

# === Step 5: 可视化 Top 15 规则（提升度最高） ===
top_rules = rules.sort_values(by="lift", ascending=False).head(15)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    top_rules["support"],
    top_rules["confidence"],
    s=top_rules["lift"] * 300,
    c=top_rules["lift"],
    cmap="viridis",
    alpha=0.8,
    edgecolors="black"
)
texts = []
for _, row in top_rules.iterrows():
    ant = ",".join(row["antecedents"])
    con = ",".join(row["consequents"])
    label = f"{ant}→{con}"
    texts.append(plt.text(row["support"], row["confidence"], label, fontsize=9))

adjust_text(texts, only_move={"points": "y", "text": "xy"}, arrowprops=dict(arrowstyle="->", color="gray"))
cbar = plt.colorbar(scatter, pad=0.01)
cbar.set_label("提升度 (Lift)", fontsize=10)
plt.xlabel("支持度 (Support)")
plt.ylabel("置信度 (Confidence)")
plt.title("退款商品组合规则：支持度 vs 置信度 vs 提升度")
plt.tight_layout()
plt.savefig("../data/figs_10G_data/refund_rules_bubble_chart.png")
plt.close()

end_time = time.time()
total_time = end_time - start_time
print(f"总耗时{total_time:.2f} 秒")
