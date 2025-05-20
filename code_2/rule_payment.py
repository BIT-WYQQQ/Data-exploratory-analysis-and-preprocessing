import pandas as pd
import ast
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from adjustText import adjust_text

start_time = time.time()
# === Step 1: 加载数据并解析 main_categories ===
print("正在加载 structured_transactions.csv，并将类别字段转换为列表")
df = pd.read_csv("../data/processed_30G_data/structured_transactions.csv")
df["main_categories"] = df["main_categories"].apply(ast.literal_eval)
payment_methods = {"现金", "微信支付", "支付宝", "储蓄卡", "信用卡", "银联", "云闪付"}

# === Step 2: 全部商品的规则挖掘 ===
print("正在进行全部商品的规则挖掘")
transactions_all = []
for _, row in df.iterrows():
    pay = row["payment_method"]
    for cat in row["main_categories"]:
        transactions_all.append([pay, cat])
te = TransactionEncoder()
te_array_all = te.fit(transactions_all).transform(transactions_all)
basket_all = pd.DataFrame(te_array_all, columns=te.columns_)

frequent_itemsets_all = apriori(basket_all, min_support=0.01, use_colnames=True)
rules_all = association_rules(frequent_itemsets_all, metric="confidence", min_threshold=0.6)
if rules_all.empty:
    print("没有满足置信度规则的结果，尝试使用confidence > 0.3")
    rules_all = association_rules(frequent_itemsets_all, metric="confidence", min_threshold=0.3)
if rules_all.empty:
    print("没有满足置信度规则的结果，尝试使用confidence > 0.05")
    rules_all = association_rules(frequent_itemsets_all, metric="confidence", min_threshold=0.05)
if rules_all.empty:
    print("没有满足置信度规则的结果，尝试使用lift > 0.3")
    rules_all = association_rules(frequent_itemsets_all, metric="lift", min_threshold=0.3)
if rules_all.empty:
    print("没有满足置信度规则的结果，尝试使用lift > 0.1")
    rules_all = association_rules(frequent_itemsets_all, metric="lift", min_threshold=0.1)
if rules_all.empty:
    print("没有满足置信度规则的结果，不进行任何限制")
    rules_all = association_rules(frequent_itemsets_all, metric="confidence", min_threshold=0)
rules_all = rules_all.round(3)

rules_all_pay = rules_all[rules_all["antecedents"].apply(lambda x: any(p in x for p in payment_methods))]
rules_all_pay.to_csv("../data/processed_30G_data/payment_to_category_rules.csv", index=False)
print("已保存全部规则，即将进行高价值商品首选支付方式分析以及规则可视化")

# === Step 3: 高价值商品首选支付方式分析===
df_high = df[df["price"] > 5000]
print("高价值商品的首选支付方式：", df_high["payment_method"].value_counts().idxmax())

# === Step 4: 可视化（气泡图） ===
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
top_rules = rules_all_pay.sort_values(by="lift", ascending=False).head(10)
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(
    rules_all_pay["support"],
    rules_all_pay["confidence"],
    s=rules_all_pay["lift"] * 100,
    c=rules_all_pay["lift"],
    cmap="viridis",
    alpha=0.7,
    edgecolors="black"
)
texts = []
for _, row in top_rules.iterrows():
    ant = ",".join(row['antecedents'])
    con = ",".join(row['consequents'])
    label = f"{ant}→{con}"
    texts.append(ax.text(row["support"], row["confidence"], label, fontsize=9))
adjust_text(texts, ax=ax, only_move={"points": "y", "text": "xy"}, arrowprops=dict(arrowstyle="->", color="gray"))
cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
cbar.set_label("提升度 (Lift)", fontsize=10)
ax.set_xlabel("支持度 (Support)")
ax.set_ylabel("置信度 (Confidence)")
ax.set_title("支付方式 → 商品类别：规则气泡图")
plt.tight_layout()
plt.savefig("../data/figs_30G_data/payment_rules_bubble_chart.png")

'''
# === Step 5: 可视化（网络图） ===
filtered_rules = rules_all_pay[rules_all_pay['confidence'] > 0.1]
G = nx.DiGraph()
for _, row in filtered_rules.iterrows():
    ant = ",".join(row["antecedents"])
    con = ",".join(row["consequents"])
    G.add_edge(ant, con, label=f"lift={row['lift']}, conf={row['confidence']}")
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, k=2.0, seed=42)
nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="lightblue")
nx.draw_networkx_labels(G, pos, font_size=9, font_family='SimHei')
nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20)
edge_labels = nx.get_edge_attributes(G, "label")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, rotate=True)
plt.title("支付方式 → 商品类别：网络结构图")
plt.axis("off")
plt.tight_layout()
plt.savefig("../data/figs_30G_data/payment_rules_network_graph.png")
'''

end_time = time.time()
total_time = end_time - start_time
print(f"总耗时{total_time:.2f} 秒")
