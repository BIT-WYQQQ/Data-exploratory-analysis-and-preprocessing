import pandas as pd
import ast
import time
import matplotlib
import matplotlib.pyplot as plt
from pylab import mpl
import seaborn as sns
import networkx as nx
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from adjustText import adjust_text

start_time = time.time()
# === 1. 加载 structured_transactions.csv，并将类别字段转换为列表 ===
print("正在加载 structured_transactions.csv，并将类别字段转换为列表")
df = pd.read_csv("../data/processed_10G_data/structured_transactions.csv")
df["main_categories"] = df["main_categories"].apply(ast.literal_eval)

# === 2. 构造事务列表，每个用户的一次购买是一条事务 ===
print("正在构造事务列表，每个用户的一次购买是一条事务")
transactions = df["main_categories"].tolist()

# === 3. One-hot 编码 ===
print("正在进行One-hot编码")
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

# === 4. Apriori 频繁项集挖掘（支持度 ≥ 0.02）===
print("正在进行Apriori频繁项集挖掘")
frequent_itemsets = apriori(df_encoded, min_support=0.02, use_colnames=True)

# === 5. 关联规则生成（置信度 ≥ 0.5）===
print("正在进行关联规则生成")
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
if rules.empty:
    print("没有满足置信度规则的结果，尝试使用 lift > 1.0")
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules["support"] = rules["support"].round(3)
rules["confidence"] = rules["confidence"].round(3)
rules["lift"] = rules["lift"].round(3)

# === 6. 保存全部规则 ===
print("正在保存全部规则")
rules.to_csv("../data/processed_10G_data/main_category_rules.csv", index=False)
print("已保存全部规则，即将进行可视化")

# === 7. 选出与“电子产品”相关的规则（用于可视化）===
rules_electronics = rules[
    rules["antecedents"].apply(lambda x: "电子产品" in set(x)) |
    rules["consequents"].apply(lambda x: "电子产品" in set(x))
]

# === 8. 可视化：气泡图 ===
sns.set_theme(style="whitegrid")
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False
top_rules = rules_electronics.sort_values(by='lift', ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(
    rules_electronics['support'],
    rules_electronics['confidence'],
    s=rules_electronics['lift'] * 80,
    c=rules_electronics['lift'],
    cmap='viridis',
    alpha=0.7,
    edgecolors='black'
)
texts = []
for _, row in top_rules.iterrows():
    label = f"{','.join(row['antecedents'])}→{','.join(row['consequents'])}"
    texts.append(plt.text(row['support'], row['confidence'], label, fontsize=9))
adjust_text(texts, only_move={'points':'y', 'text':'xy'}, arrowprops=dict(arrowstyle='->', color='gray'))

cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
cbar.set_label("提升度 (Lift)", fontsize=10)
ax.set_xlabel("支持度 (Support)", fontsize=10)
ax.set_ylabel("置信度 (Confidence)", fontsize=10)
ax.set_title("电子产品类规则：支持度 vs 置信度 vs 提升度", fontsize=12)
plt.tight_layout()
plt.savefig("../data/figs_10G_data/electronics_rules_bubble_chart.png")
# plt.show()

# === 9. 可视化：网络图 ===
filtered_rules = rules_electronics[rules_electronics['lift'] > 0.9]
G = nx.DiGraph()
for _, row in filtered_rules.iterrows():
    ant = ','.join(row['antecedents'])
    con = ','.join(row['consequents'])
    G.add_edge(ant, con, label=f"lift={row['lift']}, conf={row['confidence']}")
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, k=1.2, seed=42)
nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue')
nx.draw_networkx_labels(G, pos, font_size=9)
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, rotate=True)
plt.title("电子产品类规则结构图")
plt.axis("off")
plt.tight_layout()
plt.savefig("../data/figs_10G_data/electronics_rules_network_graph.png")
# plt.show()

end_time = time.time()
total_time = end_time - start_time
print(f"总耗时{total_time:.2f} 秒")
