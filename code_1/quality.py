import os
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
import time
import gc

# ==== 路径配置 ====
csv_folder = '../data/processed_10G_data'
csv_files = [f for f in os.listdir(csv_folder) if f.endswith('_processed.csv')]

# ==== JSON 字符串修复函数 ====
def parse_json_field(raw_str):
    try:
        fixed_str = raw_str.replace('""', '"').strip()
        return json.loads(fixed_str)
    except Exception:
        return {}

# ==== 用户质量评分函数 ====
def compute_score(row):
    # 年龄评分
    age = row['age']
    if 25 <= age <= 45:
        age_score = 1
    elif 20 <= age < 25 or 45 < age <= 55:
        age_score = 0.7
    else:
        age_score = 0.3
    # 收入评分
    income = row['income']
    # 活跃度评分
    active_score = 1.0 if row['is_active'] else 0.0
    # 购买评分
    try:
        p_hist = parse_json_field(row['purchase_history'])
        avg_price = float(p_hist.get('avg_price', 0))
        status = p_hist.get('payment_status', '已退款')
        payment_weight = {
            '已支付': 1.0,
            '部分退款': 0.5,
            '已退款': 0.0
        }.get(status, 0.0)
        purchase_score = avg_price * payment_weight
    except:
        purchase_score = 0
    # 登录评分
    try:
        l_hist = parse_json_field(row['login_history'])
        login_count = int(l_hist.get('login_count', 0))
    except:
        login_count = 0
    return {
        'id': row['id'],
        'fullname': row['fullname'],
        'age_score': age_score,
        'income': income,
        'active_score': active_score,
        'purchase_score': purchase_score,
        'login_count': login_count
    }

# ==== 打分阶段 ====
all_scores = []
start_all = time.time()
start_scoring = time.time()

print("正在为用户打分并收集特征...")
for file in tqdm(csv_files, desc="处理文件"):
    file_path = os.path.join(csv_folder, file)
    df = pd.read_csv(file_path)

    for _, row in df.iterrows():
        result = compute_score(row)
        all_scores.append(result)

    del df
    gc.collect()

end_scoring = time.time()
scoring_time = end_scoring - start_scoring
print(f"打分完成，总用户数：{len(all_scores)}，耗时：{scoring_time:.2f} 秒")

# ==== 数据整理 + 标准化 ====
start_calc = time.time()
score_df = pd.DataFrame(all_scores)

def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-6)

score_df['income_score'] = min_max_normalize(score_df['income'])
score_df['login_score'] = min_max_normalize(score_df['login_count'])
score_df['purchase_score_norm'] = min_max_normalize(score_df['purchase_score'])

# 综合得分
score_df['quality_score'] = (
    0.15 * score_df['age_score'] +
    0.25 * score_df['income_score'] +
    0.15 * score_df['active_score'] +
    0.25 * score_df['purchase_score_norm'] +
    0.20 * score_df['login_score']
)

# ==== 输出 Top 100 用户 ====
top_100 = score_df.sort_values(by='quality_score', ascending=False).head(100)
end_calc = time.time()
calc_time = end_calc - start_calc
print(f"得分计算、标准化以及用户排序完成，耗时：{calc_time:.2f} 秒")
top_100[['id', 'fullname', 'quality_score']].to_csv('../data/10G_data/top100_high_quality_users.csv', index=False)

# ==== 总耗时统计 ====
total_time = time.time() - start_all
print(f"\n全流程结束，已输出 Top 100 用户：top100_high_quality_users.csv")
print(f"总耗时：{total_time:.2f} 秒")