import os
import pandas as pd
import json
from tqdm import tqdm
import gc

# ==== 路径设置 ====
csv_folder = '../data/processed_30G_data'
csv_files = [f for f in os.listdir(csv_folder) if f.endswith('_processed.csv')]
product_catalog_path = '../data/product_catalog.json'

# ==== 初始化记录列表 ====
records = []

# ==== 加载商品目录映射 ====
with open(product_catalog_path, 'r', encoding='utf-8') as f:
    catalog = json.load(f)
product_map = {item['id']: item for item in catalog['products']}  # ID -> 商品信息

# ==== 小类 → 大类映射 ====
subcategory_to_category = {
    "智能手机": "电子产品", "笔记本电脑": "电子产品", "平板电脑": "电子产品", "智能手表": "电子产品",
    "耳机": "电子产品", "音响": "电子产品", "相机": "电子产品", "摄像机": "电子产品", "游戏机": "电子产品",
    "上衣": "服装", "裤子": "服装", "裙子": "服装", "内衣": "服装", "鞋子": "服装", "帽子": "服装",
    "手套": "服装", "围巾": "服装", "外套": "服装",
    "零食": "食品", "饮料": "食品", "调味品": "食品", "米面": "食品", "水产": "食品", "肉类": "食品",
    "蛋奶": "食品", "水果": "食品", "蔬菜": "食品",
    "家具": "家居", "床上用品": "家居", "厨具": "家居", "卫浴用品": "家居",
    "文具": "办公", "办公用品": "办公",
    "健身器材": "运动户外", "户外装备": "运动户外",
    "玩具": "玩具", "模型": "玩具", "益智玩具": "玩具",
    "婴儿用品": "母婴", "儿童课外读物": "母婴",
    "车载电子": "汽车用品", "汽车装饰": "汽车用品"
}


# ==== JSON 解析函数 ====
def parse_purchase_json(purchase_str):
    try:
        fixed = purchase_str.replace('""', '"')
        return json.loads(fixed)
    except:
        return {}

print("\n正在提取 purchase_history 中的结构化信息...")

# ==== 处理csv文件 ====
for file in tqdm(csv_files, desc="处理 CSV 文件"):
    file_path = os.path.join(csv_folder, file)

    try:
        df = pd.read_csv(file_path, usecols=['id', 'purchase_history'])
    except Exception as e:
        print(f"文件读取失败：{file}，跳过。")
        continue

    for _, row in df.iterrows():
        user_id = row['id']
        p_json = parse_purchase_json(row['purchase_history'])
        item_list = p_json.get('items', [])
        if not isinstance(item_list, list) or len(item_list) == 0:
            continue

        method = p_json.get('payment_method')
        status = p_json.get('payment_status')
        price = p_json.get('avg_price')
        date = p_json.get('purchase_date')
        category_set = set()
        for item in item_list:
            item_id = item.get('id')
            if item_id not in product_map:
                continue
            subcategory = product_map[item_id]['category']
            main_category = subcategory_to_category.get(subcategory)
            if main_category:
                category_set.add(main_category)

        if category_set and all([method, status, price, date]):
            records.append({
                'user_id': user_id,
                'purchase_date': date,
                'main_categories': list(category_set),
                'payment_method': method,
                'payment_status': status,
                'price': float(price)
            })

    del df
    gc.collect()

# === 构造 DataFrame 与导出 ===
if not records:
    print("错误：所有记录都无效，无法构建 transactions_df，请检查字段解析逻辑。")
    exit(1)

# 构建 DataFrame 并转换时间格式
transactions_df = pd.DataFrame(records)
transactions_df['purchase_date'] = pd.to_datetime(transactions_df['purchase_date'], errors='coerce')

# 保存结果
transactions_df.to_csv('../data/processed_30G_data/structured_transactions.csv', index=False)
print(f"成功保存结构化数据，共 {len(transactions_df)} 条记录。")
