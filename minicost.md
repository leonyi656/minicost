# minicost
the code of minicost  model
import pandas as pd, numpy as np, networkx as nx, matplotlib.pyplot as plt
import math
from pathlib import Path

# ---------------- 参数 ------------------------------------------------
YEAR       = 2023
BASE_DIR   = Path(r'G:/生态系统服务价值网络/水资源盈亏')

BAL_FILE   = BASE_DIR / f'city_water_balance{YEAR}.csv'      # 含供/需列
DIST_FILE  = BASE_DIR / f'neighbor_distance{YEAR}.csv'       # 近邻距离

BIG_CAP    = 1_000_000_000         # 源→需运输边最大容量 (足够大)
SCALE      = 100                   # weight = km × SCALE  → 整数化

# ---------------- 1. 读入盈亏 ----------------------------------------
df = pd.read_csv(BAL_FILE, encoding='utf-8-sig')

# ★ 若列名不同，请在此处对应修改
df['balance_val'] = df['water_supply_yim3'] - df['water_demand_yim3']

sources = df[df.balance_val > 0].set_index('city').copy()
sinks   = df[df.balance_val < 0].set_index('city').copy()
sources['surplus'] = sources['balance_val']
sinks['deficit']   = sinks['balance_val'].abs()

print(f"✅ {YEAR}: {len(sources)} 源区, {len(sinks)} 需区")

# ---------------- 2. 读邻接 & 过滤 -----------------------------------
dist_df = pd.read_csv(DIST_FILE, encoding='utf-8-sig')
# 如列名不是 src_city/dst_city/distance_km，可在此重命名
# dist_df = dist_df.rename(columns={'市1':'src_city','市2':'dst_city','NEAR_DIST':'distance_km'})

dist_df = (dist_df
           .merge(sources.reset_index()[['city']], left_on='src_city', right_on='city')
           .drop(columns='city')
           .merge(sinks.reset_index()[['city']],   left_on='dst_city', right_on='city')
           .drop(columns='city')
           .reset_index(drop=True))

# 距离若是米 → 转公里
if dist_df['distance_km'].max() > 1000:
    dist_df['distance_km'] /= 1000

print(f"✅ 邻接行数: {len(dist_df)}")

# ---------------- 3. 构图 & 最小成本流 -------------------------------
FC = nx.DiGraph()

# 3-1  超级源 / 汇（整数容量 = 实际供需，单位：亿 m³）
for s in sources.index:
    FC.add_edge('SRC', s,
                capacity=int(round(sources.at[s, 'surplus'])),
                weight=0)
for t in sinks.index:
    FC.add_edge(t, 'DST',
                capacity=int(round(sinks.at[t, 'deficit'])),
                weight=0)

# 3-2  源→需运输边：大容量 + 整数权重
for _, r in dist_df.iterrows():
    FC.add_edge(r.src_city, r.dst_city,
                capacity=BIG_CAP,
                weight=int(round(r.distance_km * SCALE)))

# 3-3  旁路平衡（SRC→DST，0 成本）
extra = sources.surplus.sum() - sinks.deficit.sum()
if abs(extra) > 1e-6:
    FC.add_edge('SRC', 'DST',
                capacity=int(math.ceil(abs(extra))),
                weight=0)
    print(f"⚙️  旁路 SRC→DST 容量 = {abs(extra):,.2f} 亿m³")

# 3-4  求解并计算成本
flow_dict   = nx.max_flow_min_cost(FC, 'SRC', 'DST')
cost_scaled = nx.cost_of_flow(FC, flow_dict)           # 单位：km × SCALE × 亿m³
print(f"✅ Min-cost 完成, 总成本 = {cost_scaled / SCALE:,.2f} km·亿m³")

# ---------------- 4. 抽取有效边 --------------------------------------
edge_list = [(u, v, {'flow_yim3': f,
                     'distance_km': FC[u][v]['weight'] / SCALE})
             for u, nbr in flow_dict.items()
             for v, f in nbr.items()
             if u not in ('SRC', 'DST') and v != 'DST' and f > 0]

print(f"✅ 有效边数: {len(edge_list)}")

# ---------------- 5. 节点指标 ----------------------------------------
G = nx.DiGraph()
G.add_nodes_from(df.city)
G.add_edges_from([(u, v, {'weight': d['flow_yim3']}) for u, v, d in edge_list])

node_df = pd.DataFrame({'city': df.city})
node_df['outflow_yim3'] = node_df.city.map(dict(G.out_degree(weight='weight'))).fillna(0)
node_df['inflow_yim3']  = node_df.city.map(dict(G.in_degree (weight='weight'))).fillna(0)
node_df['betweenness']  = node_df.city.map(nx.betweenness_centrality(G, weight='weight'))

# ---------------- 6. CSV 输出 ----------------------------------------
edge_df = pd.DataFrame(
    [(u, v, d['flow_yim3'], d['distance_km'], d['flow_yim3'] * d['distance_km'])
     for u, v, d in edge_list],
    columns=['src_city', 'dst_city', 'flow_yim3', 'distance_km', 'cost_km_yim3']
)

edge_csv = BASE_DIR / f'water_flow_edges_{YEAR}.csv'
node_csv = BASE_DIR / f'water_flow_nodes_{YEAR}.csv'
sum_csv  = BASE_DIR / f'water_flow_summary_{YEAR}.csv'

edge_df.to_csv(edge_csv, index=False, encoding='utf-8-sig')
node_df.to_csv(node_csv,  index=False, encoding='utf-8-sig')
pd.DataFrame({'total_cost_km_yim3': [edge_df.cost_km_yim3.sum()]}
             ).to_csv(sum_csv, index=False, encoding='utf-8-sig')

print(f"💾 边表: {edge_csv.name}")
print(f"💾 节点: {node_csv.name}")
print(f"💾 汇总: {sum_csv.name}  (total_cost_km_yim3 = {edge_df.cost_km_yim3.sum():,.2f})")

# ---------------- 7. 可选：绘图 --------------------------------------
# 如需绘图，可在此处加入 networkx + matplotlib 代码；略。
