import pygame
import pandas as pd
import json
import random
import ast
import sys
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import numpy as np
# 数据预处理依赖导入
import geopandas as gpd
from shapely.geometry import LineString, Point
import os

# 未知导入
import tkinter as tk
from tkinter import filedialog
import tempfile
import shutil
from tkinter import messagebox
# ================================
# 全局常量定义
# ================================
CELL_SIZE = 7.5  # 元胞长度(米)

# 自动寻找可用字体
def get_font_path():
    for f in ['C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/msyh.ttc']:
        if os.path.exists(f):
            return f
    return None

# 全局辅助函数
def time_str_to_seconds(t_str):
    """将 'HH:MM' 格式的时间字符串转换为总秒数"""
    h, m = map(int, t_str.split(':'))
    return h * 3600 + m * 60

# ==========================================
# 数据预处理
# ==========================================

def split_lines(gdf):
    nodes_set = set()
    segments = []
    pts_list = []  # 临时存储所有端点，用于后续容差合并

    for idx, row in gdf.iterrows():
        line = row.geometry
        coords = list(line.coords)
        for i in range(len(coords) - 1):
            seg = LineString([coords[i], coords[i + 1]])
            start_pt = coords[i]
            end_pt = coords[i+1]
            nodes_set.add(start_pt)
            nodes_set.add(end_pt)
            segments.append({
                'geom': seg,
                'attrs': row.to_dict(),
                'start': start_pt,
                'end': end_pt
            })
            pts_list.append(start_pt)
            pts_list.append(end_pt)

    # ---- 容差合并：将相距小于 0.1 米的点归为同一个 node_id ----
    tolerance = 0.1  # 米
    pts_array = np.array(pts_list)
    merged_ids = list(range(len(pts_list)))  # 初始每个点一个 id
    for i in range(len(pts_list)):
        for j in range(i+1, len(pts_list)):
            dist = np.hypot(pts_array[i][0]-pts_array[j][0], pts_array[i][1]-pts_array[j][1])
            if dist < tolerance:
                # 将 j 的 id 指向 i
                merged_ids[j] = merged_ids[i]

    # 创建坐标到 node_id 的映射（取每个组第一个点的坐标）
    coord_to_nid = {}
    for i, pt in enumerate(pts_list):
        root = merged_ids[i]
        if root not in coord_to_nid:
            coord_to_nid[root] = (pt, f"node_{len(coord_to_nid)}")

    # 为 seg 分配 start_node / end_node
    for seg in segments:
        def find_nid(pt):
            for i, p in enumerate(pts_list):
                if p == pt:  # 精确匹配（原始点）
                    root = merged_ids[i]
                    return coord_to_nid[root][1]
            # 如果没找到，使用最近点（安全兜底，但不应该发生）
            return None
        seg['start_node'] = find_nid(seg['start'])
        seg['end_node'] = find_nid(seg['end'])

    # 构建 nodes_df
    nodes_df = pd.DataFrame([
        {'node_id': nid, 'x': pt[0], 'y': pt[1]}
        for (pt, nid) in coord_to_nid.values()
    ])

    return segments, nodes_df

def densify_line(line, distance):
    """沿中心线等距重采样"""
    if line.length < distance:
        return [line.coords[0], line.coords[-1]]

    num_points = int(math.ceil(line.length / distance)) + 1
    points = [line.interpolate(i * distance).coords[0] for i in range(num_points)]

    # 确保终点精确
    if math.hypot(points[-1][0] - line.coords[-1][0], points[-1][1] - line.coords[-1][1]) > 1.0:
        points.append(line.coords[-1])

    return points


def process_network(line_shp,point_shp,output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取数据
    line_gdf = gpd.read_file(line_shp)
    point_gdf = gpd.read_file(point_shp)

    # 2. 打断线段
    segments, nodes_df = split_lines(line_gdf)

    # 3. 生成有向边
    edges_records = []
    edge_counter = 0

    for seg in segments:
        attrs = seg['attrs']
        geom = seg['geom']
        sn = seg['start_node']
        en = seg['end_node']

        # 车道数：先尝试 'Lanes'，再尝试 'lanes'，默认 1
        _lanes = attrs.get('Lanes', attrs.get('lanes', 1))
        lanes = int(_lanes) if not pd.isna(_lanes) else 1

        # 是否单向
        _is_oneway = attrs.get('is_oneway', attrs.get('Is_Oneway', 0))
        is_oneway = int(_is_oneway) if not pd.isna(_is_oneway) else 0

        # 单向方向
        _oneway_dir = attrs.get('Oneway_Dir', attrs.get('oneway_dir', 0))
        oneway_dir = int(_oneway_dir) if not pd.isna(_oneway_dir) else 0

        # 是否允许换道
        _is_flex = attrs.get('is_flex', attrs.get('Is_Flex', 0))
        is_flex = int(_is_flex) if not pd.isna(_is_flex) else 0

        # POI 类型
        poi_type = attrs.get('POI_Type', attrs.get('poi_type', None))
        if pd.isna(poi_type):
            poi_type = None

        # 生成方向（In/Out）
        spawn_dir = attrs.get('Spawn_Dir', attrs.get('spawn_dir', None))
        if pd.isna(spawn_dir):
            spawn_dir = None

        # 道路名称
        name = attrs.get('name', attrs.get('Name', None))
        if pd.isna(name):
            name = None

        # 判断是否需要反转几何体（单向且方向为 1 时反转）
        reverse_geom = (is_oneway == 1 and oneway_dir == 1)
        if reverse_geom:
            geom = LineString(list(geom.coords)[::-1])
            sn, en = en, sn

        # 采样坐标
        resampled = densify_line(geom, CELL_SIZE)
        cell_count = len(resampled)

        # 决定原边的 spawn_dir：若为 POI 边且未指定方向，默认 In
        final_spawn_dir = spawn_dir
        if poi_type and not spawn_dir:
            final_spawn_dir = 'In'

        edges_records.append({
            'edge_id': f'edge_{edge_counter}',
            'start_node': sn, 'end_node': en,
            'lanes': lanes, 'cell_count': cell_count,
            'max_speed': 5, 'length_m': round(geom.length, 2),
            'is_spawn_edge': poi_type is not None,
            'poi_type': poi_type,
            'spawn_dir': final_spawn_dir,
            'is_flex': is_flex,
            'road_name': name,
            'resampled_coords': ';'.join([f"{x},{y}" for x, y in resampled]),
            'connected_edges_at_end': ''  # 后续补全
        })
        edge_counter += 1

        # 生成反向边（非单向且无 spawn_dir 强制指定时）
        should_gen_reverse = (is_oneway == 0 and not spawn_dir)

        if should_gen_reverse:
            rev_resampled = list(reversed(resampled))
            rev_spawn = 'Out' if poi_type else None

            edges_records.append({
                'edge_id': f'edge_{edge_counter}',
                'start_node': en, 'end_node': sn,
                'lanes': lanes, 'cell_count': cell_count,
                'max_speed': 5, 'length_m': round(geom.length, 2),
                'is_spawn_edge': poi_type is not None,
                'poi_type': poi_type,
                'spawn_dir': rev_spawn,
                'is_flex': is_flex,
                'road_name': f"{name}(反)" if name else None,
                'resampled_coords': ';'.join([f"{x},{y}" for x, y in rev_resampled]),
                'connected_edges_at_end': ''
            })
            edge_counter += 1

    edges_df = pd.DataFrame(edges_records)

    # 4. 计算 connected_edges_at_end
    start_node_map = {}
    for _, row in edges_df.iterrows():
        sn = row['start_node']
        start_node_map.setdefault(sn, []).append(row['edge_id'])

    for idx, row in edges_df.iterrows():
        en = row['end_node']
        connected = start_node_map.get(en, [])
        edges_df.at[idx, 'connected_edges_at_end'] = str(connected)

    # 5. 处理点 SHP (生成可视化标签)
    poi_labels = []
    for _, row in point_gdf.iterrows():
        if pd.notna(row.get('name')) and pd.notna(row.get('POI_Type')):
            poi_labels.append({
                'name': row['name'],
                'poi_type': row['POI_Type'],
                'x': row.geometry.x,
                'y': row.geometry.y
            })
    poi_labels_df = pd.DataFrame(poi_labels)


    # 保存文件
    nodes_df.to_csv(os.path.join(output_dir, 'nodes.csv'), index=False)
    edges_df.to_csv(os.path.join(output_dir, 'edges.csv'), index=False)
    poi_labels_df.to_csv(os.path.join(output_dir, 'poi_labels.csv'), index=False)



    print(f"处理完成！共生成 {len(nodes_df)} 个节点, {len(edges_df)} 条有向边, {len(poi_labels_df)} 个标签。")


# 1. 数据加载
def load_data(output_dir):
    nodes_df = pd.read_csv(os.path.join(output_dir, 'nodes.csv'))
    edges_df = pd.read_csv(os.path.join(output_dir, 'edges.csv'))
    poi_df = pd.read_csv(os.path.join(output_dir, 'poi_labels.csv'))

    with open(os.path.join(output_dir,'demand_config.json'), 'r', encoding='utf-8') as f:
        config = json.load(f)

    nodes = {}
    for _, row in nodes_df.iterrows():
        nodes[row['node_id']] = {'x': row['x'], 'y': row['y']}

    edges = {}
    for _, row in edges_df.iterrows():
        coords = []
        for p in row['resampled_coords'].split(';'):
            x, y = map(float, p.split(','))
            coords.append((x, y))
        coord_dict = {l: coords for l in range(int(row['lanes']))}
        edges[row['edge_id']] = {
            'start_node': row['start_node'], 'end_node': row['end_node'],
            'lanes': int(row['lanes']), 'cell_count': int(row['cell_count']),
            'max_speed': int(row['max_speed']), 'is_flex': int(row.get('is_flex', 0)),
            'is_spawn_edge': str(row['is_spawn_edge']).lower() == 'true',
            'poi_type': row['poi_type'] if pd.notna(row['poi_type']) else None,
            'spawn_dir': row['spawn_dir'] if pd.notna(row['spawn_dir']) else None,
            'coords': coord_dict,
            'connected_edges': ast.literal_eval(row['connected_edges_at_end']) if pd.notna(
                row['connected_edges_at_end']) else [],
            'road_name': row['road_name'] if pd.notna(row['road_name']) else None
        }

    labels = poi_df.to_dict('records')
    return nodes, edges, config, labels


# ==========================================
# CA 仿真引擎
# ==========================================
class Simulation:
    def __init__(self, edges, config):
        self.edges = edges
        self.all_config = config  # 保存完整配置供时间轴使用
        self.base_config = config.get('base', {})
        self.delta_config = {}
        self.current_scenario_desc = "默认场景"
        self.tick = 0
        self.grids = {eid: [[0] * e['cell_count'] for _ in range(e['lanes'])] for eid, e in edges.items()}
        self.ever_had_vehicle = set()

        # 时间尺度相关初始化
        time_cfg = config.get('time_settings', {})
        self.seconds_per_tick = time_cfg.get('seconds_per_tick', 6)
        self.start_seconds = time_str_to_seconds(time_cfg.get('start_time', '06:00'))
        self.end_seconds = time_str_to_seconds(time_cfg.get('end_time', '22:30'))
        self.current_time_str = time_cfg.get('start_time', '06:00') + ":00"

        self.timeline = config.get('timeline', [])
        self.timeline_idx = 0
        self.next_switch_seconds = float('inf')
        self.next_scenario_name = None
        self._init_timeline_state()
        # 统计图表存储结构
        self.stats = {'time': [], 'avg_speed': [], 'density': [], 'flow': [], 'total_vehicles': []}
        # 计算最大tick数用于自动结束仿真
        self.max_ticks = (self.end_seconds - self.start_seconds) // self.seconds_per_tick
        # 构建物理路段配对
        self.segments = self._build_physical_segments(edges)
        # 创建路段级双方向网格
        self.seg_grids = {}
        for i, seg in enumerate(self.segments):
            cells = edges[seg['forward_edge']]['cell_count']
            total_lanes = seg['fwd_lanes'] + seg['rev_lanes']
            self.seg_grids[i] = [[0] * cells for _ in range(total_lanes)]
    def _build_physical_segments(self, edges):
        segments = []
        paired = set()
        for eid_a, ea in edges.items():
            if eid_a in paired: continue
            rev = None
            for eid_b, eb in edges.items():
                if eid_b == eid_a or eid_b in paired: continue
                if ea['start_node'] == eb['end_node'] and ea['end_node'] == eb['start_node']:
                    rev = eid_b
                    break
            if rev:
                paired.add(eid_a);
                paired.add(rev)
                segments.append({
                    'forward_edge': eid_a,
                    'reverse_edge': rev,
                    'fwd_lanes': ea['lanes'],
                    'rev_lanes': edges[rev]['lanes'],
                    'total_lanes': ea['lanes'] + edges[rev]['lanes'],
                    'cells': ea['cell_count'],
                    'is_flex': ea.get('is_flex', 0) or edges[rev].get('is_flex', 0)
                })
            else:
                paired.add(eid_a)
                segments.append({
                    'forward_edge': eid_a,
                    'reverse_edge': None,
                    'fwd_lanes': ea['lanes'],
                    'rev_lanes': 0,
                    'total_lanes': ea['lanes'],
                    'cells': ea['cell_count'],
                    'is_flex': ea.get('is_flex', 0)
                })
        return segments
    def _init_timeline_state(self):
        """初始化时间轴，找到起始时刻对应的场景和下一个切换点"""
        for i, slot in enumerate(self.timeline):
            start_s = time_str_to_seconds(slot['start'])
            end_s = time_str_to_seconds(slot['end'])
            if start_s <= self.start_seconds < end_s:
                self.timeline_idx = i
                self._apply_scenario(slot['scenario'])
                break

        if self.timeline_idx < len(self.timeline) - 1:
            next_slot = self.timeline[self.timeline_idx + 1]
            self.next_switch_seconds = time_str_to_seconds(next_slot['start'])
            self.next_scenario_name = next_slot['scenario']

    def _apply_scenario(self, scenario_name):
        """核心方法：仅改变概率参数，绝不触碰 grids（保留场内车辆）"""
        if scenario_name == 'base':
            self.delta_config = {}
            self.current_scenario_desc = "默认场景"
        else:
            self.delta_config = self.all_config.get(scenario_name, {})
            self.current_scenario_desc = self.delta_config.get('description', scenario_name)

    def set_scenario(self, scenario_name):
        """供外部（如按键）手动调用，行为与自动切换保持一致：不清空车辆"""
        self._apply_scenario(scenario_name)

    def _get_real_prob(self, poi_type):
        if not poi_type: return 0
        base_p = self.base_config.get(poi_type, {}).get('spawn_prob', 0)
        delta_p = self.delta_config.get(poi_type, {}).get('delta_prob', 0)
        return max(0.0, min(1.0, base_p + delta_p))

    def spawn_vehicles(self):
        for eid, edata in self.edges.items():
            if edata['spawn_dir'] != 'Out': continue
            prob = self._get_real_prob(edata['poi_type'])
            if prob <= 0: continue
            for l in range(edata['lanes']):
                if self.grids[eid][l][0] == 0 and random.random() < prob:
                    self.grids[eid][l][0] = 1

    def _try_transfer(self, eid, l, v, pending, new_grids):
        # ★ 驶入 POI 的边（spawn_dir == 'In'）视为终点，直接吸收车辆
        if self.edges[eid].get('spawn_dir') == 'In':
            return True

        next_edges = self.edges[eid]['connected_edges']
        if not next_edges:
            return True

        candidates = list(next_edges)
        random.shuffle(candidates)
        for neid in candidates:
            nedata = self.edges[neid]
            for tl in range(nedata['lanes']):
                key = (neid, tl)
                if key not in pending and new_grids[neid][tl][0] == 0:
                    pending[key] = max(min(v, nedata['max_speed']), 1)
                    return True
        return False

    def update_ca(self):
        BORROW_FWD_BASE = 100
        BORROW_REV_BASE = 200

        # 1) 清空路段网格，映射当前边网格车辆
        for seg_idx, seg in enumerate(self.segments):
            cells = seg['cells']
            total = seg['total_lanes']
            fwd = seg['fwd_lanes']
            seg_grid = self.seg_grids[seg_idx]
            for l in range(total):
                for c in range(cells):
                    seg_grid[l][c] = 0
            eid_fwd = seg['forward_edge']
            for l in range(fwd):
                for c in range(cells):
                    v = self.grids[eid_fwd][l][c]
                    if v != 0:
                        seg_grid[l][c] = v
            if seg['reverse_edge']:
                eid_rev = seg['reverse_edge']
                rev_lanes = seg['rev_lanes']
                for l in range(rev_lanes):
                    for c in range(cells):
                        v = self.grids[eid_rev][l][c]
                        if v != 0:
                            row = fwd + (rev_lanes - 1 - l)
                            col = cells - 1 - c
                            seg_grid[row][col] = v

        # 2) 换道
        for seg_idx, seg in enumerate(self.segments):
            if seg['total_lanes'] < 2:
                continue
            grid = self.seg_grids[seg_idx]
            self._lane_change_segment(grid, seg)

        # 3) 双方向 NaSch 跟驰更新
        new_seg_grids = {}
        for seg_idx, seg in enumerate(self.segments):
            grid = self.seg_grids[seg_idx]
            cells = seg['cells']
            fwd = seg['fwd_lanes']
            total = seg['total_lanes']
            new_grid = [[0] * cells for _ in range(total)]

            for l in range(total):
                for c in range(cells):
                    state = grid[l][c]
                    if state == 0:
                        continue

                    is_borrow_fwd = (BORROW_FWD_BASE <= state < BORROW_FWD_BASE + 10)
                    is_borrow_rev = (BORROW_REV_BASE <= state < BORROW_REV_BASE + 10)
                    is_borrow = is_borrow_fwd or is_borrow_rev

                    if state == -1:
                        dir = 1 if l < fwd else -1
                        is_queued = True
                        spd = 0
                    else:
                        is_queued = False
                        if is_borrow_fwd:
                            spd = state - BORROW_FWD_BASE
                            dir = 1
                        elif is_borrow_rev:
                            spd = state - BORROW_REV_BASE
                            dir = -1
                        else:
                            spd = state
                            dir = 1 if l < fwd else -1

                    if is_queued:
                        # 已经在末端的排队车保持 -1
                        if (dir == 1 and c == cells - 1) or (dir == -1 and c == 0):
                            new_grid[l][c] = -1
                            continue
                        v = 0
                    else:
                        v = spd

                    # 加速
                    v = min(v + 1, 3)

                    # 计算前方空距（按本车方向）
                    gap = 0
                    if dir == 1:
                        for i in range(c + 1, cells):
                            if grid[l][i] == 0:
                                gap += 1
                            else:
                                break
                    else:
                        for i in range(c - 1, -1, -1):
                            if grid[l][i] == 0:
                                gap += 1
                            else:
                                break
                    v = min(v, gap)

                    # 随机慢化
                    if random.random() < 0.1:
                        v = max(v - 1, 0)

                    # 保留借道标记
                    if is_borrow_fwd:
                        write_state = BORROW_FWD_BASE + v
                    elif is_borrow_rev:
                        write_state = BORROW_REV_BASE + v
                    else:
                        write_state = v

                    # 位置更新
                    if v == 0:
                        if is_borrow:
                            new_grid[l][c] = write_state
                        else:
                            new_grid[l][c] = -1
                        continue

                    if dir == 1:
                        target = c + v
                        if target >= cells:
                            if is_borrow_fwd:
                                # 借道车尝试返回正向最右车道
                                orig_l = fwd - 1
                                if orig_l >= 0 and grid[orig_l][cells - 1] == 0:
                                    new_grid[orig_l][cells - 1] = v
                                else:
                                    new_grid[l][c] = BORROW_FWD_BASE
                            else:
                                # 普通车推到末端元胞
                                if new_grid[l][cells - 1] == 0:
                                    new_grid[l][cells - 1] = -1
                                else:
                                    new_grid[l][c] = -1   # 末端被占，留在原位
                        else:
                            new_grid[l][target] = write_state
                    else:  # dir == -1
                        target = c - v
                        if target < 0:
                            if is_borrow_rev:
                                orig_l = fwd
                                if orig_l < total and grid[orig_l][0] == 0:
                                    new_grid[orig_l][0] = v
                                else:
                                    new_grid[l][c] = BORROW_REV_BASE
                            else:
                                # 普通车推到末端元胞（反向末端为 0）
                                if new_grid[l][0] == 0:
                                    new_grid[l][0] = -1
                                else:
                                    new_grid[l][c] = -1
                        else:
                            new_grid[l][target] = write_state

            new_seg_grids[seg_idx] = new_grid

        # 4) 将路段网格写回边网格，并处理末端转移
        for eid in self.edges:
            self.grids[eid] = [[0] * self.edges[eid]['cell_count'] for _ in range(self.edges[eid]['lanes'])]

        pending_ends = []
        for seg_idx, seg in enumerate(self.segments):
            new_grid = new_seg_grids[seg_idx]
            cells = seg['cells']
            fwd = seg['fwd_lanes']
            # 正向边
            eid_fwd = seg['forward_edge']
            for l in range(fwd):
                for c in range(cells):
                    v = new_grid[l][c]
                    if v == 0:
                        continue
                    raw = v
                    if BORROW_FWD_BASE <= v < BORROW_FWD_BASE + 10:
                        raw = v - BORROW_FWD_BASE
                        if raw == 0:
                            raw = -1
                    elif BORROW_REV_BASE <= v < BORROW_REV_BASE + 10:
                        raw = v - BORROW_REV_BASE
                        if raw == 0:
                            raw = -1
                    self.grids[eid_fwd][l][c] = raw
                    if c == cells - 1 and v == -1:
                        pending_ends.append((eid_fwd, l, 'fwd'))

            # 反向边
            if seg['reverse_edge']:
                eid_rev = seg['reverse_edge']
                rev_lanes = seg['rev_lanes']
                for l in range(rev_lanes):
                    for c in range(cells):
                        row = fwd + (rev_lanes - 1 - l)
                        col_rev = cells - 1 - c
                        v = new_grid[row][col_rev]
                        if v == 0:
                            continue
                        raw = v
                        is_borrow = False
                        if BORROW_FWD_BASE <= v < BORROW_FWD_BASE + 10:
                            raw = v - BORROW_FWD_BASE
                            is_borrow = True
                            if raw == 0:
                                raw = -1
                        elif BORROW_REV_BASE <= v < BORROW_REV_BASE + 10:
                            raw = v - BORROW_REV_BASE
                            is_borrow = True
                            if raw == 0:
                                raw = -1
                        self.grids[eid_rev][l][c] = raw
                        if c == cells - 1 and v == -1 and not is_borrow:
                            pending_ends.append((eid_rev, l, 'rev'))

        # 消散/转移处理
        pending_transfer = {}
        for (eid, l, _) in pending_ends:
            transferred = self._try_transfer(eid, l, 1, pending_transfer, self.grids)
            if transferred:
                self.grids[eid][l][-1] = 0

        for (eid, l), speed in pending_transfer.items():
            if self.grids[eid][l][0] == 0:
                self.grids[eid][l][0] = speed
    def _lane_change_segment(self, grid, seg):
        # 执行换道，包括同向和可选的对向借道
        BORROW_FWD_BASE = 100
        BORROW_REV_BASE = 200
        cells = seg['cells']
        total_lanes = seg['total_lanes']
        fwd_lanes = seg['fwd_lanes']
        safety_end = 3
        for c in range(cells - safety_end):
            for l in range(total_lanes):
                v = grid[l][c]
                if v <= 0: continue
                # 确定本车属于哪个区域
                if l < fwd_lanes:  # 正向区
                    dir = 1
                    my_lanes = range(fwd_lanes)
                else:              # 反向区
                    dir = -1
                    my_lanes = range(fwd_lanes, total_lanes)
                desired = min(v + 1, 3)
                # 计算本车前方空距
                gap_cur = 0
                if dir == 1:
                    for i in range(c + 1, cells):
                        if grid[l][i] == 0: gap_cur += 1
                        else: break
                else:
                    for i in range(c - 1, -1, -1):
                        if grid[l][i] == 0: gap_cur += 1
                        else: break
                if gap_cur >= desired: continue
                # 同向换道
                targets = []
                if l > min(my_lanes): targets.append(l - 1)
                if l < max(my_lanes): targets.append(l + 1)
                for tl in targets:
                    if tl not in my_lanes: continue
                    gap_tgt = 0
                    if dir == 1:
                        for i in range(c + 1, cells):
                            if grid[tl][i] == 0: gap_tgt += 1
                            else: break
                    else:
                        for i in range(c - 1, -1, -1):
                            if grid[tl][i] == 0: gap_tgt += 1
                            else: break
                    back_safe = (c == 0 and dir == 1) or (c == cells-1 and dir == -1) or (grid[tl][c - dir] == 0)
                    if grid[tl][c] == 0 and back_safe and gap_tgt > gap_cur:
                        grid[tl][c] = v
                        grid[l][c] = 0
                        break
                # 对向借道超车
                if seg['is_flex'] == 1:
                    if dir == 1 and l == fwd_lanes - 1:
                        tl = fwd_lanes
                    elif dir == -1 and l == fwd_lanes:
                        tl = fwd_lanes - 1
                    else:
                        continue
                    if dir == 1:
                        target_c = cells - 1 - c
                    else:
                        target_c = cells - 1 - c
                    if target_c < 0 or target_c >= cells: continue
                    back_safe = True
                    if grid[tl][target_c] == 0 and back_safe:
                        if dir == 1:
                            grid[tl][target_c] = BORROW_FWD_BASE + v
                        else:
                            grid[tl][target_c] = BORROW_REV_BASE + v
                        grid[l][c] = 0
                        break

    def step(self):
        self.update_ca()
        self.spawn_vehicles()
        for eid, grid in self.grids.items():
            for lane in grid:
                if any(cell != 0 for cell in lane):
                    self.ever_had_vehicle.add(eid)
        self.tick += 1
        # 时间推进与自动场景切换
        current_seconds = self.start_seconds + self.tick * self.seconds_per_tick
        h = int(current_seconds // 3600)
        m = int((current_seconds % 3600) // 60)
        s = int(current_seconds % 60)
        self.current_time_str = f"{h:02d}:{m:02d}:{s:02d}"
        # 统计图标计算逻辑
        total_cells = sum(e['cell_count'] * e['lanes'] for e in self.edges.values())
        total_vehicles = sum(1 for g in self.grids.values() for lane in g for v in lane if v != 0)
        total_speed = sum(v for g in self.grids.values() for lane in g for v in lane if v > 0)
        self.stats['time'].append(current_seconds / 3600)  # 记录小时数
        self.stats['density'].append(total_vehicles / total_cells)
        self.stats['total_vehicles'].append(total_vehicles)
        # 乘以元胞长度(7.5m) 和 每秒仿真步数(10 tick/s) 得到 m/s
        # 虚拟速度 (m/s) = 平均 cell/tick × 元胞长度 (m) / 每 tick 虚拟秒
        avg_speed_mps = (total_speed / total_vehicles) * 7.5 / self.seconds_per_tick if total_vehicles else 0
        self.stats['avg_speed'].append(avg_speed_mps)
        # 流量：统计本 tick 通过某断面的车辆数，简化可用总移动车辆数替代
        moving_vehicles = sum(1 for g in self.grids.values() for lane in g for v in lane if v > 0)
        self.stats['flow'].append(moving_vehicles)

        # 检查是否跨越了时间轴上的下一个节点
        if current_seconds >= self.next_switch_seconds and self.next_scenario_name is not None:
            self._apply_scenario(self.next_scenario_name)
            # 指针后移，寻找再下一个节点
            self.timeline_idx += 1
            if self.timeline_idx < len(self.timeline) - 1:
                next_slot = self.timeline[self.timeline_idx + 1]
                self.next_switch_seconds = time_str_to_seconds(next_slot['start'])
                self.next_scenario_name = next_slot['scenario']
            else:
                self.next_switch_seconds = float('inf')
                self.next_scenario_name = None

# ==========================================
# 可视化引擎
# ==========================================
class Visualizer:
    def __init__(self, nodes, edges, labels,segments):
        pygame.init()
        self.width, self.height = 1400, 900
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("校园交通流 CA 仿真")
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.Font(r'C:\Windows\Fonts\SIMHEI.TTF', 18)
        except:
            self.font = pygame.font.Font(None, 18)
        self.edges = edges
        self.labels = labels
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.dragging = False
        self.last_mouse_pos = (0, 0)
        self.LANE_WIDTH = 1
        self._calc_base_transform(edges)
        self.segments = segments
        self.bg_surface = None  # 用于存放预渲染的道路背景
        self.bg_zoom = None  # 生成背景时的 zoom 值
        self.bg_pan_x = None  # 生成背景时的 pan_x
        self.bg_pan_y = None  # 生成背景时的 pan_y
        self.bg_size = None  # 生成背景时的窗口 (width, height)
    def _calc_base_transform(self, edges):
        all_x, all_y = [], []
        for e in edges.values():
            for coords in e['coords'].values():
                for x, y in coords: all_x.append(x); all_y.append(y)
        for lb in self.labels: all_x.append(lb['x']); all_y.append(lb['y'])
        margin = 120
        dx = max(all_x) - min(all_x) or 1
        dy = max(all_y) - min(all_y) or 1
        sx = (self.width - 2 * margin) / dx
        sy = (self.height - 2 * margin) / dy
        self.base_scale = min(sx, sy)
        self.base_offset_x = margin + ((self.width - 2 * margin) - dx * self.base_scale) / 2 - min(
            all_x) * self.base_scale
        self.base_offset_y = margin + ((self.height - 2 * margin) - dy * self.base_scale) / 2 - min(
            all_y) * self.base_scale



    def _transform(self, x, y):
        sx = x * self.base_scale + self.base_offset_x
        sy = self.height - (y * self.base_scale + self.base_offset_y)
        cx, cy = self.width / 2, self.height / 2
        zoomed_x = cx + (sx - cx) * self.zoom + self.pan_x
        zoomed_y = cy + (sy - cy) * self.zoom + self.pan_y
        return (int(zoomed_x), int(zoomed_y))

    def _calc_normals(self, screen_pts):
        n = len(screen_pts)
        normals = []
        for i in range(n):
            if i == 0:
                dx = screen_pts[1][0] - screen_pts[0][0]
                dy = screen_pts[1][1] - screen_pts[0][1]
            elif i == n - 1:
                dx = screen_pts[-1][0] - screen_pts[-2][0]
                dy = screen_pts[-1][1] - screen_pts[-2][1]
            else:
                dx = screen_pts[i + 1][0] - screen_pts[i - 1][0]
                dy = screen_pts[i + 1][1] - screen_pts[i - 1][1]
            length = math.hypot(dx, dy)
            if length == 0:
                normals.append((0, 0))
            else:
                normals.append((-dy / length, dx / length))
        return normals

    def _offset_points(self, pts, normals, offset):
        return [(int(pts[i][0] + normals[i][0] * offset), int(pts[i][1] + normals[i][1] * offset)) for i in
                range(len(pts))]

    def _draw_dashed_line(self, surface, color, points, width=1):
        dash_len = max(4, int(10 * self.zoom))
        gap_len = max(3, int(7 * self.zoom))
        dist = 0
        drawing = True
        for i in range(len(points) - 1):
            seg_len = math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
            if drawing:
                pygame.draw.line(surface, color, points[i], points[i + 1], width)
            dist += seg_len
            threshold = dash_len if drawing else gap_len
            if dist >= threshold:
                dist = 0
                drawing = not drawing

    def _draw_road_background(self):
        """将静态道路（路面、边线、车道线）绘制到 bg_surface 上"""
        # 如果背景尺寸与当前窗口不符，重新创建
        if self.bg_surface is None or self.bg_surface.get_size() != (self.width, self.height):
            self.bg_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        self.bg_surface.fill((0, 0, 0, 0))   # 透明背景
        lane_w = self.LANE_WIDTH * self.zoom

        for seg in self.segments:
            eid_fwd = seg['forward_edge']
            edata_fwd = self.edges[eid_fwd]
            coords_fwd = edata_fwd['coords'][0]
            pts = [self._transform(x, y) for x, y in coords_fwd]
            if len(pts) < 2: continue
            normals = self._calc_normals(pts)

            total_lanes = seg['total_lanes']
            fwd_lanes = seg['fwd_lanes']
            rev_lanes = seg['rev_lanes']
            half_road = lane_w * total_lanes / 2.0

            left_edge = self._offset_points(pts, normals, -half_road)
            right_edge = self._offset_points(pts, normals, half_road)
            polygon = left_edge + list(reversed(right_edge))
            pygame.draw.polygon(self.bg_surface, (55, 55, 55), polygon)

            edge_w = max(1, int(1.5 * self.zoom))
            pygame.draw.lines(self.bg_surface, (110, 110, 110), False, left_edge, edge_w)
            pygame.draw.lines(self.bg_surface, (110, 110, 110), False, right_edge, edge_w)

            if total_lanes > 1:
                for i in range(1, total_lanes):
                    offset = lane_w * (i - total_lanes / 2.0)
                    sep_pts = self._offset_points(pts, normals, offset)
                    self._draw_dashed_line(self.bg_surface, (96, 96, 96), sep_pts, max(1, int(self.zoom)))

        # 记录当前生成背景时的状态
        self.bg_zoom = self.zoom
        self.bg_pan_x = self.pan_x
        self.bg_pan_y = self.pan_y
        self.bg_size = (self.width, self.height)

    def draw(self, grids, tick, scenario_desc, current_time_str, speed_multiplier):
        # 清屏
        self.screen.fill((30, 30, 35))
        # --- 检查是否需要重建背景 ---
        if (self.bg_surface is None or
            self.zoom != self.bg_zoom or
            self.pan_x != self.bg_pan_x or
            self.pan_y != self.bg_pan_y or
            (self.width, self.height) != self.bg_size):
            self._draw_road_background()

        # 先将背景绘制到主屏幕
        self.screen.blit(self.bg_surface, (0, 0))

        # ---------- 绘制车辆 ----------
        lane_w = self.LANE_WIDTH * self.zoom
        car_size = max(2, int(lane_w * 0.7))
        for seg in self.segments:
            eid_fwd = seg['forward_edge']
            edata_fwd = self.edges[eid_fwd]
            coords_fwd = edata_fwd['coords'][0]
            pts = [self._transform(x, y) for x, y in coords_fwd]
            if len(pts) < 2: continue
            normals = self._calc_normals(pts)

            total_lanes = seg['total_lanes']
            fwd_lanes = seg['fwd_lanes']
            rev_lanes = seg['rev_lanes']

            for lane_idx_in_seg in range(total_lanes):
                if seg['reverse_edge'] is None:
                    edge_id = eid_fwd
                    lane_id = lane_idx_in_seg
                else:
                    if lane_idx_in_seg < fwd_lanes:
                        edge_id = eid_fwd
                        lane_id = lane_idx_in_seg
                    else:
                        rev_lane_local = lane_idx_in_seg - fwd_lanes
                        edge_id = seg['reverse_edge']
                        lane_id = rev_lanes - 1 - rev_lane_local

                off = lane_w * (lane_idx_in_seg - (total_lanes - 1) / 2.0)
                lane_pts = self._offset_points(pts, normals, off)

                if edge_id in grids and lane_id < len(grids[edge_id]):
                    for c, v in enumerate(grids[edge_id][lane_id]):
                        if v != 0 and c < len(lane_pts):
                            cx, cy = lane_pts[c]
                            if v == -1 or v <= 1:
                                color = (255, 60, 60)
                            elif v <= 3:
                                color = (255, 200, 0)
                            else:
                                color = (0, 255, 100)
                            pygame.draw.circle(self.screen, color, (cx, cy), car_size // 2)

        # ---------- POI 标签 ----------
        for lb in self.labels:
            sx, sy = self._transform(lb['x'], lb['y'])
            text = self.font.render(lb['name'], True, (100, 200, 255))
            self.screen.blit(text, (sx - text.get_width() // 2, sy - 25))
            pygame.draw.circle(self.screen, (100, 200, 255), (sx, sy), 4)

        # ---------- HUD ----------
        total_veh = sum(1 for g in grids.values() for lane in g for v in lane if v != 0)
        hud = self.font.render(
            f"时间:{current_time_str} | 场景:{scenario_desc} | 车辆:{total_veh} | 倍速:{speed_multiplier}x | 缩放:{self.zoom:.1f}",
            True, (255, 255, 255))
        self.screen.blit(hud, (20, 15))
        tip = self.font.render("↑↓加速减速 | 滚轮缩放 | 拖拽平移 | 1/2/3强制切换场景 | 空格暂停", True, (160, 160, 160))
        self.screen.blit(tip, (20, self.height - 30))
        pygame.display.flip()

# ==========================================
# 主程序
# ==========================================
def run_simulation(output_dir):
    nodes, edges, config, labels = load_data(output_dir)
    #z最大速度设置
    for e in edges.values():
        e['max_speed'] = 3
    sim = Simulation(edges, config)
    # 覆盖 JSON 配置中的 seconds_per_tick，使每个 tick 对应 2.25 虚拟秒
    sim.seconds_per_tick = 2.25
    # 重新计算最大 tick 数，确保虚拟时间跑到 22:30
    sim.max_ticks = int((sim.end_seconds - sim.start_seconds) / sim.seconds_per_tick)
    viz = Visualizer(nodes, edges, labels,sim.segments)

    scenario_map = {pygame.K_1: 'base', pygame.K_2: 'to_class_peak', pygame.K_3: 'after_class_peak'}
    # tick/s(real)设置
    SIM_STEPS_PER_SECOND = 15
    SIM_STEP_INTERVAL = 1.0 / SIM_STEPS_PER_SECOND
    accumulator = 0.0
    RENDER_FPS = 30

    # --- 新增：时间倍率控制 倍速控制 ---
    speed_multiplier = 1.0

    running, paused = True, False
    clock = pygame.time.Clock()

    while running:
        dt = clock.tick(RENDER_FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                viz.width, viz.height = event.size
                viz.screen = pygame.display.set_mode((viz.width, viz.height), pygame.RESIZABLE)
                viz._calc_base_transform(viz.edges)
                viz.bg_surface = None  # 强制重建背景
            elif event.type == pygame.MOUSEWHEEL:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                old_zoom = viz.zoom
                viz.zoom *= 1.1 if event.y > 0 else 0.9
                viz.zoom = max(0.1, min(viz.zoom, 10.0))
                viz.pan_x = mouse_x - (mouse_x - viz.pan_x) * (viz.zoom / old_zoom)
                viz.pan_y = mouse_y - (mouse_y - viz.pan_y) * (viz.zoom / old_zoom)
                viz.bg_surface = None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    viz.dragging = True
                    viz.last_mouse_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    viz.dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if viz.dragging:
                    dx = event.pos[0] - viz.last_mouse_pos[0]
                    dy = event.pos[1] - viz.last_mouse_pos[1]
                    viz.pan_x += dx
                    viz.pan_y += dy
                    viz.last_mouse_pos = event.pos
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                # --- 新增：加速减速按键 ---
                elif event.key == pygame.K_UP:
                    speed_multiplier = min(speed_multiplier + 1, 8)  # 最大 64 倍速
                elif event.key == pygame.K_DOWN:
                    speed_multiplier = max(speed_multiplier - 1, 1)  # 最慢 0.25 倍速(慢放)
                elif event.key in scenario_map:
                    sim.set_scenario(scenario_map[event.key])

        # --- 修改：累加器加入倍率 ---
        if not paused:
            accumulator += dt * speed_multiplier
            while accumulator >= SIM_STEP_INTERVAL:
                sim.step()
                if sim.tick >= sim.max_ticks:  # 新增
                    running = False
                    break
                accumulator -= SIM_STEP_INTERVAL

        # --- 修改：传递新参数给 draw ---
        viz.draw(sim.grids, sim.tick, sim.current_scenario_desc, sim.current_time_str, speed_multiplier)
    # 仿真结束，释放pygame
    pygame.quit()
    import gc
    gc.collect()
    # ========== 仿真结束，绘制统计图 ==========

    font_path = get_font_path()
    chinese_font = FontProperties(fname=font_path, size=12) if font_path else FontProperties(size=12)
    fig, axes = plt.subplots(4, 1, figsize=(10, 6))
    titles = ['平均速度 (m/s)', '密度 (veh/cell)', '流量 (veh/tick)','车辆总数']
    keys = ['avg_speed', 'density', 'flow','total_vehicles']
    colors = ['red', 'blue', 'green','purple']

    for ax, title, key, color in zip(axes, titles, keys, colors):
        ax.plot(sim.stats['time'], sim.stats[key], color=color, linewidth=0.8)
        ax.set_title(title, fontproperties=chinese_font)
        ax.set_xlabel('时间 (时)', fontproperties=chinese_font)
        ax.set_xlim(6, 22.5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    # agg保存图片
    #plt.show()
    try:
        # ... 原有的四子图和散点图代码 ...
        plt.show()
    except Exception as e:
        print(f"绘图失败：{e}")
    # ========== 密度-流量基本图（散点图） ==========
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(sim.stats['density'], sim.stats['flow'], s=1, color='orange', alpha=0.6)
    ax2.set_title('密度-流量基本图', fontproperties=chinese_font)
    ax2.set_xlabel('密度 (veh/cell)', fontproperties=chinese_font)
    ax2.set_ylabel('流量 (veh/tick)', fontproperties=chinese_font)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    # agg保存图片
    #plt.show()
    try:
        # ... 原有的四子图和散点图代码 ...
        plt.show()
    except Exception as e:
        print(f"绘图失败：{e}")



def setup_and_run():
    root = tk.Tk()
    root.withdraw()

    # 选择道路线 SHP
    line_shp = filedialog.askopenfilename(title="选择道路线SHP文件", filetypes=[("Shapefile", "*.shp")])
    if not line_shp:
        return
    # 选择 POI 点 SHP
    point_shp = filedialog.askopenfilename(title="选择POI点SHP文件", filetypes=[("Shapefile", "*.shp")])
    if not point_shp:
        return
    # 选择配置文件 (JSON)
    config_src = filedialog.askopenfilename(title="选择仿真配置文件", filetypes=[("JSON", "*.json")])
    if not config_src:
        return
    # 销毁Tkinter根窗口，释放事件循环，否则会不显示图表
    root.destroy()

    # 创建临时输出目录
    output_dir = tempfile.mkdtemp(prefix="CA_output_")
    print(f"临时目录：{output_dir}")

    # 1) 运行路网预处理
    process_network(line_shp, point_shp, output_dir)

    # 2) 复制用户选择的配置文件
    shutil.copy(config_src, os.path.join(output_dir, 'demand_config.json'))

    # 3) 启动仿真
    run_simulation(output_dir)

    # 4) 清理临时文件（可选）
    shutil.rmtree(output_dir, ignore_errors=True)

# if __name__ == "__main__":
#     setup_and_run()
if __name__ == "__main__":
    try:
        setup_and_run()
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        # 尝试销毁之前可能未关闭的pygame窗口
        try:
            pygame.quit()
        except:
            pass
        # 创建一个新的Tk根窗口来显示错误（如果需要）
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("程序运行出错", f"错误信息：\n{str(e)}\n\n详细堆栈：\n{tb}")
        root.destroy()
