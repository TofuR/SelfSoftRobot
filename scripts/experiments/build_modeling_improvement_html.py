#!/usr/bin/env python3
"""Build a canonical, portable visual-report input from completed modeling results.

The HTML is packaged by the Data Analytics portable report builder. This script
only prepares reviewed datasets, narrative and native chart specifications.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import json
import math
import sqlite3
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

LABEL = {
    'hov': 'HOV', 'chen_direction': 'Chen 方向网络', 'bezier_gru': 'Yu Bézier–GRU',
    'oscillator': 'Krauss 振子', 'koopman': 'Koopman', 'pcc': 'PCC',
    'mlp': '静态 MLP', 'linear': '线性回归', 'hov_no_play': 'HOV 无 play',
    'hov_no_maxwell': 'HOV 无 Maxwell', 'hov_no_memory': 'HOV 无记忆',
}
GROUP = {'seq_20260819_172644': '172644 · 四通道运动',
         'seq_20260819_181044': '181044 · 下段运动',
         'seq_20260819_181548': '181548 · 上段运动'}
METRICS = {'mean_node_mm': ('节点平均误差', 'mm'), 'node_rmse_mm': ('节点 RMSE', 'mm'), 'endpoint_mm': ('末端误差', 'mm'),
           'mask_iou': ('Mask IoU', ''), 'mask_dice': ('Mask Dice', '')}


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def project_snapshot(report, output):
    """Materialize the reviewed chart rows and execute their source projections."""
    statements = {}
    table_names = {}
    with sqlite3.connect(output / 'chart_snapshot.sqlite') as connection:
        for item in report.charts + report.tables:
            name, source_id = item['dataset'], item['sourceId']
            rows = report.datasets[name]
            columns = list(rows[0])
            table = 'chart_' + name
            types = {key: ('INTEGER' if all(type(row[key]) is int for row in rows) else
                           'REAL' if all(isinstance(row[key], (int, float)) for row in rows) else 'TEXT')
                     for key in columns}
            connection.execute(f'DROP TABLE IF EXISTS "{table}"')
            definition = ', '.join(f'"{key}" {types[key]}' for key in columns)
            connection.execute(f'CREATE TABLE "{table}" ({definition})')
            connection.executemany(f'INSERT INTO "{table}" VALUES ({",".join("?" for _ in columns)})',
                                   [[row[key] for key in columns] for row in rows])
            sql = f'SELECT * FROM "{table}" ORDER BY rowid;'
            cursor = connection.execute(sql)
            projected = [dict(zip(columns, row)) for row in cursor.fetchall()]
            assert projected == rows, name
            report.datasets[name] = projected
            statements.setdefault(source_id, []).append(sql)
            table_names.setdefault(source_id, []).append(table)
    for source in report.sources:
        query = source['query']
        query.update(sql='\n'.join(statements[source['id']]), language='sql',
                     engine='Python / NumPy metrics; SQLite chart snapshot projection')
        query['description'] += ' 指标由Python依据保存结果计算；所附SQL已在图表快照SQLite中执行，逐行投影为最终展示数据。'
        query['tables_used'].extend(table_names[source['id']])
        query['tables_used'].append((output.relative_to(ROOT) / 'chart_snapshot.sqlite').as_posix())
    (output / 'analysis.sql').write_text('\n\n'.join(
        '-- ' + name + '\n' + '\n'.join(sql) for name, sql in statements.items()) + '\n')


class Report:
    def __init__(self, title, sources):
        self.title, self.sources = title, sources
        self.blocks, self.charts, self.tables, self.datasets, self.chart_map = [], [], [], {}, []

    def md(self, name, text, source=None):
        block = dict(id=name, type='markdown', body=text)
        if source:
            block['sourceId'] = source
        self.blocks.append(block)

    def chart(self, name, title, rows, *, kind='horizontalBar', x='model_label', y='value',
              source='results', unit='mm', color=None, question='', description='',
              zero=False, heat=False):
        if heat:
            # The shared heatmap pivots a numeric measure by its series field.
            y, color = color, y
        self.datasets[name] = rows
        chart = dict(id=name, title=title, type=kind, dataset=name, sourceId=source,
                     layout='full', maxRows=max(500, len(rows)), valueFormat='number',
                     encodings={'x': {'field': x, 'type': 'quantitative' if kind in ('line', 'scatter') else 'nominal'},
                                'y': {'field': y, 'type': 'quantitative'}},
                     labels={'values': 'all' if kind in ('horizontalBar', 'bar') else 'none'},
                     settings={'sort': 'none', 'showValues': kind in ('horizontalBar', 'bar')},
                     palette={'kind': 'sequential', 'name': 'blue'})
        if unit:
            chart['unit'] = unit
            if not heat:
                chart['encodings']['y']['label'] = unit
        if color:
            chart['encodings']['color'] = {'field': color, 'type': 'nominal'}
            chart['palette'] = {'kind': 'sequential' if heat else 'categorical', 'name': 'blue' if heat else 'default'}
            if not heat:
                chart['legend'] = {'position': 'bottom', 'sort': 'spec'}
        if kind == 'line':
            chart['settings']['showPoints'] = 'never'
            chart['xAxisTitle'] = {'epoch':'Epoch', 'node':'节点编号（基座至末端）',
                                  'error_mm':'单帧节点平均误差（mm）'}.get(x, x)
            chart['yAxisTitle'] = '累计比例' if y == 'fraction' else '误差（mm）'
        if kind == 'scatter':
            chart['encodings']['label'] = {'field': 'point_label', 'type': 'text'}
            chart['xAxisTitle'] = {'seed': '训练 seed 编号', 'log10_parameters': '参数量（log10）'}.get(x, x)
        if zero:
            chart['referenceLines'] = [{'axis': 'y', 'value': 0, 'label': '与完整 HOV 相同', 'color': 'neutral', 'lineStyle': 'dashed'}]
        if description:
            chart.update(subtitle=description, showDescription=True)
        self.charts.append(chart)
        self.blocks.append(dict(id=name + '_block', type='chart', chartId=name))
        self.chart_map.append(dict(id=name, title=title, question=question or title, type=kind,
                                   rows=len(rows), fields=dict(x=x, y=y, color=color),
                                   source=source, palette=chart['palette']))

    def table(self, name, title, rows, columns, *, source='results', sort='order'):
        self.datasets[name] = rows
        self.tables.append(dict(id=name, title=title, dataset=name, sourceId=source,
                                layout='full', density='spacious', defaultSort={'field': sort, 'direction': 'asc'},
                                columns=[{'field': key, 'label': label, 'type': typ} for key, label, typ in columns]))
        self.blocks.append(dict(id=name + '_block', type='table', tableId=name))


def build(study, output):
    results = study / 'results'
    comparison, ablation = (read_json(results / f'{name}.json') for name in ('comparison', 'ablation'))
    assert comparison['status'] == ablation['status'] == 'complete'
    assert comparison['seeds'] == ablation['seeds'] == [0, 1, 2, 3, 4]
    main, abl = comparison['models'], ablation['models']
    models = list(dict.fromkeys(main + abl))
    seed_rows, sequence_rows = read_csv(results / 'per_seed.csv'), read_csv(results / 'per_sequence.csv')
    raw_summary = read_csv(results / 'summary.csv')
    summary = {row['model']: row for row in raw_summary}
    assert len(seed_rows) == 55 and len(sequence_rows) == 165 and len(summary) == 11
    numeric = lambda model, field: float(summary[model][field])
    value = lambda model, metric: numeric(model, metric + '_mean')
    sd = lambda model, metric: numeric(model, metric + '_std')
    paired = {(r['model'], int(r['seed'])): r for r in seed_rows}
    manifest, frozen = read_json(study / 'data/dataset_manifest.json'), read_json(study / 'frozen_configs.json')
    prefix = study.relative_to(ROOT).as_posix()
    stamp = datetime.now(timezone.utc).isoformat()
    sources = []

    def source(name, title, path, description, files, definitions):
        sources.append(dict(id=name, label=title, path=prefix + '/' + path,
                            query=dict(engine='Python / NumPy', language='python', description=description,
                                       tables_used=[prefix + '/' + f for f in files],
                                       metric_definitions=definitions, executed_at=stamp,
                                       filters=['三条固定序列', 'seed = 0,1,2,3,4', '固定划分', '5 Hz', 'H = 20'])))

    source('results', '全部五次重复的测试结果', 'results/summary.csv',
           '每次运行先合并三序列全部2958个测试帧，再对五个预定seed计算均值及样本标准差。',
           ['results/summary.csv', 'results/per_seed.csv', 'results/verification.json'],
           ['节点误差：逐帧15个对应节点欧氏距离的均值，单位mm', '末端误差：第15节点欧氏距离',
            'IoU和Dice：逐帧固定8mm管状投影与标签mask比较后求均值', '±为五个训练seed之间样本标准差，ddof=1'])
    source('sequences', '三个运动模式的测试结果', 'results/per_sequence.csv',
           '每条序列先在每个seed内汇总，再计算五次均值；总表仍按测试帧数合并。',
           ['results/per_sequence.csv', 'data/dataset_manifest.json'], ['按原始采集序列分组', '差值=对照−完整HOV，正误差差值表示HOV较好'])
    source('training', '已保存的训练曲线与耗时', 'training_summary.json',
           '55次运行均来自冻结配置；梯度模型训练100epoch，每5epoch验证，线性回归闭式拟合。',
           ['training_summary.json', 'formal/*/seed_*/history.json', 'formal/*/seed_*/run_manifest.json'],
           ['验证均距按所有验证帧合并', '训练墙钟时间包含参考拟合、初始化、训练及验证',
            'GPU2、3各两个并发任务；墙钟时间受并发影响', '存储系数含拟合参考buffer以及消融中停用的参数'])
    source('latency', '同卡完整窗口推理计时', 'latency/hov/seed_0.json',
           '物理GPU2，seed0，float32；20次预热后100次同步计时；所有方法串行执行。',
           ['latency/*/seed_0.json', 'frozen_configs.json'],
           ['输入驻留GPU，计入H20完整前向与毫米输出转换', 'B1为一次单窗口；B256为整批256窗口',
            'p50和p95为100次计时分位数，不是五seed标准差', '该协议测量独立窗口，而非缓存状态的在线单步接口'])
    source('statistics', '固定训练种子的配对统计', 'results/ablation.json',
           '双侧精确Wilcoxon为主检验；探索性配对t另成校正族。主对照35项、消融15项分别Holm校正。',
           ['results/comparison.json', 'results/ablation.json'],
           ['统计单位为固定划分上的训练seed，n=5', '差值=对照−HOV',
            '精确Wilcoxon五个非零配对最小双侧p=0.0625', '配对t假定差值近似正态，结果仅供探索'])
    source('protocol', '数据和冻结训练协议', 'study_plan.json',
           '三条序列沿用各自连续时间6:2:2边界，同角色合并；仅训练集拟合，验证集选型，冻结后测试。',
           ['study_plan.json', 'frozen_configs.json', 'data/dataset_manifest.json'],
           ['train/val/test原始帧数9045/3015/3015；有效窗口8988/2958/2958',
            '平面视觉骨架，固定1.25px/mm尺度', '三序列为参考历史表现后选定的探索性子集'])
    source('diagnostics', '保存预测数组的空间与误差分布', 'evaluations/hov/seed_0/seq_20260819_172644_predictions.npz',
           '从已保存的55次测试预测中计算节点分布和误差累计分布；未改动训练或评价。',
           ['evaluations/*/seed_*/*_predictions.npz'],
           ['空间曲线按对应节点编号1至15汇总', 'CDF分母为同一批测试帧上的五次预测，只作分布描述'])
    out = Report('5 Hz三序列建模实验结果', sources)
    out.md('title', '# ' + out.title)
    out.md('summary', f"""## 结果概览

**HOV的两个记忆分支都有稳定贡献，整体精度介于Yu与其余主对照之间。** HOV测试节点误差为 **{value('hov','mean_node_mm'):.3f} ± {sd('hov','mean_node_mm'):.3f} mm**，Yu Bézier–GRU为 **{value('bezier_gru','mean_node_mm'):.3f} ± {sd('bezier_gru','mean_node_mm'):.3f} mm**。完整HOV在五个seed上均优于三个消融。

**模型规模是明确优势。** HOV共760个存储系数，约为Yu的1.43%。下面按主对照、运行波动、消融、运动序列、误差分布、收敛和计算成本展开，图旁给出解释，详细表保留精确数值。""", 'results')
    out.md('scope', """## 先看数据范围与读图口径

本轮共 **11种模型配置 × 5个seed = 55次训练与测试**。使用172644（四通道运动）、181044（下段运动）、181548（上段运动）三条序列，5 Hz采样；每条序列按时间连续6:2:2划分，再合并相同角色数据。每次测试评分 **2,958帧**，输入历史20步，梯度训练预算100 epoch。

主表先按帧合并三序列，再对五个seed等权平均；“±”表示训练随机性的样本标准差。节点、末端单位为mm，越低越好；IoU、Dice为0–1，越高越好。毫米尺度来自二维视觉骨架的固定标定。Chen、Yu、Krauss均为本任务的建模适配，Yu使用Bézier–GRU表示组合。

三条序列是参考历史结果后确定的探索性子集；下列结论覆盖此固定划分与训练预算。""", 'protocol')
    partition = [dict(sequence=GROUP[r['group']], role={'train':'训练','val':'验证','test':'测试'}[r['role']],
                      frames=r['frames'], usable_windows=r['frames'] - 19)
                 for r in manifest['files']]
    out.md('split_intro', '三条序列在每个角色内共同参与训练或评价。172644最长，在按帧合并的总指标中占较高权重；后面的分序列图用于观察运动模式差异。', 'protocol')
    out.chart('partition', '各序列的训练、验证和测试帧数', partition, kind='horizontalStackedBar',
              x='sequence', y='frames', color='role', source='protocol', unit='帧',
              description='原始采样帧数；每个角色内各序列前19帧仅用于窗口历史')

    def main_data(metric, names):
        return [dict(model_label=LABEL[m], model=m, value=value(m, metric), seed_std=sd(m, metric),
                     n_seeds=5, frames_per_seed=2958, node_mm=value(m, 'mean_node_mm'),
                     endpoint_mm=value(m, 'endpoint_mm'), iou=value(m, 'mask_iou'),
                     parameters=numeric(m, 'parameter_count_including_fitted_reference_mean')) for m in names]

    out.md('main', '## 主对照：Yu精度最高，HOV优于其余主对照\n\n以下五张图分别展示骨架、末端与整体mask表现。柱长表示五次平均；运行波动见后面的散点与均值±标准差表。', 'results')
    intros = {
        'mean_node_mm': '按节点误差从低到高排序，HOV均值为1.474 mm，Chen为1.553 mm，Yu为1.286 mm。主对照覆盖动态建模、静态回归与几何基线。',
        'node_rmse_mm': '节点RMSE先平均所有对应节点的欧氏误差平方，再开方，对较大误差更敏感。HOV为1.900 mm，Yu为1.632 mm；与平均误差图结合判断长尾误差。',
        'endpoint_mm': '末端单独评分，Yu为2.066 mm，HOV为2.631 mm。整体平均误差较小不保证末端同样准确，因此保留独立末端图。',
        'mask_iou': 'IoU衡量整个预测投影区域与mask的重叠。HOV与Chen、Krauss、Koopman的差距小于骨架误差的差距，需要结合逐seed波动判断。',
        'mask_dice': 'Dice提供另一种区域重叠指标；其方向与IoU一致。两者来自同一组mask，不能当作相互独立的证据。',
    }
    for metric, (label, unit) in METRICS.items():
        out.md('intro_' + metric, intros[metric], 'results')
        order = sorted(main, key=lambda m: value(m, metric), reverse=metric.startswith('mask_'))
        out.chart('main_' + metric, '主对照 · ' + label, main_data(metric, order), unit=unit,
                  description='五次训练的均值；每次合并三序列全部测试帧')
    out.md('repeat', '## 五个seed的散点展示运行波动\n\n每个点代表一次完整训练的测试均值，横轴仅为预定seed编号。HOV的节点误差标准差约0.006 mm；五个点都保留，可以直接看出模型之间的差距是否大于训练波动。', 'results')
    points = [dict(seed=int(r['seed']), model_label=LABEL[r['model']], value=float(r['mean_node_mm']),
                   endpoint_mm=float(r['endpoint_mm']), mask_iou=float(r['mask_iou']),
                   point_label=f"{LABEL[r['model']]} / seed {r['seed']}") for r in seed_rows if r['model'] in main]
    out.chart('seed_scatter', '主对照 · 每次训练的节点误差', points, kind='scatter', x='seed', color='model_label',
              description='40个点：8种模型 × 5个seed；横轴不是时间')
    columns = [('order','序号','number'), ('model_label','模型','text')] + [(m, label, 'text') for m, (label, _) in METRICS.items()]
    rows = [dict(order=i+1, model_label=LABEL[m], **{k:f'{value(m,k):.4f} ± {sd(m,k):.4f}' for k in METRICS}) for i,m in enumerate(main)]
    out.md('exact_main', '下表提供精确的均值和样本标准差，节点与末端单位为mm。主对照排序沿用实验计划。', 'results')
    out.table('main_table', '主对照精确数值', rows, columns)

    out.md('ablation', '## 消融：两个记忆分支都有贡献\n\n相对无play、无Maxwell和无记忆模型，完整HOV的节点误差分别降低 **6.2%、5.1%、23.5%**。四种配置共享参考几何与训练设置，仅移除指定的记忆分支。', 'results')
    out.chart('ablation_node', '消融 · 节点平均误差', main_data('mean_node_mm', abl), description='五次均值；越低越好')
    out.md('ablation_tip_intro', '末端对记忆建模同样敏感：完整模型约2.631 mm，三个消融分别约2.947、2.966和4.378 mm。', 'results')
    out.chart('ablation_tip', '消融 · 末端误差', main_data('endpoint_mm', abl), description='五次均值；第15节点欧氏距离')
    for metric in ('mask_iou','mask_dice'):
        out.md(metric+'_ablation_intro', f"消融的{METRICS[metric][0]}图展示整体区域重叠的变化；这与骨架、末端指标互为补充，仍来自相同视觉标签。", 'results')
        out.chart('ablation_'+metric, '消融 · '+METRICS[metric][0], main_data(metric,abl),unit='',description='五次均值；0–1，越高越好')
    diff = [dict(seed=seed, variant=LABEL[m], value=float(paired[m,seed]['mean_node_mm']) - float(paired['hov',seed]['mean_node_mm']),
                 full_mm=float(paired['hov',seed]['mean_node_mm']), ablated_mm=float(paired[m,seed]['mean_node_mm']),
                 point_label=f'{LABEL[m]} / seed {seed}') for m in abl if m!='hov' for seed in range(5)]
    out.md('ablation_pairs_intro', '每个点是同一个seed下“消融−完整HOV”的误差差值。15个差值都在零线上方，说明三个消融的退化出现在全部五次训练中；统计检验的解释边界见后文。', 'statistics')
    out.chart('ablation_pairs', '消融 · 与完整HOV的配对差值', diff, kind='scatter', x='seed', color='variant',
              source='statistics', zero=True, description='正值表示完整HOV误差更低；每个点是一个训练seed配对')
    out.md('ablation_lookup', '下表保留四种配置全部指标的均值与样本标准差，便于读取小幅差异。', 'results')
    out.table('ablation_table','消融精确数值',
              [dict(order=i+1,model_label=LABEL[m],**{k:f'{value(m,k):.4f} ± {sd(m,k):.4f}' for k in METRICS}) for i,m in enumerate(abl)], columns)

    out.md('sequences', '## 分序列：下段运动中的排序不同\n\n在181044下段运动中，HOV节点误差约1.350 mm，Yu约1.520 mm；在四通道与上段运动中，Yu更低。热图展示全部主对照，帮助判断总体均值对应哪些运动模式。', 'sequences')
    seq_data = []
    for group in GROUP:
        for m in main:
            selected = [r for r in sequence_rows if r['model']==m and r['group']==group]
            assert len(selected)==5
            seq_data.append(dict(model_label=LABEL[m], group=GROUP[group], model=m,
                                 frames_per_seed=int(selected[0]['frames']),
                                 **{metric:float(np.mean([float(r[metric]) for r in selected])) for metric in METRICS}))
    out.chart('sequence_heatmap', '分序列 · 节点误差热图', seq_data, kind='heatmap', x='group', y='model_label',
              color='mean_node_mm', heat=True, source='sequences', description='8种模型 × 3条序列；每格为五seed均值/mm，颜色越深数值越高')
    out.md('sequence_endpoint_intro', '末端热图展示相同模型与序列下的末端误差。对比上图可以区分整体形态较准但末端仍有偏差的情况。', 'sequences')
    out.chart('sequence_endpoint_heatmap', '分序列 · 末端误差热图', seq_data, kind='heatmap', x='group', y='model_label',
              color='endpoint_mm', heat=True, source='sequences', description='每格为五seed末端误差均值/mm')
    out.md('sequence_group_intro', '将HOV和三种近期动态适配方法并排展示，可以更直观看到三个运动模式中的排序变化。完整八模型比较见上方热图。', 'sequences')
    out.chart('sequence_grouped', '运动模式与近期动态方法', [r for r in seq_data if r['model'] in main[:4]], kind='bar',
              x='group', y='mean_node_mm', color='model_label', source='sequences', description='每条序列分别汇总；五seed均值')

    from scripts.experiments.modeling_report_diagnostics import compute_diagnostics
    diagnostics = compute_diagnostics(study)
    profiles = [dict(row, model_label=LABEL[row['model']]) for row in diagnostics['node_profiles']]
    cdf = [dict(row, model_label=LABEL[row['model']]) for row in diagnostics['frame_error_cdf']]
    out.md('space', '## 沿机器人长度查看误差在哪里累积\n\n节点1为基座、节点15为末端。曲线按对应节点汇总五次预测，显示局部形状误差如何沿身体分布；点间连线表示身体顺序。', 'diagnostics')
    out.chart('body_profile', '各节点的平均位置误差', profiles, kind='line', x='node', y='mean_mm',
              color='model_label', source='diagnostics', description='15个对应节点；三序列按帧合并后对五seed取均值')
    out.md('distribution', '## 累计分布补充平均误差\n\n横轴为单帧节点平均误差阈值，纵轴为误差不超过该阈值的比例。相同阈值下曲线越高，达到该精度的预测越多。此处合并五次对同一批测试帧的预测，仅描述分布。', 'diagnostics')
    out.chart('error_cdf', '逐帧节点误差的累计分布', cdf, kind='line', x='error_mm', y='fraction',
              color='model_label', source='diagnostics', unit='', description='横轴mm；纵轴0–1；每模型14,790次预测，来自2,958帧 × 5seed')

    out.md('convergence', '## 训练曲线：早期误差与后续波动一起看\n\n以下展示每个梯度模型的全部五条原始验证曲线，每条21个记录点；没有用“截至当前最优值”替换实际波动。HOV先做参考拟合及记忆读出初始化，因此较低的初始误差同时包含这些训练成本。各图自动纵轴范围不同，精确数值可悬停查看。', 'training')
    training_rows = []
    for m in models:
        if m=='linear':
            continue
        curves=[]
        for seed in range(5):
            history=read_json(study/'formal'/m/f'seed_{seed}'/'history.json')
            curves.extend(dict(model=LABEL[m], seed_label=f'seed {seed}', epoch=h['epoch'],
                               value=h['validation_node_mean_mm'], best_value=h['best_validation_node_mean_mm'],
                               elapsed_seconds=h['elapsed_seconds'], learning_rate=h['lr']) for h in history)
        improving = sum('still_improving' in paired[m,s]['convergence_flags'] for s in range(5))
        out.md('curve_intro_'+m, f"**{LABEL[m]}**：五次最优验证节点误差平均{numeric(m,'validation_node_mean_mm_mean'):.3f} mm；末段仍改善的运行有{improving}/5次。曲线用于观察本轮100 epoch预算内的优化情况。", 'training')
        out.chart('curve_'+m, LABEL[m]+' · 原始验证曲线', curves, kind='line', x='epoch', color='seed_label',
                  source='training', description='每条线一个seed；纵轴为当次验证节点均距/mm')
        training_rows.extend(curves)
    out.md('closed_form', '线性回归以闭式求解完成，每个seed只有一次拟合评价，因此在成本表中记录，不绘制100 epoch曲线。', 'training')

    resources = [dict(model_label=LABEL[m], model=m, parameters=numeric(m,'parameter_count_including_fitted_reference_mean'),
                      trainable_parameters=numeric(m,'trainable_parameter_count_mean'),
                      log10_parameters=math.log10(numeric(m,'parameter_count_including_fitted_reference_mean')),
                      node_mm=value(m,'mean_node_mm'), train_seconds=numeric(m,'wall_seconds_mean'),
                      b1_ms=numeric(m,'latency_b1_p50_ms_mean'), b1_p95_ms=numeric(m,'latency_b1_p95_ms_mean'),
                      b256_ms=numeric(m,'latency_b256_p50_ms_mean'), point_label=LABEL[m]) for m in main]
    out.md('cost', '## 模型规模与运行时间分别评价\n\nHOV的760个系数包含训练拟合的参考系数，Yu为53,004个。散点图用log10参数量轴展开数量级差异：越靠左规模越小，越靠下节点误差越低；它展示的是本轮实现的取舍。', 'results')
    out.chart('size_accuracy', '参数量与测试精度', resources, kind='scatter', x='log10_parameters', y='node_mm',
              color='model_label', description='8种主对照；横轴log10(系数数量)，原始数量保留在图表数据中')
    out.md('parameter_intro', '参数柱图显示原始系数数量。不同模型的归纳结构、固定几何和计算路径不同，参数量不直接等于运行时间。', 'training')
    out.chart('parameter_count', '主对照 · 存储系数数量', sorted(resources,key=lambda r:r['parameters']), y='parameters', unit='个', source='training')
    out.md('training_cost_intro', 'HOV每次训练平均约119秒，Yu约47秒，Krauss振子约154秒。以下为包含参考拟合、训练与验证的墙钟时间；GPU2、3每卡两个任务并发，时间受该调度影响。', 'training')
    out.chart('training_cost', '主对照 · 训练墙钟时间', sorted(resources,key=lambda r:r['train_seconds']), y='train_seconds', unit='秒', source='training')
    out.md('latency_intro', '完整HOV的单窗口推理中位耗时约4.76 ms，Yu约1.54 ms，Krauss约8.11 ms。下面使用相同物理GPU2、相同H20输入预算，全部模型串行计时；这些值是完整窗口成本。', 'latency')
    out.chart('latency_single', '主对照 · 单窗口推理耗时', sorted(resources,key=lambda r:r['b1_ms']), y='b1_ms', unit='ms', source='latency', description='B1、seed0；20次预热，100次同步计时的p50')
    out.md('latency_batch_intro', '批量图计量一次处理256个窗口的总耗时。与单窗口图一起看，可区分Python调度、算子调用和矩阵并行带来的开销；整批延迟没有除以256。', 'latency')
    out.chart('latency_batch', '主对照 · 256窗口整批推理耗时', sorted(resources,key=lambda r:r['b256_ms']), y='b256_ms', unit='ms', source='latency', description='B256整批p50；输入驻留GPU，含毫米输出转换')
    out.md('resource_lookup', '下表补充全部11种配置的计算成本，包括三个消融。存储系数总量包括消融中停用的参数，实际梯度优化参数单独列出。消融分支停用并不一定减少当前实现的窗口计算开销。')
    out.table('resource_table','全部模型的参数与运行成本',
              [dict(order=i+1,model=LABEL[m],parameters=int(numeric(m,'parameter_count_including_fitted_reference_mean')),
                    trainable=int(numeric(m,'trainable_parameter_count_mean')),
                    train=f"{numeric(m,'wall_seconds_mean'):.2f} ± {numeric(m,'wall_seconds_std'):.2f}",
                    single=f"{numeric(m,'latency_b1_p50_ms_mean'):.3f}",batch=f"{numeric(m,'latency_b256_p50_ms_mean'):.3f}") for i,m in enumerate(models)],
              [('order','序号','number'),('model','模型','text'),('parameters','存储系数','number'),('trainable','可训练参数','number'),
               ('train','训练/秒','text'),('single','B1 p50/ms','text'),('batch','B256 p50/ms','text')], source='training')

    out.md('statistics', '## 显著性：同时看效应量与检验分辨率\n\n主检验采用五个训练seed的双侧精确Wilcoxon，最小可能原始p值为0.0625，本轮均未达到0.05。探索性配对t检验假设差值近似正态，另做Holm校正；三个消融的节点误差检验通过该探索性校正。统计仅描述此固定划分上的训练随机性。', 'statistics')
    stats_rows=[]
    for family,data in [('主对照',comparison),('消融',ablation)]:
        for c in data['paired_tests']['comparisons']:
            if c['metric']!='mean_node_mm':
                continue
            stats_rows.append(dict(order=len(stats_rows)+1, family=family, model=LABEL[c['model']],
                                   difference=f"{c['mean_difference']:+.4f}",
                                   exact_p=f"{c['wilcoxon']['p_value']:.3g}", exact_holm=f"{c['wilcoxon']['p_holm']:.3g}",
                                   t_holm=f"{c['t_test']['p_holm']:.3g}",
                                   all_pairs='5/5' if all(v>0 for v in c['differences']) else str(sum(v>0 for v in c['differences']))+'/5'))
    out.md('stats_lookup', '差值为“对照−HOV”，正值表示HOV误差更低。表中展示节点主指标；其余指标与完整配对值在相应统计来源中保存。', 'statistics')
    out.table('statistics_table','节点误差的配对统计',stats_rows,
              [('order','序号','number'),('family','比较族','text'),('model','对照','text'),('difference','差值/mm','text'),
               ('exact_p','精确p','text'),('exact_holm','精确Holm p','text'),('t_holm','探索t Holm p','text'),('all_pairs','HOV较低次数','text')],source='statistics')
    out.md('next', """## 这些图支持的结论与下一步问题

本轮支持：在该三序列固定划分中，两个记忆分支对预测都有贡献；HOV以较少系数取得了优于多数适配对照的精度，Yu整体精度更高。下段运动的排序变化提示，应把运动模式与加载路径分开研究。

后续可围绕三个问题设计新实验：用相同当前输入、不同加载历史检验play记忆；用持压与不同变化速率检验Maxwell松弛；用新增采集日和完整运动覆盖检验泛化。若目标是在线速度，还应为所有动态模型建立统一的缓存状态单步协议。当前完整窗口测速可作为部署预算参考。

本轮100 epoch预算内部分对照仍有改善；中心线与mask指标同源，三序列也共享采集条件。已有测试结果用于本报告，后续模型开发需继续通过独立验证与新的实验设计推进。""")
    output.mkdir(parents=True, exist_ok=True)
    project_snapshot(out, output)
    artifact = dict(surface='report', manifest=dict(version=1,surface='report',title=out.title,
                    description='三序列固定6:2:2划分，五次重复，主对照、消融、逐序列与计算成本',
                    generatedAt=stamp,blocks=out.blocks,charts=out.charts,tables=out.tables,cards=[],filters=[],sources=sources),
                    snapshot=dict(version=1,generatedAt=stamp,status='ready',datasets=out.datasets),sources=sources)
    write_json(output/'artifact.json',artifact)
    write_json(output/'chart_map.json',out.chart_map)
    write_json(output/'diagnostics.json',diagnostics)
    write_json(output/'build_notes.json',dict(source_study=prefix, audience='technical', delivery='portable_html',
        canonical_reader='same packaged reader as first modeling report', charts=len(out.charts),
        sections=['summary','scope','comparison','seeds','ablation','sequences','profiles','cdf','training','cost','statistics','next'],
        quantitative_scope='55 completed runs, fixed five seeds; per-frame pooling followed by seed mean/sampleSD',
        palette_policy='shared approved categorical palettes for genuine groups; sequential blue for magnitudes',
        curve_policy='all ten gradient-model families, all five raw validation traces, 21 observations per trace',
        omitted={'geometry_images':'This report visualizes scored errors; coordinate traces are retained in diagnostics for later inspection.',
                 'linear_epoch_curve':'Closed-form fitting has one recorded evaluation.'}))
    (output/'README.md').write_text(f"# {out.title}\n\n入口：[HTML报告](report.html)。共{len(out.charts)}张交互图和{len(out.tables)}张明细表，数据内嵌，可离线打开。\n\n"
        f"数据来源：`{prefix}/results/`。图表定义与数据：`artifact.json`；图表清单：`chart_map.json`；生成记录：`build_notes.json`。\n\n"
        "生成入口：`scripts/experiments/build_modeling_improvement_html.py`，随后使用Data Analytics的`report:deliver`命令打包。\n",encoding='utf-8')
    print(f'Prepared {len(out.charts)} charts, {len(out.tables)} tables, {len(out.blocks)} blocks: {output / "artifact.json"}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    build(args.study.resolve(),args.output.resolve())
