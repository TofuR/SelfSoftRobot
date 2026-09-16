#!/usr/bin/env python3
"""Assemble the canonical artifact for the historical-response research report."""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import sqlite3
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.experiments.build_modeling_improvement_html import Report

TITLE='全身自模型的路径记忆与时间记忆分析'


def load(path):
    return json.loads(path.read_text())


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def source(id,label,path,script,definitions,stamp):
    return dict(id=id,label=label,path=path,query=dict(engine='Python / NumPy / PyTorch',
        language='python',description='由已有采集数据、固定模型与本次独立分析计算；完整方法和中间结果保存在来源文件。',
        tables_used=[path,script],metric_definitions=definitions,
        executed_at=stamp,filters=['已有数据的探索性机制分析','保留全部预定重复与敏感性结果']))


def add_chart(report,c,source_id,prefix=''):
    rows=c.get('rows',c.get('data',[]))
    if not rows:
        return
    x,y=c['x'],c['y']
    color=c.get('color')
    # Renderer tables retain all scalar audit dimensions, not just encodings.
    clean=[]
    for index,row in enumerate(rows):
        if row.get(x) is None or row.get(y) is None:
            continue
        item={key: (json.dumps(value,ensure_ascii=False) if isinstance(value,(dict,list)) else value)
              for key,value in row.items()}
        if isinstance(item[y],(float,int)) and not math.isfinite(item[y]):
            continue
        item.setdefault('point_label',str(row.get('label',row.get('group',index))))
        clean.append(item)
    if not clean:
        return
    name=prefix+c['id']
    kind=c.get('kind','bar')
    report.chart(name,c['title'],clean,kind=kind,x=x,y=y,color=color,
        unit=c.get('unit',''),source=source_id,description=c.get('description',c.get('note','')),
        question=c.get('question',c['title']),heat=kind=='heatmap')
    spec=report.charts[-1]
    labels={'node':'节点编号（基座至末端）','tolerance':'压力匹配容差（kPa）',
        'tolerance_kpa':'压力匹配容差（kPa）','pairs':'合格帧对数',
        'model_label':'模型','delta_error_mm':'形态差异预测误差（mm）',
        'lateral_delta_mm':'两帧横向坐标差（mm）','variant_label':'干预方式',
        'mean_node_mm':'节点平均误差（mm）','rms_bend_deg':'局部弯曲贡献 RMS（°）',
        'history_steps':'历史窗口步数','pressure_kpa':'压力指令（kPa）',
        'probe_x_mm':'NDI 探头 x 坐标（mm）','gap_3d_mm':'同压两支三维间距（mm）',
        't_sec':'采集时间（s）','probe_x_relative_mm':'NDI 探头相对 x 坐标（mm）',
        'cycle_id':'完整周期编号','pressure_mean_gap_mm':'压力区间平均间距（mm）',
        'mean_abs_command_slope_kpa_s':'实际平均扫压速度（kPa/s）',
        'gap_75kpa_mm':'75 kPa 两支间距（mm）','elapsed_s':'保持时长（s）',
        'expected_remaining_fraction':'时间记忆差值的剩余比例',
        'mean_node_time_correction_mm':'净时间修正的节点位移（mm）',
        'tau_s':'固定时间常数 τ（s）','state_rms_mean':'时间记忆差值 RMS',
        'epoch':'正式优化 epoch','val_node_mm':'验证集节点误差（mm）',
        'seed':'训练 seed','parameter_count':'模型参数量','mean_3d_mm':'NDI 三维位置误差（mm）'}
    spec['xAxisTitle']=c.get('x_label',labels.get(x,x.replace('_',' ')))
    spec['yAxisTitle']=c.get('y_label',labels.get(y,y.replace('_',' ')+(f'（{c["unit"]}）' if c.get('unit') else '')))
    if kind=='line':
        spec['settings']['showPoints']='auto'


def table(report,id,title,rows,columns,source_id):
    if not rows:
        return
    rows=[dict(order=i+1,**row) for i,row in enumerate(rows)]
    report.table(id,title,rows,[('order','序号','number')]+columns,source=source_id,sort='order')


def chart_data(module,id):
    return next(c for c in module['charts'] if c['id']==id)


def table_data(module,id):
    return next(t['rows'] for t in module['tables'] if t['id']==id)


def aggregate(rows,keys,measure):
    groups=defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(float(row[measure]))
    return [dict(zip(keys,key),**{measure:statistics.mean(vals),'sd':statistics.stdev(vals) if len(vals)>1 else 0.,'n':len(vals)})
            for key,vals in groups.items()]


def custom_chart(out,id,title,rows,x,y,src,kind='bar',color=None,**kwargs):
    add_chart(out,dict(id=id,title=title,rows=rows,x=x,y=y,kind=kind,color=color,**kwargs),src)


def project_snapshot(report,output):
    statements={}
    with sqlite3.connect(output/'chart_snapshot.sqlite') as db:
        for item in report.charts+report.tables:
            name,source_id=item['dataset'],item['sourceId']
            rows=report.datasets[name]
            columns=list(rows[0])
            assert all(set(r)==set(columns) for r in rows),name
            types={}
            for column in columns:
                values=[r[column] for r in rows if r[column] is not None]
                types[column]='REAL' if all(isinstance(v,(int,float)) for v in values) else 'TEXT'
            table_name='chart_'+name
            db.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            definition=','.join(f'"{k}" {types[k]}' for k in columns)
            db.execute(f'CREATE TABLE "{table_name}" ({definition})')
            db.executemany(f'INSERT INTO "{table_name}" VALUES ({",".join("?" for _ in columns)})',
                           [[r[k] for k in columns] for r in rows])
            sql=f'SELECT * FROM "{table_name}" ORDER BY rowid;'
            projected=[dict(zip(columns,row)) for row in db.execute(sql)]
            assert projected==rows,name
            report.datasets[name]=projected
            statements.setdefault(source_id,[]).append(sql)
    for src in report.sources:
        if src['id'] in statements:
            src['query'].update(language='sql',sql='\n'.join(statements[src['id']]),
                engine='Python analysis; executed SQLite display-data projection')
            src['query']['tables_used'].append((output.relative_to(ROOT)/'chart_snapshot.sqlite').as_posix())
    (output/'analysis.sql').write_text('\n\n'.join('-- '+k+'\n'+'\n'.join(v) for k,v in statements.items())+'\n')


def history_sections(out,h):
    summaries=h['history']['summary']
    metric=lambda category,model: next(r['delta_error_mm'] for r in summaries
        if r['tolerance_kpa']==5 and r['category']==category and r['model']==model)
    count=lambda category: next(r['pairs'] for r in summaries if r['tolerance_kpa']==5 and r['category']==category)
    out.md('history_story',f'''## 全身配对：方向不同有收益，同方向的更早历史仍需进一步验证

以各通道当前压力差不超过5 kPa为条件，在同一序列内寻找相隔至少20帧的输入窗口。历史差异由前19步压力的RMS距离定义，要求至少20 kPa；选帧规则不读取形状标签或模型误差。每个类别内按压力距离优先选择不复用目标帧的配对。

得到异方向 {count('opposite')} 对、同方向异历史 {count('same_direction')} 对，以及最近两步都近似相同的 {count('same_recent_two')} 对。这里的“形态差异预测误差”比较的是两帧**坐标差向量**：先相减，再与观测差相比较，最后对15个节点取平均，单位mm；不是两帧各自预测误差的相减。''','history')
    for c in h['charts']:
        if c['id']=='history_pair_coverage':add_chart(out,c,'history')
    out.md('history_opposite_note',f'''在异方向帧对中，HOV的差异预测误差为 {metric('opposite','hov'):.3f} mm，静态MLP为 {metric('opposite','mlp'):.3f} mm；Chen方向网络为 {metric('opposite','chen_direction'):.3f} mm，Yu Bézier–GRU适配为 {metric('opposite','bezier_gru'):.3f} mm。历史模型相对静态模型更能预测这类变化，但本结果不支持HOV在历史表示上全面领先这些对照。''','history')
    add_chart(out,next(c for c in h['charts'] if c['id']=='history_delta_opposite'),'history')
    out.md('history_same_note',f'''同方向异历史帧对中，HOV为 {metric('same_direction','hov'):.3f} mm，静态MLP为 {metric('same_direction','mlp'):.3f} mm；最近两步匹配组也未显示一致优势。当前数据对“更早历史的独立贡献”提供的证据有限。压力匹配残差、测量噪声与跨时段漂移都可能进入观测差异，不能把每个形态差异都解释为迟滞。

下图保留这些结果；较严格的1/2 kPa容差同时减少样本覆盖。不同阈值与类别之间会共享数据，图中的帧对数量不是独立试验次数。''','history')
    for id in ('history_delta_same_direction','history_delta_same_recent_two','history_tolerance_sensitivity'):
        add_chart(out,next(c for c in h['charts'] if c['id']==id),'history')
    out.md('history_example_note','### 沿臂分布的预测差异\n\n以下示例取相应类别中按序列及时间顺序出现的首个合格帧对，使用seed0。纵轴为两帧在各节点的横向坐标差，正负方向保留；示例的选取不依据预测效果。','history')
    for c in h['charts']:
        if c['id'].startswith('history_example'):add_chart(out,c,'history')
    out.md('frozen_story','''## 固定模型的干预说明两个分支确实参与预测

重训消融允许剩余参数重新适应。这里固定完整HOV的全部权重和参考映射，分别把路径项、时间项置零，或在同一序列内置换该项与目标帧的对应关系。置换保留贡献项的边际分布，但打乱其与实际历史的关联。

误差上升说明所学预测实际依赖相应分支。该操作作用于模型内部读出，属于计算干预；它不能单独证明状态与某个材料内部变量具有唯一对应关系。''','history')
    for id in ('frozen_branch_interventions','branch_geometry_profile'):
        add_chart(out,next(c for c in h['charts'] if c['id']==id),'history')
    values=h['history_horizon']['summary']
    out.md('horizon_note',f'''### 可用历史的长度也会影响预测

保持同样的H20训练权重和2958个目标帧，截短输入窗口。两步历史的节点误差为 {values[0]['mean_node_mm']:.3f} mm，20步为 {values[-1]['mean_node_mm']:.3f} mm，绝对改善 {values[0]['mean_node_mm']-values[-1]['mean_node_mm']:.3f} mm。局部历史已解释较多响应，更长历史带来进一步改善。

截短同时改变可见历史与状态初始化位置，因此该实验衡量固定模型的窗口敏感性；最优历史长度仍需重新训练或专门配对激励验证。''','history')
    add_chart(out,next(c for c in h['charts'] if c['id']=='history_horizon'),'history')


def finish(out,output,stamp):
    project_snapshot(out,output)
    artifact=dict(surface='report',manifest=dict(version=1,surface='report',title=out.title,
        description='周期迟滞、历史匹配、时间尺度、分支干预、插件迁移与训练收敛的已有数据分析',
        generatedAt=stamp,blocks=out.blocks,charts=out.charts,tables=out.tables,cards=[],filters=[],sources=out.sources),
        snapshot=dict(version=1,generatedAt=stamp,status='ready',datasets=out.datasets),sources=out.sources)
    save(output/'artifact.json',artifact)
    save(output/'chart_map.json',out.chart_map)
    return artifact


def sweep_sections(out,s):
    out.md('sweep_story','''## 周期扫压首先确认了需要历史信息的现象

早期两条记录使用 c0 通道的0→150→0 kPa三角扫压，步进15 kPa，其余通道保持零指令。172916记录501帧、含24个完整周期；173114记录226帧、含10个完整周期。实际平均扫压速度分别为73.85和30.00 kPa/s。

这里的独立测量是 **NDI探头三维位置**，探头位置与机器人末端的对应关系尚未确认。回线说明驱动指令与局部位置之间存在历史相关响应，不能直接作为全身骨架误差。以下曲线在各完整周期内按相同压力匹配，再跨周期平均；1 kPa绘图插值不增加实测样本，原始75 kPa点直接存在。150 kPa两支共用峰值，间距按定义为零；0 kPa比较周期首尾，可包含漂移。''','sweep')
    loop_rows=[]
    gap_rows=[]
    for seq in ('172916','173114'):
        rows=aggregate(chart_data(s,'loops_x_'+seq)['rows'],['pressure_kpa','branch'],'probe_x_mm')
        for r in rows:r['series']=seq+' / '+{'loading':'加载','unloading':'卸载'}[r['branch']]
        loop_rows+=rows
        rows=aggregate(chart_data(s,'gap_pressure_'+seq)['rows'],['pressure_kpa'],'gap_3d_mm')
        for r in rows:r['sequence']=seq
        gap_rows+=rows
    custom_chart(out,'sweep_mean_loops','加载与卸载沿不同位置分支变化',loop_rows,'pressure_kpa','probe_x_mm','sweep','line','series',unit='mm',description='每条线为同一记录完整周期的均值；周期标准差保存在图表数据中。')
    custom_chart(out,'sweep_mean_gap','相同压力下的三维位置间距随压力变化',gap_rows,'pressure_kpa','gap_3d_mm','sweep','line','sequence',unit='mm')
    summary=table_data(s,'sequence_summary')
    table(out,'sweep_summary','周期迟滞的测量量',[
        dict(sequence=r['sequence'][-6:],cycles=r['complete_cycles'],speed=r['mean_command_speed_kpa_s'],
             gap=r['gap_75kpa_mm_mean'],sd=r['gap_75kpa_mm_std'],mean_gap=r['pressure_mean_gap_mm_mean']) for r in summary],
        [('sequence','序列','text'),('cycles','完整周期数','number'),('speed','扫压速度 kPa/s','number'),
         ('gap','75 kPa间距 mm','number'),('sd','周期间SD mm','number'),('mean_gap','全压区间平均间距 mm','number')],'sweep')
    out.md('sweep_repeat_note','''75 kPa处的加载—卸载间距为 **2.369±0.377 mm** 和 **1.665±0.209 mm**。同一分支跨周期位置离散较小：快速记录加载/卸载分别为0.345/0.357 mm，慢速为0.162/0.247 mm。方向分支差异因此具有可重复的结构，超过该测量下的同分支周期间变动。

下面每点代表一个完整周期，保留连续采集中的波动。两种速度各只有一次连续记录，速度与记录条件相互混杂，不能据此把差异唯一归因于时间效应；这些周期也不是独立重新装配、重置后的重复实验。''','sweep')
    add_chart(out,chart_data(s,'speed_gap'),'sweep','sweep_')
    for seq in ('172916','173114'):
        add_chart(out,chart_data(s,'cycle_gap_'+seq),'sweep','sweep_')
    out.md('sweep_fit_story','''### 历史特征能否拟合留出的迟滞响应

以完整周期按时间顺序划分训练、验证、测试：快速记录14/5/5个周期，慢速6/2/2个周期；测试共7周期、140个不重复帧。每条记录只使用首个训练零压位置去除坐标偏移。各预测器在相同验证岭系数网格内选型，再冻结配置评估测试周期。

这是同类记忆方程的**单输入NDI位置拟合**：比较静态三次基函数、加入方向、路径记忆、时间记忆及双记忆的读出。方向编码借鉴Chen的思路；它与四通道全身HOV、Chen原文网络的任务及输出不同。五类模型每坐标的特征数为4/8/19/19/34，因此该小型验证也包含容量变化。

合并测试误差从静态的1.264 mm降至双记忆的0.406 mm，改善67.9%；方向编码为0.540 mm。时间记忆单独达到0.415 mm，双记忆只再改善2.24%，慢速记录上时间记忆略优。因此，这批扫压数据能证明历史特征的拟合作用，对路径分支的额外贡献仍需更丰富的反转轨迹。''','sweep')
    c=chart_data(s,'test_error')
    rows=[dict(model=r['label'],sequence={'pooled':'合并','seq_20260627_172916':'快速172916','seq_20260627_173114':'慢速173114'}[r['sequence']],
               mean_3d_mm=r['mean_3d_mm'],frames=r['frames'],cycles=r['cycles']) for r in c['rows']]
    custom_chart(out,'sweep_fit_error','留出周期：历史特征改善位置预测',rows,'model','mean_3d_mm','sweep',color='sequence',unit='mm',x_label='NDI预测器')
    out.md('sweep_trace_note','以下分别展示两条记录时间顺序上的首个测试周期。曲线保留符号和时间顺序，用于检查转折附近的误差；示例选择与拟合效果无关。','sweep')
    names={'Observed':'NDI观测','static_cubic':'静态三次','direction_cubic':'方向特征','path':'路径记忆','time':'时间记忆','dual':'双记忆'}
    for seq in ('172916','173114'):
        c=dict(chart_data(s,'held_trace_'+seq))
        c['rows']=[dict(t_sec=r['t_sec'],probe_x_relative_mm=r['probe_x_relative_mm'],model=names.get(r['model'],r['model'])) for r in c['rows']]
        add_chart(out,c,'sweep','sweep_')
    out.md('sweep_loop_fit_note','''平均位置误差之外，再检查预测能否恢复同压两支的间距。下图先分别计算预测与实测间距曲线在压力区间内的平均值，再对二者的绝对差按测试周期平均；它衡量平均回线宽度的拟合，曲线局部偏差仍可能相互抵消。完整数值还包括有向NDI x—压力回线面积，单位mm·kPa，它是几何面积，不是耗散能。现有结果评价前向拟合；真实迟滞补偿还需要模型改变指令后的实物执行实验。''','sweep')
    c=dict(chart_data(s,'loop_gap_fit_error'))
    c.update(title='测试周期：平均回线宽度的拟合误差',x_label='NDI预测器',y_label='平均回线宽度绝对误差（mm）')
    c['rows']=[dict(model=names.get(r['model'],r['model']),sequence=r['sequence'][-6:],mean_gap_absolute_error_mm=r['mean_gap_absolute_error_mm']) for r in c['rows']]
    add_chart(out,c,'sweep','sweep_')


def time_sections(out,t):
    out.md('time_story','''## 时间记忆：跨采样分析支持递推使用，持压机制还需要实测

从182253和182519原始处理来源恢复了4211帧名义10 Hz观测，包含旧合并目录遗漏的227帧。剔除初始历史不足及失效时序窗口后，各协议共同评价3937个目标帧。所用完整HOV和无时间分支模型均已在前三条5 Hz序列上训练并冻结，这两条记录用于跨记录探索性诊断。

原生窗口39点、抽样窗口20点，使用相同历史起止帧。名义步长0.1/0.2 s对应同样3.8 s；实际相机均频约9.05–9.08 Hz，窗口实际平均跨度约4.19 s。两种抽样相位一起覆盖全部共同目标帧，每帧只计一次。**抽样改变模型可获得的中间指令，不改变机器人真实运动速度。** 骨架为已有单帧SAM2标签，本轮抽查6张原图，尚未采用新的整批重建标签流程。

原生名义10 Hz的HOV节点误差为 **1.676±0.022 mm**，抽样5 Hz为 **1.704±0.020 mm**；原生10 Hz的无时间分支模型为 **1.759±0.025 mm**。细采样与时间分支在这两条记录中有小幅收益，这项证据同时包含历史信息完整性与跨记录条件。''','time')
    protocols={'nominal_10Hz_H39':'原生10 Hz / H39','decimated_5Hz_H20':'抽样5 Hz / H20',
        'event_issue_10Hz_H39':'发送时刻积分 / 原生','event_issue_5Hz_H20':'发送时刻积分 / 抽样',
        'event_ack_10Hz_H39':'确认时刻积分 / 原生','event_ack_5Hz_H20':'确认时刻积分 / 抽样',
        'wrong_dt_10Hz_H39_dt0.2':'误用0.2 s / 仅seed 0'}
    c=chart_data(t,'native_rate_prediction')
    def rate_rows(keep):
        return [dict(protocol=protocols[r['protocol']],model={'hov':'完整HOV','hov_no_maxwell':'无时间分支'}[r['model']],
                     mean_node_mm=r['node_mean_mm_mean'],sd_mm=r['node_mean_mm_sd']) for r in c['rows'] if r['protocol'] in keep]
    custom_chart(out,'time_rate','相同目标帧：原生与抽样历史的预测',rate_rows(['nominal_10Hz_H39','decimated_5Hz_H20']),
                 'protocol','mean_node_mm','time',color='model',unit='mm',x_label='输入采样协议')
    by_seq=chart_data(t,'native_by_sequence')['rows']
    custom_chart(out,'time_sequences','两条原生记录中的结果',[
        dict(sequence=r['sequence'][-6:],model=r['model'],mean_node_mm=r['node_mean_mm_mean'],sd_mm=r['node_mean_mm_sd'])
        for r in by_seq if r['protocol']=='nominal_10Hz_H39'],'sequence','mean_node_mm','time',color='model',unit='mm',x_label='采集序列')
    out.md('time_dt_note','''### 时间尺度必须由物理时间定义

按发送或确认日志时刻分段积分时，完整模型原生误差分别为1.672和1.671 mm，接近名义步长结果。两种日志语义均保留，未用测试误差选择时序配置。下图一并显示抽样与误用步长的敏感性；误用步长仅运行seed 0，其余协议为全部5 seed均值。

另将每段0.2 s恒定输入传播拆成两个0.1 s子步，保持相同输入及原有时间常数，最大节点差为3.48×10⁻⁵ mm。这验证了指数递推的数值半步一致性。它不包含新的真实观测，也不能替代跨真实驱动速度的验证。''','time')
    rows=[r for r in rate_rows(list(protocols)) if r['model']=='完整HOV']
    custom_chart(out,'time_integrators','完整HOV的时间积分敏感性',rows,'protocol','mean_node_mm','time','horizontalBar',unit='mm',x_label='时间积分方式')
    out.md('tau_note','''### 多个时间尺度在起作用，但还不能分别对应材料机制

模型使用6个固定时间常数，约0.600、0.763、0.971、1.236、1.572和2.000 s。学习的是它们与几何之间的读出关系，不能把这些预设τ解释为辨识所得的材料谱。

测试窗口中相邻τ的状态相关性平均为 **0.993**，说明各尺度高度重叠。在归一化形状坐标中，单独分量RMS之和约为合成RMS的 **14.34倍**，存在明显抵消。移除整个时间项导致的净节点修正约0.645 mm；单个分量较大不等于该尺度具有独立的预测增益。''','time')
    c=dict(chart_data(t,'tau_state_correlation'))
    c['rows']=[dict(tau_a=f"{r['tau_a_s']:.3f}",tau_b=f"{r['tau_b_s']:.3f}",pearson_r_mean=r['pearson_r_mean']) for r in c['rows']]
    c.update(x_label='τ（s）',y_label='τ（s）',unit='相关系数')
    add_chart(out,c,'time','time_')
    c=dict(chart_data(t,'tau_geometric_profile'))
    c['rows']=[dict(node=r['node'],tau_s=f"τ={r['tau_s']:.3f} s",removal_mean_displacement_mm_mean=r['removal_mean_displacement_mm_mean']) for r in c['rows']]
    c.update(y_label='移除单尺度后的节点位移（mm）')
    add_chart(out,c,'time','time_')
    out.md('hold_note','''### 现有持压数据的覆盖与可做的近似分析

检查13条原始记录，以全通道压力范围与持续时间检测保持片段；0.001、0.5和2 kPa阈值下均没有足以验证本问题的≥1 s非零全通道持压段。183526只有39帧零压短段，缺少此前加载历史及对应骨架，无法判断时间记忆对持压形变的拟合能力。

因此，以下采用**固定模型的反事实推演**：从72个已有历史窗口末态出发，将当前输入保持不变，递推至8 s。各尺度差值满足指数衰减；带符号几何读出的合成幅值可能先增后减，但最终趋于零。它说明结构如何响应持压，实测准确性仍待非零持压数据验证。''','time')
    for id in ('counterfactual_decay','counterfactual_geometry'):
        c=dict(chart_data(t,id));c['title']='模型推演：'+c['title']
        if c.get('color')=='tau_s':c['rows']=[dict(r,tau_s=f"τ={r['tau_s']:.3f} s") for r in c['rows']]
        add_chart(out,c,'time','time_')


def plugin_sections(out,p):
    out.md('plugin_story','''## 记忆可以作为输入插件，改善线性模型和MLP

把相同递推方程产生的路径量q和时间差值h−e接到基础模型输入，直接预测15节点坐标。本实验迁移记忆结构；参考映射权重和HOV几何解码器没有参与插件拟合。四通道当前输入为4维，加入路径为12维，加入时间为28维，双记忆为36维。

使用与正式模型相同的三序列训练、验证、测试划分和5 Hz、H20窗口。线性模型在岭系数网格选型，MLP在学习率×宽度网格训练100 epoch，全部48个候选在验证集筛选、冻结后进行60次拟合。所有seed 0–4均保留，标准化只使用训练数据。

双记忆使线性模型节点误差从 **2.223降至1.824 mm（17.94%）**，使MLP从 **1.965降至1.480 mm（24.70%）**。两类记忆均可单独改善基础模型，组合达到这组插件变体中的最低误差。''','plugin')
    variants={'base':'当前输入','path':'+路径记忆','time':'+时间记忆','both':'+双记忆',
              'static_capacity':'静态扩展特征','window':'完整窗口'}
    rows=[dict(family={'linear':'线性','mlp':'MLP'}[r['family']],variant=variants[r['variant']],
               mean_node_mm=r['mean_node_mm_mean'],sd_mm=r['mean_node_mm_sd'],
               endpoint_mm=r['endpoint_mm_mean'],endpoint_sd_mm=r['endpoint_mm_sd'],
               parameter_count=r['parameter_count'],input_dim=r['input_dim'],config=json.dumps(r['config'],ensure_ascii=False))
          for r in table_data(p,'plugin_summary')]
    custom_chart(out,'plugin_node_mean','加入记忆后，两个基础模型的节点误差均下降',rows,'variant','mean_node_mm','plugin',color='family',unit='mm',x_label='输入表示')
    out.md('plugin_controls','''静态容量对照使用32个仅依赖当前输入的多项式/三角基，与双记忆保持相同输入维数。线性静态扩展模型为1.958 mm，比同为1665参数的双记忆线性模型差。MLP静态扩展模型为1.988 mm、27053参数；双记忆MLP为1.480 mm、9453参数。后者使用更少参数仍有收益，说明改善并非仅靠增加参数量。

完整20步窗口MLP为 **1.414 mm、12269参数**，比双记忆MLP更准确。因此，插件结果支持紧凑、递推的历史特征对基础模型有帮助；它不支持记忆结构在所有时序表示中具有最高精度。各MLP宽度由验证集决定，参数量并未强制完全相同。''','plugin')
    custom_chart(out,'plugin_capacity_mean','参数量与预测误差：保留静态和窗口对照',
        [dict(r,model=r['family']+' / '+r['variant'],point_label=r['family']+' / '+r['variant']) for r in rows],
        'parameter_count','mean_node_mm','plugin','scatter','family',unit='mm')
    out.md('plugin_endpoint_note','末端是节点15，下面使用同一批预测独立计算末端误差。插件提升全身节点指标的同时也改善末端位置；本次机制分析未重新计算mask指标。','plugin')
    custom_chart(out,'plugin_tip','末端位置随历史输入改善',rows,'variant','endpoint_mm','plugin',color='family',unit='mm',x_label='输入表示',y_label='末端平均误差（mm）')
    table(out,'plugin_metrics','全部插件与控制条件',rows,
          [('family','基础模型','text'),('variant','输入','text'),('input_dim','维数','number'),
           ('parameter_count','参数量','number'),('mean_node_mm','节点均值 mm','number'),('sd_mm','seed SD mm','number'),
           ('endpoint_mm','末端 mm','number'),('config','验证选择配置','text')],'plugin')
    seed_rows=table_data(p,'plugin_seeds')
    base={r['seed']:r['mean_node_mm'] for r in seed_rows if r['family']=='mlp' and r['variant']=='base'}
    improvements=[dict(seed=r['seed'],variant=variants[r['variant']],improvement_mm=base[r['seed']]-r['mean_node_mm'])
                  for r in seed_rows if r['family']=='mlp' and r['variant'] in ('path','time','both','window')]
    out.md('plugin_stats','''### 五次训练的变化方向一致，统计精度仍受重复数限制

双记忆MLP在全部5个seed均优于基础MLP。这里seed改变初始化和批次顺序，数据划分保持固定；统计单位是优化重复。配对双侧精确Wilcoxon检验p=0.0625，五个变体与基础模型的比较经Holm校正后p=0.3125。因此应报告一致改善和效应大小，尚不能称为该检验下p<0.05的显著提升。

线性拟合是确定性的，同一配置的5个seed标签产生相同预测，不计作5次独立证据，也不计算其显著性。''','plugin')
    custom_chart(out,'plugin_seed_gain','MLP各seed的节点误差改善',improvements,'seed','improvement_mm','plugin','scatter','variant',unit='mm',y_label='基础误差 − 变体误差（mm）')


def convergence_sections(out,p,h):
    out.md('convergence_story','''## 训练效率：首轮精度较高的来源是分阶段初始化

现有日志中的epoch从正式小批量优化开始计数。在此之前，参考映射已经完成500步坐标拟合和250步几何微调，随后用8988个训练窗口进行记忆读出岭回归初始化。因此，epoch 1不是从未训练模型出发的一次完整数据学习。

五次验证均值依次为：预拟合参考形态 **2.046 mm**，记忆初始化后的epoch 0 **1.631 mm**，epoch 1 **1.652 mm**，验证最佳 **1.509 mm**。epoch 1到最佳仍改善约8.53%，最佳epoch为95、95、85、75、95。已有证据更适合表述为“结构化初始化使模型在正式优化早期获得较低误差”，尚不能说“一epoch已完成训练”或“模型容量已饱和”。''','plugin')
    stages={'prefitted_reference':'预拟合参考形态','epoch0_after_memory_initialization':'记忆初始化 / epoch 0',
            'epoch1':'epoch 1','best_epoch':'验证最佳'}
    rows=aggregate([r for r in chart_data(p,'epoch0_stages')['rows'] if r['stage'] in stages],['stage'],'mean_node_mm')
    for r in rows:r['stage']=stages[r['stage']]
    custom_chart(out,'convergence_stages','完整HOV各训练阶段的验证误差',rows,'stage','mean_node_mm','plugin',unit='mm',x_label='训练阶段',y_label='验证节点平均误差（mm）')
    rows=aggregate([r for r in chart_data(p,'convergence_epoch')['rows'] if r['model'] in ('hov','mlp','chen_direction','bezier_gru')],
                   ['model_label','epoch'],'val_node_mm')
    custom_chart(out,'convergence_curves','正式优化阶段：各模型的验证收敛轨迹',rows,'epoch','val_node_mm','plugin','line','model_label',unit='mm',description='各点为全部5 seed均值；HOV此前已完成参考与记忆初始化。')
    originals=table_data(p,'original_convergence')
    hov=[r for r in originals if r['model']=='hov']
    init=statistics.mean(r['model_initialization_and_prior_seconds'] for r in hov)
    mem=statistics.mean(r['memory_initialization_seconds'] for r in hov)
    wall=statistics.mean(r['wall_seconds'] for r in hov)
    opt=statistics.mean(r['official_training_seconds'] for r in hov)
    out.md('convergence_cost',f'''### 完整成本与在线使用应分开评价

原训练日志中，HOV参考/模型初始化平均{init:.3f} s、记忆初始化{mem:.3f} s、正式100 epoch优化{opt:.3f} s，整次运行墙钟时间{wall:.3f} s。原正式训练在GPU上并发进行，本轮插件及epoch 0重建在CPU运行，这两类计时不能直接用于跨模型速度排名。

下面按原日志列出主要模型的平均时间分项；初始化、验证和保存等成本应计入训练效率讨论。较低的参数量不自动意味着更少计算，HOV逐步递推、几何输出与反向传播的成本也会影响耗时。''','plugin')
    rows=aggregate([r for r in chart_data(p,'training_time_components')['rows'] if r['model'] in ('hov','mlp','chen_direction','bezier_gru')],['model_label','phase'],'seconds')
    custom_chart(out,'convergence_costs','原训练日志中的时间构成',rows,'model_label','seconds','plugin',color='phase',unit='s',x_label='模型',y_label='平均耗时（s）')
    out.md('streaming_story','''缓存内部状态后，模型能够逐条输入更新形态。在本机CPU单线程、批量1、20次预热和200次计时下，完整重算20步窗口的中位耗时为 **3.271 ms**，缓存状态递推一步为 **0.806 ms**。这支持讨论模型用于在线预测的计算可行性。

计时只包含已在内存中的压力输入、状态处理和骨架输出，不含相机、分割、通信、规划或参数更新；同机分析任务可能影响调度。在线学习需要另测参数更新耗时、数据需求、漂移适应及遗忘，目前没有对应实测结论。''','history')
    table(out,'streaming_timing','同一实现的CPU预测耗时',h['streaming_timing']['rows'],
          [('mode','计算方式','text'),('p50_ms','p50 ms','number'),('p95_ms','p95 ms','number'),('samples','计时次数','number')],'history')


def build(output):
    output.mkdir(parents=True,exist_ok=True)
    modules={k:load(output/(f+'.json')) for k,f in {
        'history':'history_mechanisms','sweep':'sweep_hysteresis','time':'time_memory',
        'plugin':'plugin_convergence','literature':'literature'}.items()}
    stamp=datetime.now(timezone.utc).isoformat()
    filenames={'history':'history_mechanisms','sweep':'sweep_hysteresis','time':'time_memory',
               'plugin':'plugin_convergence','literature':'literature'}
    scripts={'history':'analyze_modeling_history_mechanisms.py','sweep':'analyze_modeling_sweep_hysteresis.py',
             'time':'analyze_modeling_time_memory.py','plugin':'analyze_modeling_plugin_convergence.py',
             'literature':'../../workspace/reports/paper_preparation_20260910_000/sources'}
    titles={'history':'历史匹配、固定分支干预与推理','sweep':'早期NDI周期扫压','time':'原生10 Hz与时间记忆',
            'plugin':'记忆插件及完整训练过程','literature':'已调研论文的原文核查'}
    sources=[]
    for k,m in modules.items():
        defs=m.get('definitions',m.get('common_definitions',[]))
        if isinstance(defs,dict):defs=[str(key)+': '+str(value) for key,value in defs.items()]
        defs=[str(v) for v in defs]
        script='scripts/experiments/'+scripts[k] if k!='literature' else 'workspace/reports/paper_preparation_20260910_000/sources'
        sources.append(source(k,titles[k],(output.relative_to(ROOT)/(filenames[k]+'.json')).as_posix(),script,defs,stamp))
    overview=source('overview','机制分析总览与数据范围',(output.relative_to(ROOT)/'README.md').as_posix(),
        'scripts/experiments/build_modeling_mechanisms_html.py',
        ['NDI位置、骨架节点与配对差异误差使用各自分母','所有seed与数据范围见各模块JSON'],stamp)
    overview['query']['tables_used'] += [s['path'] for s in sources]
    sources.append(overview)
    out=Report(TITLE,sources)
    out.md('opening','''# 全身自模型的路径记忆与时间记忆分析

研究问题：软臂的历史相关响应是否需要显式记忆，所提出的分支是否实际参与预测，以及这种表示能否复用于基础模型。分析基于已有采集数据、固定正式模型和本次新增插件拟合，整理日期为2026-09-13。

**已有证据支持“历史信息有助于建模、两分支被模型使用、记忆可复用”这一论证。** 早期扫压显示重复的加载—卸载差异；固定完整模型后移除或置换任一分支均降低预测精度；双记忆输入使普通MLP误差从1.965降至1.480 mm。模型对同方向更早历史的额外优势尚未出现，持压数据也不足以验证对应的物理解释。

报告按“观测现象→历史条件预测→分支功能→时间尺度→插件复用→训练效率”展开。机制名称统一为**时间记忆**；仅描述递推变量h时使用**时间记忆状态**。''')
    out.md('scope','''## 数据范围与指标口径

全身建模使用此前选定的172644、181044和181548三条序列，在每条序列内按时间6:2:2划分后合并训练、验证、测试；本轮沿用固定划分。它是针对已有选定数据的探索性分析，不代表对全部七条序列或未知机器人的总体评估。

骨架指标先在每帧15个对应节点上计算欧氏距离，再合并所有目标帧取均值，最后汇总5个seed的均值与样本标准差。不同模块的NDI位置误差、全身节点误差、配对差异误差各有不同分母，数值不能横向混为一个排名。所有记忆插件、干预与采样协议均使用完整预定结果，测试数据不参与新增拟合配置选择。''')
    data=[
        dict(data='20260627 / 172916、173114',measurement='727帧NDI探头位置，34完整周期',protocol='按周期14/5/5和6/2/2；测试140帧',question='同压迟滞回线与未来周期拟合'),
        dict(data='20260819 / 172644、181044、181548',measurement='5 Hz、15节点视觉骨架；测试2958窗口',protocol='每序列按时间6:2:2，H20；固定seed 0–4',question='历史匹配、分支干预、插件、收敛'),
        dict(data='20260819 / 182253、182519',measurement='名义10 Hz原始骨架4211帧；共同评分3937帧',protocol='冻结5 Hz模型；原生H39和抽样H20',question='同一运动记录的采样与时间积分诊断')]
    table(out,'data_scope','各数据承担的验证任务',data,[('data','数据','text'),('measurement','测量及数量','text'),('protocol','协议','text'),('question','验证任务','text')],'history')
    out.md('literature_story','''## 从已有论文借鉴的是验证逻辑

Chen先测量同压不同方向的形态差，再在多种网络容量下验证方向输入；Park进一步检查历史长度和时序编码，再做真实轨迹补偿。它们提示我们先证明输入信息的必要性，再评价建模，最后以控制检验应用价值。

Schäfke的状态预热、清零和冻结提供了分支功能诊断的思路；Krauss采用动力学核与组件交叉实验，并以模拟持压/释放检查行为。Yu和SoftNeRF则让任务直接使用其几何表示提供的能力。具体借鉴与原文定位如下；不同工作的采样、测量与真实控制频率分开理解。''','literature')
    lessons={
        'chen2025':('同压方向差 → 多容量输入对照','周期回线和方向网络；进一步检验同方向异历史'),
        'park2024':('历史长度/编码 → 未见轨迹的实际补偿','窗口以秒计；拟合和真实补偿分别评价'),
        'schafke2024':('状态预热/清零/冻结 → 跨激励预测','固定权重的内部贡献干预'),
        'krauss2026':('动力学核×组件；模拟持压/释放','功能诊断和插件；区分模拟与实测'),
        'yu2026':('形状表示 → 几何控制任务','保留全身、末端和下游任务证据'),
        'softnerf2024':('身体表示与采样 → 收敛和多任务','按完整成本解释快速建模')}
    papers=[p for p in modules['literature']['papers'] if p['id'] in lessons]
    rows=[dict(paper=p['title'],logic=lessons[p['id']][0],use=lessons[p['id']][1]) for p in papers]
    table(out,'literature_logic','论文的问题—实验对应关系',rows,[('paper','论文','text'),('logic','验证逻辑','text'),('use','本项目的采用方式','text')],'literature')
    out.md('literature_links','原文定位：'+ '；'.join(f"[{p['id']}]({p['sources'][0]['url']})" for p in papers)+'。逐篇原文证据、频率语义及适用范围保存在同目录literature.md。','literature')
    sweep_sections(out,modules['sweep'])
    history_sections(out,modules['history'])
    time_sections(out,modules['time'])
    plugin_sections(out,modules['plugin'])
    convergence_sections(out,modules['plugin'],modules['history'])
    out.md('paper_position','''## 对论文论证和后续实验的建议

引言提出的问题可以落到三个已经有对应数据的判断：相同当前驱动会产生历史相关响应；显式记忆为全身预测提供有用信息；递推记忆可以与不同基础映射结合。固定分支干预、沿臂贡献图和插件对照应与重训消融互补，承担“分支如何被使用”的分析。

可采用的结果表述是：“周期驱动数据呈现可重复的加载与卸载分支差异。所构造的历史特征改善留出响应的预测，并在不同基础映射中表现出一致的预测收益。对完整自模型的固定参数干预进一步表明，路径记忆和时间记忆均参与全身几何预测。”应同时报告当前同方向历史配对中的有限收益与时间尺度相关性，将物理解释限定为现象学建模。

如果会议版篇幅有限，建议正文保留：一张实测回线及同支离散图、一张固定分支干预与配对差异图、一张插件对照图、一张包含初始化的收敛图。采样/积分敏感性、全部seed和τ相关图放补充材料。主实验仍报告全身预测精度，下游形状规划与遮挡反馈承担模型用途的验证。''')
    next_rows=[
        dict(priority='1',experiment='同方向、同近期输入、不同早期反转',design='两条路径经过不同反转幅值，再共享完全相同的末段指令；交错执行，每条件独立重置≥5次。',measurement='全身配对坐标差及其预测误差；静态、方向、时序与HOV',purpose='检验路径记忆是否提供方向特征之外的信息'),
        dict(priority='2',experiment='非零压力保持与释放',design='从不同历史到达相同非零目标压力，保持8–10 s后释放；同步图像、命令及可用实测压力。',measurement='随保持时长的骨架变化、时间分支开关误差和暂态拟合',purpose='用实际观测验证时间记忆，区分装置滞后与材料响应'),
        dict(priority='3',experiment='相同路径的真实变速',design='固定幅值和路径，改变真实执行速度；独立重复并留出一种速度测试，采样率保持充足。',measurement='回线宽度、全身误差、真实dt与固定步长对照',purpose='评价速率泛化，和同记录降采样区分'),
        dict(priority='4',experiment='流式更新与数据效率',design='只用过去数据更新读出，在后续数据评估；按样本数及墙钟成本对比固定模型。',measurement='更新前后误差、更新延迟、漂移恢复和旧任务保持',purpose='为在线学习和快速适应提供专门证据')]
    table(out,'next_experiments','能直接检验剩余假设的补充采集',next_rows,[('priority','顺序','text'),('experiment','实验','text'),('design','操作','text'),('measurement','指标','text'),('purpose','回答的问题','text')],'literature')
    out.md('audit_note','''## 分析材料与复现

本报告的图表数据来自同目录的history_mechanisms.json、sweep_hysteresis.json、time_memory.json和plugin_convergence.json。各模块保留了预测、逐帧或逐周期表、验证选择配置与计算脚本；完整命令与核验范围见README.md。

核验包括输入/标签对齐、评分帧一致性、由保存预测重算指标、历史配对规则和seed完整性。原始图像只做针对性抽查，原始采集数据未改动。图表均可查阅数据表；HTML同时保留离线阅读所需的数据与文本。''')
    for block in out.blocks:
        if block['type']=='markdown' and 'sourceId' not in block:
            block['sourceId']='overview'
    for item in out.tables:
        if item['id']=='data_scope':item['sourceId']='overview'
    artifact=finish(out,output,stamp)
    print(json.dumps(dict(artifact=str(output/'artifact.json'),charts=len(out.charts),tables=len(out.tables),blocks=len(out.blocks)),ensure_ascii=False))
    return artifact


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'workspace/reports/modeling_mechanisms_20260913_001')
    build(p.parse_args().output.resolve())
