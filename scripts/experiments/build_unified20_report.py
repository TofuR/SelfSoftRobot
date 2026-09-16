#!/usr/bin/env python3
"""Create the source-backed portable report manifest for completed repetitions."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.dont_write_bytecode=True
RUN=ROOT/'workspace/runs/training/modeling_unified20_20260913_004'
OUT=ROOT/'workspace/runs/analysis/modeling_unified20_20260913_005'
REPORT=ROOT/'workspace/reports/modeling_unified20_20260913_005'
MAIN=['linear','pcc','base','koopman','oscillator','chen_direction','hov','window']
VARIANTS=['base','path','time','both','static_capacity','window']
CN=dict(linear='线性回归',pcc='PCC',base='静态MLP',koopman='Koopman型模型',oscillator='Krauss潜振子',
        chen_direction='Chen方向网络',hov='本文模型',window='窗口MLP',hov_no_memory='参考形态（重训）',
        hov_no_play='仅时间记忆（重训）',hov_no_maxwell='仅路径记忆（重训）',path='当前输入＋路径记忆',
        time='当前输入＋时间记忆',both='当前输入＋双记忆',static_capacity='当前输入的静态扩展')


def resolve_report_delivery():
    """Resolve the optional HTML helper only when delivery is requested."""
    explicit = os.environ.get('SELF_SOFT_REPORT_DELIVERY')
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f'SELF_SOFT_REPORT_DELIVERY is not a file: {path}. '
                'Set it to the installed deliver_portable_artifact.mjs helper.')
        return path
    codex_home = Path(os.environ.get('CODEX_HOME') or Path.home() / '.codex').expanduser()
    cache = codex_home / 'plugins/cache'
    candidates = sorted({p.resolve() for p in cache.glob(
        '*/data-analytics/*/skills/build-report/scripts/deliver_portable_artifact.mjs')
        if p.is_file()})
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise RuntimeError(
            'Multiple report delivery helpers found; set SELF_SOFT_REPORT_DELIVERY '
            'to the required version:\n' + '\n'.join(map(str, candidates)))
    raise FileNotFoundError(
        f'Optional HTML report delivery helper not found under {cache}. '
        'Install the Data Analytics report helper and its dependencies, then set '
        'SELF_SOFT_REPORT_DELIVERY to deliver_portable_artifact.mjs. '
        'HTML delivery also requires Node.js; artifact generation can run without the helper.')


class _ReportDeliveryPath(os.PathLike):
    """Keep str(shared.DELIVER) callers working without an import-time dependency."""
    def __fspath__(self):
        return str(resolve_report_delivery())

    def __str__(self):
        return self.__fspath__()


DELIVER = _ReportDeliveryPath()


def read(path): return json.loads(Path(path).read_text())
def write(path,value): Path(path).write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def clean(value):
    if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,np.generic): return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value): return None
    return value
def sci(value): return f'{float(value):.6e}'
def signed(value,places=6): return f'{float(value):+.{places}f}'
def summary_pm(record,places=4):
    return f"{record['mean']:.{places}f}"+(f" ± {record['sd']:.{places}f}" if record.get('sd') is not None else '')
def flat_pm(record,field,places=4):
    value=record[field+'_mean'];sd=record.get(field+'_sd')
    return '未记录' if value is None else f'{value:.{places}f}'+(f' ± {sd:.{places}f}' if sd is not None else '')


def pm(row,metric,places=3):
    v=row[metric];return f"{v['mean']:.{places}f}"+(f" ± {v['sd']:.{places}f}" if v['sd'] is not None else '')


class Artifact:
    def __init__(self):
        self.time=datetime.now(timezone.utc).isoformat()
        self.title='全身形态建模：20次重复实验结果与机制分析'
        self.m=dict(version=1,surface='report',title=self.title,description='统一5 Hz时间留出协议下的模型对照、记忆分析与计算成本。',generatedAt=self.time,blocks=[],charts=[],tables=[],cards=[],sources=[])
        self.datasets={}
        self.chart_map=[]
        self.notes=[]
    def source(self,id,label,path):
        path=Path(path)
        if path.is_absolute():path=path.relative_to(ROOT)
        assert '..' not in path.parts and (ROOT/path).is_file(), path
        assert id not in {s['id'] for s in self.m['sources']}, id
        self.m['sources'].append(dict(id=id,label=label,path=path.as_posix()))
    def md(self,id,body,source=None):
        b=dict(id=id,type='markdown',body=body)
        if source:b['sourceId']=source
        self.m['blocks'].append(b)
    def chart(self,id,title,subtitle,rows,x,y,kind='bar',source='main',intent='comparison',x_type='nominal',color=None,series=None,unit='mm',rationale=None,grain='每个seed的固定划分结果'):
        self.datasets[id]=clean(rows)
        enc=dict(x=dict(field=x,type=x_type,label=x),y=dict(field=y,type='quantitative',label=unit))
        if color:enc['color']=dict(field=color,type='nominal')
        c=dict(id=id,title=title,subtitle=subtitle,type=kind,dataset=id,sourceId=source,encodings=enc,valueFormat='number',
               intent=intent,question=title+'？',rationale=rationale or '在相同评价条件下比较相应量，保留必要的指标和重复单位。',
               comparisonContext=dict(grain=grain,unit=unit),layout='full',
               labels=dict(values='auto'),settings=dict(sort='none'),palette=dict(kind='sequential',name='blue'))
        if series:
            c['encodings']['y']=dict(fields=[v['field'] for v in series],type='quantitative',label=unit)
            c['palette']=dict(kind='categorical')
        if color:c['palette']=dict(kind='categorical')
        self.m['charts'].append(c);self.m['blocks'].append(dict(id=id+'_block',type='chart',chartId=id))
        self.chart_map.append(dict(id=id,question=title,type=kind,sourceId=source,fields=enc,
                                   grain=grain,rationale=c['rationale'],full_width=True))
    def table(self,id,title,rows,fields,source,sort=None):
        self.datasets[id]=clean(rows)
        cols=[dict(field=key,label=label,type='text' if typ=='text' else 'number',**({} if typ=='text' else dict(format='number'))) for key,label,typ in fields]
        self.m['tables'].append(dict(id=id,title=title,dataset=id,sourceId=source,defaultSort=dict(field=sort or fields[0][0],direction='asc'),columns=cols))
        self.m['blocks'].append(dict(id=id+'_block',type='table',tableId=id))
    def save(self):
        REPORT.mkdir(parents=True,exist_ok=True)
        write(REPORT/'artifact.json',dict(surface='report',manifest=self.m,snapshot=dict(version=1,generatedAt=self.time,status='ready',datasets=self.datasets,accessIssues=[]),sources=self.m['sources']))


def base_report():
    data=read(OUT/'summary.json');s={r['model']:r for r in data['models']};raw=pd.read_csv(RUN/'raw_test.csv');a=Artifact()
    a.source('main','正式测试逐模型逐seed指标',RUN.relative_to(ROOT)/'raw_test.csv')
    a.source('pairs','预定义配对统计及Holm校正',RUN.relative_to(ROOT)/'paired_statistics.json')
    a.source('protocol','冻结实验协议',RUN.relative_to(ROOT)/'protocol.json')
    a.source('sequences','各采集记录的测试分解',RUN.relative_to(ROOT)/'raw_test_by_sequence.csv')
    a.source('summary','完整正式结果汇总',OUT.relative_to(ROOT)/'summary.json')
    a.source('audit','独立统计、覆盖和末段验证核验',OUT/'validation/validation.json')
    a.source('plugin_definition','固定记忆特征编码器定义',RUN/'source/scripts/experiments/analyze_modeling_plugin_convergence.py')
    a.md('title','# '+a.title+'\n\n固定数据与划分 · 5 Hz · 100 epoch · 20个训练种子')
    a.md('summary','## 精度、模型规模与历史表示各有明确结果\n\n'
        f"HOV骨架误差为 **{pm(s['hov'],'mean_node_mm')} mm**，静态MLP为 **{pm(s['base'],'mean_node_mm')} mm**。HOV使用760个参数或拟合系数；窗口MLP使用12,269个，骨架误差为 **{pm(s['window'],'mean_node_mm')} mm**。HOV比窗口MLP少93.8%的参数，整体精度仍有差距。\n\n"
        '完整HOV在重训消融中优于各单分支与参考项；固定记忆编码也改善了线性与MLP的预测。几何分析进一步检验这些历史读出如何形成实际的位移修正。', 'main')
    a.md('scope','## 如何读取这些结果\n\n'
         '主测试包含2,958个目标帧。骨架误差为每帧15节点欧氏距离的平均，再合并全部测试帧；末端误差只用最后节点；IoU与Dice比较统一管状投影和视觉掩码。每个随机模型采用相同20个种子（100–119），表内“±”表示样本标准差。确定性线性模型只拟合一次。\n\n'
         '每条连续记录按时间6∶2∶2划分，再合并对应集合；训练、验证和测试窗口分别为8,988、2,958、2,958。20点窗口在各自集合内部形成。主统计推断针对固定数据条件下的训练随机性。','audit')
    a.md('main_result','## HOV优于静态与近期结构对照，窗口MLP整体精度更高\n\n'
        '主对照的七项骨架比较均在20个配对种子中方向一致。HOV相对Chen的平均误差减少0.064 mm，相对静态MLP减少0.483 mm；相对窗口MLP则增加0.074 mm。七项双侧精确Wilcoxon检验经族内Holm校正后均为1.34×10⁻⁵。\n\n'
        '这些p值检验骨架指标。末端和掩码作为补充指标，不能直接套用同一个p值。Chen的平均IoU略高于HOV，说明骨架距离与固定宽度轮廓重叠的排序不完全一致。','summary')
    box=raw[raw.model.isin(MAIN)].copy();box['模型']=box.model.map(CN);box['骨架误差']=box.mean_node_mm
    a.chart('main_distribution','各模型骨架误差分布','每点为一次训练的合并测试误差；20次随机重复，线性为单次确定性拟合。',box[['模型','骨架误差','seed','test_frames','endpoint_mm','mask_iou']].to_dict('records'),'模型','骨架误差','boxPlot',intent='distribution')
    rows=[dict(model=CN[m],skeleton=pm(s[m],'mean_node_mm'),tip=pm(s[m],'endpoint_mm'),iou=pm(s[m],'mask_iou',4),dice=pm(s[m],'mask_dice',4),parameters=s[m]['active_fitted_parameters'],n=s[m]['n']) for m in MAIN]
    a.table('main_table','全身预测主对照',rows,[('model','模型','text'),('skeleton','骨架误差/mm','text'),('tip','末端误差/mm','text'),('iou','IoU','text'),('dice','Dice','text'),('parameters','参数/拟合系数','number'),('n','重复数','number')],'summary')
    pairs=[]
    for r in data['statistics']['contrasts']:
        if r['family']=='main':pairs.append(dict(model=CN[r['alternative']],delta=-r['mean_reference_minus_alternative_mm'],ci_low=-r['bootstrap95_upper_mm'],ci_high=-r['bootstrap95_lower_mm'],p=sci(r['wilcoxon_holm_p'])))
    a.table('main_effects','主对照配对差值：对照误差−HOV误差',pairs,[('model','对照','text'),('delta','平均差/mm','number'),('ci_low','95%CI下界','number'),('ci_high','95%CI上界','number'),('p','Holm校正p','text')],'pairs',sort='delta')
    a.md('ablation_result','## 两类记忆在几何模型中共同改善预测\n\n'
        f"参考形态重训后误差为{pm(s['hov_no_memory'],'mean_node_mm')} mm，加入两类记忆后为{pm(s['hov'],'mean_node_mm')} mm，减少23.50%。仅时间记忆和仅路径记忆分别为{pm(s['hov_no_play'],'mean_node_mm')}、{pm(s['hov_no_maxwell'],'mean_node_mm')} mm。\n\n"
        '移除路径、时间或两者，使骨架误差分别增加0.099、0.079和0.453 mm；三项Holm校正p均为5.72e-06。单分支经重新训练后仍不能达到完整模型的平均精度。','summary')
    ab=raw[raw.model.isin(['hov_no_memory','hov_no_play','hov_no_maxwell','hov'])].copy();ab['模型']=ab.model.map(CN);ab['骨架误差']=ab.mean_node_mm
    a.chart('ablation_distribution','重训消融的误差分布','相同几何表示、目标帧与训练预算；每种20次。',ab[['模型','骨架误差','seed']].to_dict('records'),'模型','骨架误差','boxPlot',intent='distribution')
    a.md('ablation_parameter_note','有效拟合规模分别为：参考项196、仅时间624、仅路径352、完整模型760。原检查点为兼容性保留禁用分支，本报告在消融规模中只计参与拟合的参数及固定参考驱动系数。','summary')
    a.md('plugin_result','## 记忆编码可以复用，双分支收益取决于读出形式\n\n'
        '线性读出从当前输入的2.223 mm下降到双记忆的1.823 mm。MLP从1.959±0.020 mm下降到双记忆的1.510±0.021 mm，减少22.88%；相同输入维数的静态扩展为1.982±0.027 mm。\n\n'
        'MLP加入路径、时间、双记忆均在20次重复中改善，相关Holm校正p为9.54e-06。路径MLP的平均误差1.498 mm略低于双记忆MLP；窗口MLP仍为最优。因此，复用实验支持历史编码的价值，而双分支组合在不同读出上的最优性需要分别评价。','summary')
    lnames=['linear','linear_path','linear_time','linear_both','linear_static_capacity','linear_window'];labels=['当前输入','＋路径','＋时间','＋双记忆','静态扩展','完整窗口']
    plugin_rows=[dict(表示=label,线性=s[l]['mean_node_mm']['mean'],MLP=s[m]['mean_node_mm']['mean'],MLP标准差=s[m]['mean_node_mm']['sd'],MLP参数=s[m]['stored_parameters']) for label,l,m in zip(labels,lnames,VARIANTS)]
    a.chart('plugin_accuracy','输入表示在不同读出上的骨架误差','线性为确定性拟合；MLP为20次均值。误差标准差与配对区间见表。',plugin_rows,'表示','MLP',series=[dict(field='线性',label='线性',color='neutral',lineStyle='dashed'),dict(field='MLP',label='MLP',color='blue')],source='summary')
    plugin_effects=[]
    for r in data['statistics']['contrasts']:
        if r['family']=='plugin':plugin_effects.append(dict(model=CN[r['alternative']],delta=r['mean_reference_minus_alternative_mm'],lower=r['bootstrap95_lower_mm'],upper=r['bootstrap95_upper_mm'],wilcoxon=sci(r['wilcoxon_holm_p']),sign=sci(r['sign_test_holm_p'])))
    a.table('plugin_effects','静态MLP误差−相应表示误差：配对统计',plugin_effects,[('model','输入表示','text'),('delta','平均改善/mm','number'),('lower','95%CI下界','number'),('upper','95%CI上界','number'),('wilcoxon','Wilcoxon校正p','text'),('sign','符号检验校正p','text')],'pairs',sort='delta')
    a.md('plugin_scope','固定编码器采用新建算子的初始驱动变换与路径、时间网格，基础模型分别拟合。此实验评价编码结构复用。','plugin_definition')
    a.md('plugin_sensitivity','静态MLP减静态扩展的差值为−0.023268 mm，14/20个seed中静态MLP更低；Wilcoxon Holm p=1.068878e-02，符号检验Holm p=1.153183e-01。静态扩展平均未改善，差异是否显著对检验方法敏感。both与static_capacity的直接配对差值为−0.471410 mm，95%区间[−0.485101, −0.459176]；这项额外比较为描述性核验，不属于冻结plugin5家族。','audit')
    return a,s,data


def add_recording_section(a):
    a.md('recordings','## 不同采集记录上的模型排序有差异\n\n'
         '172644、181044、181548分别贡献2,457、238、263个测试目标。HOV在181044的骨架误差为1.380 mm，低于窗口MLP的1.455 mm；在另外两条记录上窗口MLP更低。Chen在181548的均值为1.238 mm，也低于HOV的1.320 mm。\n\n'
         '这组分解用来描述当前测试范围中的差异。主指标始终按帧合并；不能把各记录当作20次之外新增的独立重复。','sequences')
    seq=pd.read_csv(RUN/'raw_test_by_sequence.csv');g=seq[seq.model.isin(MAIN)].groupby(['model','group'],as_index=False).agg(mean_node_mm=('mean_node_mm','mean'),test_frames=('test_frames','first'));g['模型']=g.model.map(CN);g['记录']=g.group.str[-6:]
    a.chart('sequence_matrix','各采集记录的骨架误差','每组模型展示三条记录；随机方法为20次均值，线性为一次。',g.to_dict('records'),'模型','mean_node_mm','bar',source='sequences',color='记录',unit='mm',
            grain='模型×记录，先对该记录的seed结果取均值',rationale='用数值y轴的分组柱图保留所有模型与三条记录；共享验证器拒绝分类y轴热图。')
    a.table('recording_values','逐记录骨架误差与pooled权重',g.rename(columns={'mean_node_mm':'error'}).to_dict('records'),
            [('模型','模型','text'),('记录','记录','text'),('error','mean-node/mm','number'),('test_frames','每fit有效帧数','number')],'sequences')


def add_derived_sections(a):
    geometry=read(OUT/'geometry/summary.json')
    efficiency=read(OUT/'efficiency/summary.json')
    sampling=read(OUT/'sampling/summary.json')
    validation=read(OUT/'validation/validation.json')
    for part,label in [('geometry','HOV几何读出、时间核和相似输入配对'),('efficiency','初始化、训练和实测推理延迟'),('sampling','共同目标采样诊断')]:
        a.source(part,label,OUT/part/'summary.json')
    a.source('geometry_protocol','几何公式、参考输入与配对协议',OUT/'geometry/protocol.json')
    a.source('time_basis','固定指数基的独立秩结构核验',OUT/'geometry/time_basis_structure.json')
    a.source('history','正式训练的逐epoch验证记录',OUT/'efficiency/training_history_raw.csv')
    a.source('sampling_pairs','共同目标配对统计及bootstrap区间',OUT/'sampling/paired_statistics.csv')
    a.notes.append('大JSON只提取summary层与选定字段；不嵌入逐帧数组、逐次延迟或完整checkpoint来源。')

    a.md('geometry_result','## 历史状态经几何导数形成位移，一阶读出保留了大部分收益\n\n'
         '在每个完整HOV内部，联合训练的参考形态误差为1.942574±0.005567 mm；加入一阶历史位移后为1.476388±0.006884 mm，完整非线性解码为1.475481±0.007671 mm。'
         '对应残差平方能量降低50.681%与50.744%。这支持局部几何读出对当前拟合结果的解释能力。这里的reference属于已训练完整HOV；它与前述独立重训的“无记忆”消融不同。','geometry')
    a.md('geometry_specification','### 从四通道压力到15节点位移\n\n'
         '压力先经静态驱动映射得到归一化驱动e。路径状态用q=e−p表示阈值记忆偏差，时间状态用d=h−e表示指数记忆偏差；读出权重把状态映射到16维广义坐标（14个弯曲模态、2个对数长度）。'
         '在当前输入的参考广义坐标g₀附近，几何解码器F的导数J把这些坐标增量变成毫米位移：\n\n'
         '`Δg_path = W_path q；Δg_time = W_time d`\n\n'
         '`ΔX_path = J(g₀) Δg_path；ΔX_time = J(g₀) Δg_time`\n\n'
         '`X_linear = F(g₀) + ΔX_path + ΔX_time；X_full = F(g₀ + Δg_path + Δg_time)`\n\n'
         'J随参考构型变化，因此同一历史状态在不同构型上可以形成不同的节点位移。此分解描述模型内部表示，不能单独辨识真实材料的滞回或松弛机制。','geometry_protocol')
    compositions={'reference':'完整模型内参考形态','joint_linear':'参考＋一阶历史位移','full':'完整非线性读出'}
    grows=[dict(组成=label,骨架误差=r['mean_node_mm']['mean'],骨架SD=r['mean_node_mm']['sd'],
                端点误差=r['endpoint_mm']['mean'],能量降低_pct=r['residual_energy_reduction_pct']['mean'],seeds=20,test_frames=2958)
           for key,label in compositions.items() for r in [geometry['predictions'][key]]]
    a.chart('geometry_readout','参考与历史位移读出的误差','三种预测组成使用同一个完整HOV；20个seed、全部2958测试帧。',grows,'组成','骨架误差',source='geometry',
            grain='每个完整模型的预测组成，跨20个seed均值',rationale='只有三种有定义的预测组成，柱图对照其误差；精确SD和能量由相邻表提供。')
    a.table('geometry_values','预测组成及局部位移的一致性',[
        dict(组成=label,骨架=summary_pm(r['mean_node_mm'],6),末端=summary_pm(r['endpoint_mm'],6),能量=summary_pm(r['residual_energy_reduction_pct'],3))
        for key,label in compositions.items() for r in [geometry['predictions'][key]]],
        [('组成','预测组成','text'),('骨架','骨架/mm，均值±SD','text'),('末端','末端/mm，均值±SD','text'),('能量','残差能量降低/%','text')],'geometry')
    a.md('displacement_interpretation','一阶与完整预测的骨架误差仅相差0.000907 mm，但两组预测之间的平均节点距离为0.028180 mm、末端距离为0.157341 mm。'
         '“误差相近”不等于“每个位置相同”。两类位移在全部测试帧堆叠后的余弦为0.070658±0.038101；一阶合成位移与真实参考残差的余弦为0.711955±0.002800。'
         '结果表明两分支全局方向重叠较小，合成后对残差有较好的方向一致性；它不保证每一帧都相互正交或都能降低误差。'
         '全部20seed、每seed2958个测试帧的参考长度和完整长度均未触发裁剪，支持在本次测试范围内使用局部Taylor解释。','geometry')

    a.md('kernel_result','## 四通道时间核呈现低秩趋势，纵向与归一化结果限制了统一核的解释\n\n'
         '时间核把一次归一化驱动增量映射为之后的局部节点位移。横向x的原始秩一能量均值为98.896%，逐节点L2归一化后降至88.500%；'
         '纵向y分别为88.955%与85.555%，原始秩一能量最低仅60.720%。高原始秩一能量部分受到大幅值节点主导，不能据此宣称所有通道、坐标和构型共享同一时间曲线。','geometry')
    a.md('kernel_definition','时间核使用 `K(lag) = −J W_time α^(lag+1)`，其中 `α=exp(−dt/τ)`；负号来自d=h−e。'
         '单位为“mm/单位归一化驱动增量”，不是mm/kPa；未包含压力到驱动的局部斜率或静态参考响应。'
         '固定参考包括训练窗口当前输入均值，以及PC1分数10/30/50/70/90百分位附近的五个实际输入；每参考覆盖4通道、14个非基点节点、19个lag（0–3.6 s）、x/y两坐标。'
         '对每个14×19节点–lag矩阵做未中心化SVD，再检查逐节点归一化版本。','geometry_protocol')
    basis=read(OUT/'geometry/time_basis_structure.json')
    assert basis['basis_same_across_seeds'] and basis['basis_shape']==[20,6,19]
    a.md('kernel_basis_limit',
         f"**固定基的结构背景：**在学习读出W和几何J之前，6条指数曲线在19点网格上已经有{100*basis['raw_rank1_mean']:.4f}%的秩一能量；"
         f"每条曲线L2归一化后仍为{100*basis['row_normalized_rank1_mean']:.4f}%。20个seed的指数基完全相同。"
         '输出核建立在有限个相关指数基上，其高秩一比例必须结合这一结构背景解读，不能单独当作新的物理规律。'
         '基的比例与输出核比例也不能相减当作“学习贡献”。','time_basis')
    rankrows=[]
    for channel in range(4):
        for coord in ['x','y']:
            cells=[r for r in geometry['kernel_rank_by_reference_channel_coordinate'] if r['channel']==channel and r['coordinate']==coord]
            assert len(cells)==6 and all(r['defined_seeds']==20 for r in cells)
            rankrows.append(dict(通道坐标=f'通道{channel+1} · {coord}',通道=channel+1,坐标=coord,
                原始秩一_pct=100*np.mean([r['rank1_energy']['mean'] for r in cells]),
                归一化秩一_pct=100*np.mean([r['normalized_rank1_energy']['mean'] for r in cells]),
                原始最低_pct=100*min(r['rank1_energy']['min'] for r in cells),
                原始最高_pct=100*max(r['rank1_energy']['max'] for r in cells),references=6,seeds=20,cells=120))
    a.chart('kernel_rank','四通道时间核的秩一能量','各通道/坐标跨6个参考与20个seed的描述均值；两系列分别为原始矩阵和逐节点L2归一化。',
            rankrows,'通道坐标','原始秩一_pct',source='geometry',unit='%',
            series=[dict(field='原始秩一_pct'),dict(field='归一化秩一_pct')],
            grain='通道×坐标，各120个seed/参考矩阵，仅20个训练重复')
    a.table('kernel_rank_detail','通道与坐标覆盖范围',rankrows,[('通道坐标','通道与坐标','text'),('原始秩一_pct','原始秩一/%','number'),
        ('归一化秩一_pct','逐节点归一化/%','number'),('原始最低_pct','原始最低/%','number'),('原始最高_pct','原始最高/%','number')],'geometry')
    a.source('kernel_nodes','各节点的带符号时间核增益',OUT/'geometry/kernel_node_metrics.csv')
    a.source('kernel_factor_notes','固定符号规范下的节点增益和通道波形',ROOT/'docs/icra2027/figures/unified20/figure_geometry_notes.json')
    factor=read(ROOT/'docs/icra2027/figures/unified20/figure_geometry_notes.json')['figures']['time_kernel_gain']
    gain=np.asarray(factor['plotted_per_seed']['gain_ref0'])
    waveform=np.asarray(factor['plotted_per_seed']['waveform_ref0'])
    assert gain.shape==(20,4,14) and waveform.shape==(20,4,19)
    gain_rows=[dict(节点=node+1,通道=f'通道{channel}',空间增益=float(gain[:,channel,node].mean()),样本SD=float(gain[:,channel,node].std(ddof=1)),
                   线型=['solid','dashed','dotted','dashed'][channel],reference_index=0,coordinate='x',seeds=20)
               for channel in range(4) for node in range(14)]
    assert len(gain_rows)==56 and all(r['seeds']==20 for r in gain_rows)
    a.md('node_spatial_structure','### 节点空间增益在四通道中呈现不同符号与分布\n\n'
         '下图固定训练均值参考点ref0与横向x，按每个seed/通道独立SVD分解 `K≈g fᵀ，g=σ₁u₁，f=v₁`。'
         '时间波形f在19点网格上离散L2范数为1；绝对值最大的波形采样点取正，平局取最早lag，对g施加同一符号。'
         '先分解和固定符号，再跨20seed取均值；均值因子的乘积不必等于均值核。'
         '通道0/1的主导增益沿臂改变符号，通道2/3保持相反符号；远端幅值较大，各通道空间分布与波形不同。'
         '增益单位mm/单位Δe，依赖构型和因子规范；不等于完整核峰值、冲量积分或压力灵敏度。','kernel_factor_notes')
    a.chart('node_spatial_gain','训练均值参考点的节点空间增益','ref0 · x · SVD主导分量；四通道各14节点，20seed均值，样本SD保留在图表数据中。',
            gain_rows,'节点','空间增益',kind='line',source='kernel_factor_notes',x_type='quantitative',color='通道',intent='trend',
            unit='mm/单位Δe',grain='固定ref0/x的SVD主导分量：通道×节点，20seed均值',
            rationale='按基部到末端14节点顺序展示完整四通道的SVD主导signed增益；逐seed固定符号后汇总。')
    a.m['charts'][-1]['encodings']['lineStyle']=dict(field='线型',type='nominal')
    a.m['charts'][-1]['palette']=dict(kind='sequential',name='blue')
    a.notes.append('节点增益图声明蓝色系palette并绑定通道与线型；具体最终颜色由共享reader决定，structural_only不验证像素颜色。论文time_kernel_gain使用蓝/橙/灰及线型。')
    a.md('channel_waveform_interpretation','每条通道的主导时间波形在该通道内用于近似各节点；“共享波形”限于同一通道，不能扩展为四通道共用一条曲线。'
         '部分波形经过零点后改变符号，低增益节点也可能保留被主导分量忽略的差异。完整热图和秩结构提供近似误差的补充证据。','kernel_factor_notes')
    wave_rows=[dict(lag_s=lag*.2,通道=f'通道{channel}',波形=float(waveform[:,channel,lag].mean()),
                    样本SD=float(waveform[:,channel,lag].std(ddof=1)),线型=['solid','dashed','dotted','dashed'][channel],
                    reference_index=0,seeds=20) for channel in range(4) for lag in range(19)]
    a.chart('node_time_waveform','各通道主导分量的时间波形','ref0 · x；每通道19个lag，f逐seed离散L2归一化，符号规则与节点增益完全一致。',
            wave_rows,'lag_s','波形',kind='line',source='kernel_factor_notes',x_type='quantitative',color='通道',intent='trend',unit='无量纲',grain='通道×lag的20seed均值')
    a.m['charts'][-1]['encodings']['lineStyle']=dict(field='线型',type='nominal')
    a.m['charts'][-1]['palette']=dict(kind='sequential',name='blue')
    a.md('kernel_scope','图中每个通道/坐标汇总120个矩阵（6参考×20seed）；每个坐标合计480个矩阵。这些是描述性覆盖单元，独立训练重复仍为20。'
         '四通道间和x/y间的差异要求保留构型与方向信息；进一步验证应使用独立加载–保持轨迹，而不能把低秩比例解释为已测得的单一物理松弛模式。','geometry')

    a.md('history_pair_result','## 当前输入相近的历史配对同时给出改善与反例\n\n'
         '固定5 kPa当前输入容差下，反向组216对的形态差值误差由2.120216 mm降至1.631250 mm，降低0.488966 mm；'
         '同向组58对由1.184692升至1.208401 mm，最近两步相近组38对由1.169682升至1.195442 mm。'
         '完整模型只在反向组显示改善，其他两组构成当前诊断中的反例。','geometry')
    a.md('history_pair_definition','配对只依赖压力历史和时间，不使用模型误差选择。主容差为当前四通道最大差≤5 kPa；同序列帧间至少隔20帧、前19步历史RMS差≥20 kPa。'
         '“反向”要求当前变化方向相反；“同向”要求方向相近；“最近两步相近”进一步约束最近两步输入。'
         '评价量为 `mean_node ||(预测B−预测A)−(观测B−观测A)||`，因此它评估形态变化的预测误差。'
         '各类别、不同容差可能重叠；每个序列/类别/容差内使用不重复帧的贪心配对。相邻窗口与视觉标签的相关性仍存在。','geometry_protocol')
    cats={'opposite':'当前近似·变化反向','same_direction':'当前近似·变化同向','same_recent_two':'最近两步近似'}
    pairrows=[]
    for r in geometry['matched_pairs']:
        if r['tolerance_kpa']!=5:continue
        ref=r['predictions']['reference']['delta_mean_node_mm'];full=r['predictions']['full']['delta_mean_node_mm']
        pairrows.append(dict(条件=cats[r['category']],帧对数=r['pairs'],当前差_kPa=r['input_conditions']['current_gap_kpa']['mean'],
            历史差_kPa=r['input_conditions']['history_rms_kpa']['mean'],参考误差=ref['mean'],完整误差=full['mean'],
            参考SD=ref['sd'],完整SD=full['sd'],完整减参考=full['mean']-ref['mean'],seeds=20,tolerance_kpa=5))
    a.chart('history_pairs','相似当前输入配对的形态差值误差','固定5 kPa；每个seed先按同一组帧对汇总，再报告20次均值。',pairrows,'条件','完整误差',source='geometry',
            series=[dict(field='参考误差'),dict(field='完整误差')],grain='固定配对类别的跨seed均值',
            rationale='三种协议类别均完整保留；分组柱图直接呈现改善和反向结果，不扩充不存在的类别。')
    a.table('history_pair_values','5 kPa配对覆盖与误差方向',pairrows,[('条件','条件','text'),('帧对数','帧对数','number'),
        ('当前差_kPa','当前最大通道差均值/kPa','number'),('历史差_kPa','历史RMS差均值/kPa','number'),
        ('参考误差','参考误差/mm','number'),('完整误差','完整误差/mm','number'),('完整减参考','完整−参考/mm','number')],'geometry')
    sensitivity=[dict(容差=r['tolerance_kpa'],条件=cats[r['category']],帧对数=r['pairs'],
                     完整减参考=r['predictions']['full']['delta_mean_node_mm']['mean']-r['predictions']['reference']['delta_mean_node_mm']['mean'])
                 for r in geometry['matched_pairs']]
    a.table('history_pair_tolerances','1/2/5/10 kPa全部容差的描述性结果',sensitivity,[('容差','当前输入容差/kPa','number'),
        ('条件','配对条件','text'),('帧对数','帧对数','number'),('完整减参考','完整−参考误差/mm','number')],'geometry',sort='容差')
    a.md('history_pair_scope','容差表完整保留三类配对在1/2/5/10 kPa下的结果；正差表示完整模型更差。'
         '配对是近似匹配，不是控制变量实验；重复使用同一固定帧对的20个seed只描述训练随机性。结论应落到具体历史条件，不能概括为所有相似输入配对均获益。','geometry')

    add_sampling_section(a,sampling)
    add_efficiency_section(a,efficiency,validation)
    add_validation_section(a,validation)


def add_sampling_section(a,data):
    names={'hov':'HOV','hov_no_maxwell':'HOV无时间分支（重训）'}
    protocols={'nominal_10Hz_H39':'10 Hz · H39 · dt0.1','decimated_5Hz_H20':'抽样5 Hz · H20 · dt0.2',
               'wrong_dt_10Hz_H39_dt0.2':'H39 · 错误dt0.2'}
    pooled=[r for r in data['seed_summary'] if r['scope']=='pooled']
    stats=[r for r in data['paired_statistics'] if r['scope']=='pooled']
    a.md('sampling_result','## 同目标5/10 Hz诊断显示采样代价依赖模型与指标\n\n'
         '40个冻结checkpoint（HOV与无时间分支各20个seed）在两条旧10 Hz记录的3937个共同目标上比较。'
         'HOV抽样5 Hz相对10 Hz的骨架误差增加0.024999 mm，末端增加0.066762 mm；无时间分支的骨架增加0.013166 mm，末端反而降低0.041988 mm。'
         '采样损失因此不是只属于时间分支的统一效应。三种协议使用同一目标标签，权重为1061/3937与2876/3937。','sampling')
    a.md('sampling_design','10 Hz使用39帧、dt=0.1 s；抽样5 Hz沿同一执行轨迹取stride2、20帧、dt=0.2 s。'
         '二者首尾曝光和目标相同，名义历史跨度均为3.8 s，首帧均衡初始化后分别执行38与19次更新。'
         '错误dt对照保留39帧，却误用0.2 s积分：观察名义跨度仍3.8 s，模型积分跨度变为7.6 s。'
         '抽样会遗漏中间命令，并不代表机器人以5 Hz重新执行；这些诊断也没有重新训练模型。','sampling')
    rows=[]
    for model in names:
        for protocol in protocols:
            node=next(r for r in pooled if r['model']==model and r['protocol']==protocol and r['metric']=='node_mean_mm')
            tip=next(r for r in pooled if r['model']==model and r['protocol']==protocol and r['metric']=='endpoint_mean_mm')
            rows.append(dict(模型=names[model],协议=protocols[protocol],骨架误差=node['mean_mm'],骨架SD=node['sd_mm'],
                             末端误差=tip['mean_mm'],末端SD=tip['sd_mm'],seeds=20,frames=3937))
    a.chart('sampling_accuracy','共同目标上的三种采样协议','相同冻结模型和3937个标签；名义采样与错误积分步长分别比较。',rows,'协议','骨架误差',color='模型',source='sampling',
            grain='模型×采样协议，20seed的pooled误差均值')
    a.table('sampling_metrics','共同目标的骨架与末端误差',[
        dict(模型=r['模型'],协议=r['协议'],骨架=f"{r['骨架误差']:.6f} ± {r['骨架SD']:.6f}",
             末端=f"{r['末端误差']:.6f} ± {r['末端SD']:.6f}") for r in rows],
        [('模型','冻结模型','text'),('协议','协议','text'),('骨架','骨架/mm，均值±SD','text'),('末端','末端/mm，均值±SD','text')],'sampling')
    comparisons={'hov_sampling_5Hz_minus_10Hz':'HOV：抽样5Hz−10Hz',
        'no_maxwell_sampling_5Hz_minus_10Hz':'无时间：抽样5Hz−10Hz','10Hz_no_maxwell_minus_hov':'10Hz：无时间−HOV',
        '5Hz_no_maxwell_minus_hov':'5Hz：无时间−HOV','sampling_interaction_hov_minus_no_maxwell':'采样差值：HOV−无时间',
        'hov_wrong_dt_minus_correct_10Hz':'HOV：错误dt−正确10Hz',
        'no_maxwell_wrong_dt_minus_correct_10Hz':'无时间：错误dt−正确10Hz'}
    table=[]
    for r in stats:
        table.append(dict(对比=comparisons[r['comparison']],指标='骨架' if r['metric']=='node_mean_mm' else '末端',
            差值=signed(r['mean_delta_mm']),bootstrapCI=f"[{r['ci95_bootstrap_low_mm']:.6f}, {r['ci95_bootstrap_high_mm']:.6f}]",
            p=sci(r['wilcoxon_p_two_sided']),holm=sci(r['wilcoxon_p_holm']),family=r['family'],检验数=r['family_tests']))
    a.md('sampling_uncertainty','下表和论文采样图统一使用**20,000次配对seed重抽样的95% percentile bootstrap CI**；不是摘要文件中的Student-t区间。'
         '差值方向写在行名中，正值表示前者误差更高。采样诊断按5个对比×3个scope×2指标组成30项Holm家族；错误dt诊断为12项，均与正式main7/ablation3/plugin5分开。'
         '表内显示pooled的14行，校正保留整个30/12项家族；区间为逐项区间，未作多重性校正。','sampling')
    a.table('sampling_effects','采样诊断：有符号差值与配对bootstrap 95%CI',table,[('对比','比较方向','text'),('指标','指标','text'),
        ('差值','平均差/mm','text'),('bootstrapCI','配对bootstrap 95%CI/mm','text'),('p','Wilcoxon p','text'),
        ('holm','Holm p','text'),('检验数','校正家族大小','number')],'sampling_pairs')
    sequence_rows=[]
    for r in data['paired_statistics']:
        if r['scope']=='pooled' or r['comparison']!='hov_sampling_5Hz_minus_10Hz' or r['metric']!='node_mean_mm':continue
        sequence_rows.append(dict(记录=r['scope'],差值=signed(r['mean_delta_mm']),CI=f"[{r['ci95_bootstrap_low_mm']:.6f}, {r['ci95_bootstrap_high_mm']:.6f}]",p=sci(r['wilcoxon_p_holm'])))
    a.table('sampling_recordings','HOV抽样5Hz−10Hz的逐记录骨架差值',sequence_rows,[('记录','记录','text'),('差值','差值/mm','text'),
        ('CI','配对bootstrap 95%CI/mm','text'),('p','30项Holm p','text')],'sampling_pairs')
    a.md('sampling_diagnostics','总体骨架增幅主要来自182519：HOV在182253与182519的抽样差值分别为+0.001820和+0.033550 mm。'
         'HOV使用错误dt后的骨架/末端误差再增加0.043154/0.195267 mm；无时间分支的两种dt结果逐目标完全相同。'
         '这说明时间更新对步长敏感；同时，无时间模型仍有采样效应，提示遗漏命令、路径状态和目标记录也影响比较。'
         '诊断保持τ固定，按指定步长重新计算指数系数α=exp(−dt/τ)，未重新拟合；这里不能把τ称为本轮学得的物理时间常数。','sampling')
    timing=[]
    for row in data['timing_summary']:
        alignment=next(r for r in data['alignment'] if r['sequence']==row['sequence'])
        timing.append(dict(记录=row['sequence'],目标数=alignment['scored_common_frames'],实际Hz=row['actual_mean_hz'],
                          曝光跨度=alignment['actual_exposure_span_s']['mean'],剔除数=alignment['excluded_timing_or_label_windows']))
    a.table('sampling_alignment','共同目标覆盖与实际曝光时序',timing,[('记录','记录','text'),('目标数','共同目标','number'),
        ('实际Hz','实际采集均值/Hz','number'),('曝光跨度','39帧实际曝光跨度/s','number'),('剔除数','起始上下文后被剔除窗口','number')],'sampling')
    a.md('sampling_limits','实际采集约9.05–9.08 Hz，共同窗口曝光跨度约4.19 s，模型仍使用名义dt；本次没有按真实事件时间积分。'
         '共同目标过滤要求39帧命令均ACK、曝光与ACK/issue次序有效且目标标签有效。两种奇偶抽样相位合并后每个目标只计一次。'
         '这两条记录不与正式三条训练记录重叠，但仍属于旧记录上的冻结模型诊断；20seed区间不代表跨记录或机器人总体区间。'
         '标签来自旧10 Hz视觉流程，窗口重叠、平面中心线和跨帧传播限制仍须保留。','sampling')


def add_efficiency_section(a,data,validation):
    training={r['model']:r for r in data['training']['summary']}
    a.md('early_training','## HOV初始化已达较低误差，其他模型末段仍有明显改善\n\n'
         'HOV在reference与记忆初始化之后的验证误差为1.630999 mm；epoch1为1.643720±0.045616 mm，最终选中checkpoint的验证误差为1.507384±0.004827 mm。'
         '首轮到最佳的seed平均下降为8.227%，epoch80之后新增最佳改善为0.193%。对比之下，Koopman与双记忆MLP在80之后仍分别获得6.224%与5.038%的新增下降。'
         '初始化优势包含拟合先验与记忆读出初始化，并非相同随机初值下纯Adam的比较。','efficiency')
    history=pd.read_csv(OUT/'efficiency/training_history_raw.csv')
    curve_models=['hov','chen_direction','base','window']
    curve=history[history.model.isin(curve_models)].groupby(['epoch','model']).validation_node_mean_mm.mean().unstack('model')
    curves=[dict(epoch=int(epoch),**{CN[m]:float(row[m]) for m in curve_models},seeds=20,val_frames=2958) for epoch,row in curve.iterrows()]
    a.chart('training_curve','训练中的验证误差','HOV、Chen、静态MLP、窗口MLP；每个已记录epoch先跨20seed取均值，未把最佳值连成曲线。',
            curves,'epoch',CN['hov'],kind='line',source='history',x_type='quantitative',intent='trend',
            series=[dict(field=CN[m]) for m in curve_models],grain='epoch1及每5epoch的跨seed均值')
    rows=[]
    for m in MAIN:
        r=training[m]
        rows.append(dict(模型=CN[m],次数=r['n_models'],初始化=flat_pm(r,'epoch0_val_mm',3),首轮=flat_pm(r,'epoch1_val_mm',3),
            最佳=flat_pm(r,'best_val_mm',3),首轮下降=flat_pm(r,'epoch1_to_best_improvement_pct',3),
            末段下降=flat_pm(r,'after_epoch80_best_improvement_pct',3) if m!='linear' else '单次闭式解',
            fit秒=flat_pm(r,'recorded_fit_wall_seconds',3),COMPLETE秒=flat_pm(r,'task_wall_to_COMPLETE_seconds',3)))
    a.table('training_stages','全部主方法的早期验证与训练墙钟',rows,[('模型','模型','text'),('次数','fit数','number'),
        ('初始化','初始化val/mm','text'),('首轮','epoch1 val/mm','text'),('最佳','最佳val/mm','text'),
        ('首轮下降','首轮→最佳下降/%','text'),('末段下降','80后新增最佳改善/%','text'),('fit秒','记录fit wall/s','text'),('COMPLETE秒','至COMPLETE wall/s','text')],'efficiency')
    a.md('initialization_scope','history没有epoch0行，初始化后验证值来自manifest，epoch0不参与checkpoint选择。'
         'HOV原始预拟合reference的验证误差2.045591 mm由保存的geometry_config补充重建；它不是训练当时记录的history0。'
         '初始化checkpoint及epoch0时间点没有保存。线性模型的epoch0已经完成闭式求解，其epoch1没有Adam更新。','efficiency')
    a.md('late_training','## 固定100epoch保证预算一致，不能证明全部收敛\n\n'
         '独立history核验显示，280个随机fit中200个在epoch80后刷新最佳验证值，80个最佳在epoch100；180个val100低于val90。'
         'both与Koopman各20/20个fit在80之后刷新最佳值，新增改善分别为0.083124和0.116509 mm。HOV为10/20，新增改善0.002920 mm。'
         '训练末段存在明显方法差异；结果应称为固定100epoch预算下验证选择checkpoint的表现，不能预判更长训练后的排序。','audit')
    late=[dict(模型=CN.get(r['model'],r['model']),最佳在80后=r['selected_epoch_gt80'],最佳在100=r['selected_epoch100'],
               最后验证下降=r['val100_below_val90'],新增改善=r['best80_to_best100_gain_mm']['mean']) for r in validation['convergence']]
    a.table('late_validation','全部14个随机模型的末段验证诊断',late,[('模型','模型','text'),('最佳在80后','最佳epoch>80 /20','number'),
        ('最佳在100','最佳epoch100 /20','number'),('最后验证下降','val100<val90 /20','number'),('新增改善','best80−best100/mm','number')],'audit')
    a.md('training_cost','HOV记录fit wall为54.155±0.931 s，其中模型构建/reference预拟合5.395±0.479 s、记忆初始化0.144±0.012 s、正式minibatch训练47.518±0.697 s。'
         '原实验使用8个worker并发、每worker单线程；这些是共享CPU负载下的任务墙钟，不能当作独占CPU时间或把各任务时长相加视作项目历时。'
         '记录fit计时在最佳checkpoint回放之前结束，至COMPLETE约54.236 s。共享数据读取、特征构建、标准化、worker启动和调参未被完整逐fit计时，全离线管线耗时保留未知。','efficiency')

    modes={'linear_h20':'线性回归 H20','pcc_h20':'PCC H20','base_h20':'静态MLP H20','koopman_h20':'Koopman H20',
           'oscillator_h20':'Krauss H20','chen_direction_h20':'Chen H20','hov_h20_full':'HOV完整H20',
           'hov_cached_step':'HOV缓存一步','window_h20':'窗口MLP H20'}
    a.md('inference_result','## 参数少不等于本实现更快，缓存一步缩短HOV求值时间\n\n'
         '实测每seed p50的均值：HOV完整H20为2.3872 ms，缓存一步为0.6490 ms，窗口MLP为0.0967 ms，静态MLP为0.0960 ms。'
         '相应p95均值为2.6472、0.7413、0.1105和0.1095 ms。HOV缓存路径比完整重算约快3.68倍，但窗口MLP在本实现中仍更快。'
         '这说明模型参数规模与实际求值开销必须分别报告。','efficiency')
    lookup={r['mode']:r for r in data['inference']['summary']}
    rows=[dict(路径=label,p50_ms=r['p50_ms_mean'],p95_ms=r['p95_ms_mean'],p50_SD=r['p50_ms_sd'],p95_SD=r['p95_ms_sd'],
               fit数=r['n_models'],每模型次数=500,总调用数=r['pooled_n_calls'],池化p50=r['pooled_p50_ms'],池化p95=r['pooled_p95_ms'])
          for mode,label in modes.items() for r in [lookup[mode]]]
    a.chart('inference_latency','CPU单次调用延迟','batch1、单线程、预热50次，每个冻结模型500次；图中分位数先逐模型计算再取均值。',
            rows,'路径','p50_ms',series=[dict(field='p50_ms'),dict(field='p95_ms')],source='efficiency',unit='ms',grain='每个实现路径，逐冻结模型分位数的跨seed均值')
    a.table('latency_values','实际推理p50/p95及重复单位',[
        dict(路径=label,fit数=r['n_models'],p50=flat_pm(r,'p50_ms',4),p95=flat_pm(r,'p95_ms',4),
             总调用数=r['pooled_n_calls'],池化p50=f"{r['pooled_p50_ms']:.4f}",池化p95=f"{r['pooled_p95_ms']:.4f}")
        for mode,label in modes.items() for r in [lookup[mode]]],
        [('路径','计时路径','text'),('fit数','冻结模型数','number'),('p50','逐seed p50均值±SD/ms','text'),
         ('p95','逐seed p95均值±SD/ms','text'),('总调用数','实测调用数','number'),('池化p50','全部调用池化p50/ms','text'),('池化p95','全部调用池化p95/ms','text')],'efficiency')
    a.md('inference_protocol','硬件为Intel Xeon Platinum 8336C @2.30 GHz，绑核0，CPU float32，PyTorch 2.6.0，intra/inter-op各1线程，batch1。'
         '每冻结模型预热50次、实测500次；模型/seed逐样本轮换并周期反转顺序，测量未做系统隔离。'
         '主表把逐seed分位数均值±SD与全部调用池化分位数分列，两者不是同一个量。','efficiency')
    a.md('cache_scope','HOV完整路径计入窗口更新、H20平衡起点的预烧入、一步状态更新、几何读出和毫米反归一化；'
         '缓存一步从同一H20前19步预计算的状态出发，计入状态更新/返回、几何读出和毫米输出，但不计缓存准备。'
         '20个seed各550个窗口的缓存与同一H20完整重算最大坐标差为0 mm，输入缓存未被原位修改。'
         '这一正确性检查不等同于长期连续流式状态的精度验证。窗口MLP计入窗口更新、80维展平、训练mean/std标准化、MLP和15×3毫米输出。'
         '所有路径都排除磁盘读取、模型加载、视觉、通信和控制器；延迟不代表闭环控制频率或真实执行吞吐。','efficiency')


def add_validation_section(a,data):
    a.md('independent_validation','## 完整性与配对统计已独立复算，推断范围仍限定于固定数据\n\n'
         '286个正式fit=14×20+6；raw_test有286条、逐序列有858条、配对差值有300条。全部seed100–119及所有对照保留，静态MLP与窗口MLP通过别名复用同一批checkpoint。'
         '每份保存预测均对齐同一2958个测试目标；逐帧坐标指标、pooled计权、15项正式检验的p值/Holm调整与配对CI全部复现。'
         '第一序列占83.062880%测试权重，主指标不是三个序列等权平均。','audit')
    a.md('statistics_method','正式检验使用双侧exact Wilcoxon符号秩枚举；先移除零差，再对绝对值ties给平均秩。'
         '当前15项均为20个非零差且无绝对值ties；另用全零、ties+zeros等五个边界例核验原实现。'
         'main7、ablation3、plugin5分别Holm；符号检验作为敏感性结果。正式95%CI为配对seed均值差的20,000次percentile bootstrap区间，随机种子20260913。'
         '线性主对照是20个HOV结果减同一个固定线性解，线性独立fit仍为1。所有区间为逐项区间，均不代表独立机器人或新数据总体的不确定性。','audit')
    a.table('validation_receipt','独立核验的关键一致性条件',[
        dict(项目='正式fit / raw / by-sequence',结果='286 / 286 / 858，完整且唯一'),
        dict(项目='随机模型配对seed',结果='14模型均完整100–119，无按结果删seed'),
        dict(项目='正式检验与区间',结果='15项p、Holm p、均值差及CI端点均复现'),
        dict(项目='逐序列合成pooled',结果='最大绝对数值差1.78e-15'),
        dict(项目='mask范围',结果='201 fit×2958帧；85 fit按协议未评估mask'),
        dict(项目='证据层级',结果='保存预测、history、配置和状态文件；固定划分上的训练随机性')],
        [('项目','检查项','text'),('结果','结果','text')],'audit')


def stage_presentation_queries(a):
    """Real SQLite readback, preserving Python/file lineage explicitly.

    This is a presentation query, not a replacement claim about how the
    scientific metrics were originally computed.
    """
    REPORT.mkdir(parents=True,exist_ok=True)
    db_path=REPORT/'report_data.sqlite'
    upstream={s['id']:s for s in a.m['sources']}
    audit=[]
    def quote(name):return '"'+name.replace('"','""')+'"'
    with sqlite3.connect(db_path) as connection:
        for item in a.m['charts']+a.m['tables']:
            dataset=item['dataset'];rows=clean(a.datasets[dataset])
            fields=list(rows[0])
            assert all(set(r)==set(fields) for r in rows),dataset
            connection.execute(f'DROP TABLE IF EXISTS {quote(dataset)}')
            # No type affinity: retain exact int/float/text/null representations.
            connection.execute(f'CREATE TABLE {quote(dataset)} (row_index INTEGER PRIMARY KEY, '+', '.join(quote(f) for f in fields)+')')
            connection.executemany(f'INSERT INTO {quote(dataset)} VALUES ('+','.join('?' for _ in range(len(fields)+1))+')',
                                   [(i,)+tuple(r[f] for f in fields) for i,r in enumerate(rows)])
            sql='SELECT '+', '.join(quote(f) for f in fields)+f' FROM {quote(dataset)} ORDER BY row_index'
            readback=[dict(zip(fields,values)) for values in connection.execute(sql).fetchall()]
            assert json.dumps(readback,ensure_ascii=False,sort_keys=True)==json.dumps(rows,ensure_ascii=False,sort_keys=True),dataset
            a.datasets[dataset]=readback
            old_id=item['sourceId'];source_id='presentation_'+dataset
            path=upstream[old_id]['path']
            a.source(source_id,item['title']+'：真实SQLite展示查询',db_path)
            a.m['sources'][-1].update(upstreamSourceIds=[old_id],upstreamPaths=[path],
                query=dict(engine='sqlite',language='sql',sql=sql,id=dataset,
                           description='Presentation query over staged Python-derived metrics; upstream CSV/NPZ/JSON: '+path+
                               '; staging and scientific field selection: scripts/experiments/build_unified20_report.py. SQL only reads reviewed presentation rows; scientific analysis remains in upstream files.',
                           tables_used=[dataset],executed_at=a.time))
            item['sourceId']=source_id
            audit.append(dict(dataset=dataset,rows=len(rows),columns=fields,query=sql,sourceId=source_id,
                              upstreamSourceId=old_id,upstreamPath=path,exact_json_readback=True))
    for entry in a.chart_map:
        chart=next(c for c in a.m['charts'] if c['id']==entry['id'])
        entry['upstreamSourceId']=entry['sourceId'];entry['sourceId']=chart['sourceId']
    write(REPORT/'build_query_audit.json',dict(database=db_path.relative_to(ROOT).as_posix(),
        purpose='Canonical builder compatibility through real presentation SQL; upstream scientific provenance retained.',
        datasets=audit,all_rows_exact=True,executed_at=a.time))


def report_preflight(a,previous,catalog):
    """Check bounded evidence and provenance, leaving renderer schema to delivery."""
    source_ids={s['id'] for s in a.m['sources']}
    block_ids=[b['id'] for b in a.m['blocks']]
    assert len(block_ids)==len(set(block_ids))
    assert len(source_ids)==len(a.m['sources'])
    for source in a.m['sources']:
        path=Path(source['path'])
        assert not path.is_absolute() and '..' not in path.parts and (ROOT/path).is_file()
    for kind in ['charts','tables','cards','blocks']:
        for item in a.m[kind]:
            if 'sourceId' in item:assert item['sourceId'] in source_ids, item['id']
            if kind in ['charts','tables','cards']:assert 'sourceId' in item, item['id']
    for c in a.m['charts']:
        rows=a.datasets[c['dataset']]
        fields=c['encodings']['y'].get('fields') or [c['encodings']['y']['field']]
        assert rows and len(rows)<=1000
        for field in fields:
            assert all(isinstance(r.get(field),(int,float)) and not isinstance(r.get(field),bool) for r in rows), (c['id'],field)
        index=next(i for i,b in enumerate(a.m['blocks']) if b.get('chartId')==c['id'])
        assert any(0<=j<len(a.m['blocks']) and a.m['blocks'][j]['type']=='markdown' for j in [index-1,index+1]), c['id']
    p_columns=0
    for table in a.m['tables']:
        assert table['defaultSort']['field'] in [c['field'] for c in table['columns']]
        for col in table['columns']:
            if col['field'] in {'p','holm','wilcoxon','sign'}:
                p_columns+=1
                assert col['type']=='text'
                assert all(re.fullmatch(r'\d\.\d{6}e[+-]\d+',r[col['field']]) for r in a.datasets[table['dataset']])
    assert p_columns>=6
    assert len(a.datasets['main_table'])==8
    assert {'静态MLP','窗口MLP'} <= {r['model'] for r in a.datasets['main_table']}
    assert len(a.datasets['plugin_accuracy'])==6 and len(a.datasets['plugin_effects'])==5
    assert len(a.datasets['sampling_effects'])==14 and len(a.datasets['history_pair_tolerances'])==12
    assert len(a.datasets['late_validation'])==14
    preserved={}
    if previous:
        for kind in ['blocks','charts','tables','sources']:
            before={r['id'] for r in previous['manifest'][kind]}
            after={r['id'] for r in a.m[kind]}
            assert before<=after, (kind,before-after)
            preserved[kind]=dict(previous=len(before),retained=len(before&after),added=len(after-before))
    figure_files=[]
    for item in catalog:
        for fmt in item['formats']:
            path=REPORT/'figures'/f"{item['id']}.{fmt}"
            assert path.is_file() and path.stat().st_size>0,path
            figure_files.append(path.relative_to(ROOT).as_posix())
    note=dict(schema='unified20_report_build_notes_v1',built_at=a.time,surface='portable_html',audience='technical',
        canonical_artifact=(REPORT/'artifact.json').relative_to(ROOT).as_posix(),
        sources_project_relative=True,provenance='Every numeric markdown block and every chart/table has a precise sourceId; derived summary fields only.',
        chart_map=a.chart_map,notes=a.notes,
        required_structure=dict(title='title',technical_summary='summary',key_findings=['main_result','ablation_result','plugin_result','geometry_result','kernel_result','history_pair_result','sampling_result','early_training','inference_result'],
            scope='scope',methodology=['geometry_specification','geometry_specification','sampling_design','statistics_method'],
            limitations=['kernel_basis_limit','history_pair_scope','sampling_limits','cache_scope','limits'],next_steps='limits',further_questions='further_questions'),
        structure_mapping_note='Scope moved ahead of detailed comparisons so units and denominator precede evidence; methods and limitations are adjacent to findings.',
        omitted_visuals='Exact statistics, source checks and coverage use tables. Four major training curves are shown; all main models retained in the early-training table, all 14 in late-validation table.',
        native_heatmap_replacement='sequence_matrix id retained, replaced categorical-y heatmap with grouped bar using numeric y.',
        sampling_ci='paired_statistics.csv ci95_bootstrap_low_mm/high_mm, 20000 paired resamples; report and paper figure use bootstrap, not the summary.md t CI.',
        tiny_p_display='Scientific notation strings; all p columns type=text.',p_text_columns=p_columns,
        baseline_structure_preservation=preserved,
        counts=dict(blocks=len(a.m['blocks']),charts=len(a.m['charts']),tables=len(a.m['tables']),sources=len(a.m['sources']),
                    datasets=len(a.datasets),snapshot_rows=sum(len(v) for v in a.datasets.values()),paper_figures=len(catalog)),
        figure_catalog=OUT.relative_to(ROOT).as_posix()+'/figure_catalog.json',
        figure_catalog_mtime_ns=(OUT/'figure_catalog.json').stat().st_mtime_ns,
        figure_catalog_read_at=datetime.now(timezone.utc).isoformat(),figure_ids=[r['id'] for r in catalog],figure_files=figure_files,
        paper_figures_role='Optional sibling SVG/PDF/PNG links; report narrative, native charts and semantic tables are embedded in report.html.',
        verification_policy='Packaged deliver_portable_artifact.mjs; structural_only accepted, no browser installation or manual browser search.',
        preflight='passed')
    write(REPORT/'build_notes.json',note)
    return note


def validate_paper_links():
    """Check the generated HTML anchors, not just canonical Markdown paths."""
    class Anchors(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.links=[]
            self.current=None
        def handle_starttag(self,tag,attrs):
            if tag=='a':self.current=dict(href=dict(attrs).get('href'),text=[])
        def handle_data(self,text):
            if self.current is not None:self.current['text'].append(text)
        def handle_endtag(self,tag):
            if tag=='a' and self.current is not None:
                self.links.append(dict(href=self.current['href'],label=''.join(self.current['text']).strip()))
                self.current=None

    notes=read(REPORT/'build_notes.json')
    expected={f'./figures/{figure}.{fmt}' for figure in notes['figure_ids'] for fmt in ['svg','pdf','png']}
    artifact=read(REPORT/'artifact.json')
    markdown=next(b['body'] for b in artifact['manifest']['blocks'] if b['id']=='figures')
    canonical=re.findall(r'\]\(([^)]+)\)',markdown)
    parser=Anchors();parser.feed((REPORT/'report.html').read_text());parser.close()
    anchors=[link for link in parser.links if link['label'] in {'PDF','PNG'} or link['label'].endswith(' · SVG')]
    actual=Counter(link['href'] for link in anchors)
    failures=[]
    if Counter(canonical)!=Counter(expected):failures.append('canonical figure links differ from the complete catalog')
    if actual!=Counter(expected):failures.append('rendered figure hrefs differ from the complete catalog, are duplicated or were sanitized')
    checked=[]
    for link in anchors:
        href=link['href'] or ''
        parts=urlsplit(href)
        target=(REPORT/unquote(parts.path)).resolve()
        valid=(href in expected and not any([parts.scheme,parts.netloc,parts.query,parts.fragment])
               and target.parent==(REPORT/'figures').resolve() and target.is_file() and target.stat().st_size>0)
        if not valid:failures.append(f'invalid rendered figure link: {link}')
        checked.append(dict(**link,target=target.relative_to(ROOT).as_posix() if target.is_relative_to(ROOT) else str(target),
                            exists=target.is_file(),bytes=target.stat().st_size if target.is_file() else 0,passed=valid))
    result=dict(status='passed' if not failures else 'failed',html='workspace/reports/modeling_unified20_20260913_005/report.html',
                verified_at=datetime.now(timezone.utc).isoformat(),catalog_figures=len(notes['figure_ids']),
                expected_links=len(expected),rendered_links=len(anchors),all_href_targets_exist=not failures,
                scope='Parsed actual generated HTML anchor elements and resolved local targets; browser click interaction not exercised.',
                sanitizer_rule='Portable safeLinkTarget preserves ./ and ../ relative prefixes; bare figures/ is sanitized to #.',
                failures=failures,links=checked)
    write(REPORT/'build_link_validation.json',result)
    assert not failures,failures
    return result


def deliver():
    command=['node',str(DELIVER),'--input',str(REPORT/'artifact.json'),'--output',str(REPORT/'report.html')]
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    result=subprocess.run(command,cwd=ROOT,env=env,text=True,capture_output=True)
    (REPORT/'build_stdout.log').write_text(result.stdout)
    (REPORT/'build_stderr.log').write_text(result.stderr)
    try:receipt=json.loads(result.stdout.strip() or result.stderr.strip())
    except json.JSONDecodeError:
        receipt=dict(ok=False,stage='delivery_process',exit_code=result.returncode,error=result.stderr or result.stdout)
    write(REPORT/'build_receipt.json',receipt)
    if result.returncode or not receipt.get('ok'):
        print(json.dumps(receipt,ensure_ascii=False));raise SystemExit(result.returncode or 1)
    assert (REPORT/'report.html').is_file() and (REPORT/'report.html').stat().st_size>0
    link_validation=validate_paper_links()
    notes=read(REPORT/'build_notes.json')
    notes['delivery_verification']=receipt['stages']['verification']
    notes['verification_limit']='Browser chart extraction, desktop/narrow layout and source interaction were not exercised.' if receipt['stages']['verification']=='structural_only' else None
    notes['paper_links']=dict(status=link_validation['status'],checked=link_validation['rendered_links'],receipt='build_link_validation.json')
    write(REPORT/'build_notes.json',notes)
    if (REPORT/'build_content_validation.json').is_file():
        content=read(REPORT/'build_content_validation.json')
        content.update(canonical_generated_at=read(REPORT/'artifact.json')['manifest']['generatedAt'],
                       canonical_bytes=(REPORT/'artifact.json').stat().st_size,html_bytes=(REPORT/'report.html').stat().st_size,
                       official_verification=receipt['stages']['verification'],paper_links_checked=link_validation['rendered_links'],
                       paper_link_validation='passed')
        write(REPORT/'build_content_validation.json',content)
    print(json.dumps(receipt,ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deliver',action='store_true',help='Run the canonical portable packager and save its receipt.')
    args=parser.parse_args()
    previous=read(REPORT/'artifact.json') if (REPORT/'artifact.json').is_file() else None
    a,s,data=base_report();add_derived_sections(a);add_recording_section(a)
    a.md('limits','## 当前结论的适用范围与下一步\n\n'
        '这些结果来自此前分析后选定的三条采集记录；数据子集与部分结构容量曾参考旧test。新的训练种子提供了独立优化重复，但不会消除数据选择带来的局限。统计结果应解释为当前固定数据与时间留出上的模型比较。\n\n'
        '100 epoch是统一预算，不能单凭预算相同认定全部方法充分收敛。采集标签来自视觉分割与中心线，掩码宽度采用统一评价适配器；尚待补充独立尺度标定与真实执行任务。\n\n'
        '下一步最有区分力的工作是：采集控制加载路径和保持时长的独立轨迹，评估跨记录条件泛化，并分别测量连续预测与完整闭环执行。论文中真实控制任务和硬件待确认信息继续保留明确占位。','audit')
    a.md('further_questions','## 哪些后续证据可能改变当前结论\n\n'
         '在不同加载速率和保持时长下，路径与时间位移能否仍然互补？固定指数基之外是否存在可复现的通道差异？'
         '延长训练预算后，双记忆读出和窗口模型的排序是否保持？使用真实事件时间积分及长期流式初始化时，采样差异与缓存精度如何变化？'
         '这些问题需要新的轨迹、预算和执行测量来回答。')
    # Read the catalog at the end, after all evidence and sections are built.
    catalog=read(OUT/'figure_catalog.json')
    a.source('figure_catalog','论文图目录、选择规则及说明',OUT/'figure_catalog.json')
    a.md('figures','## 论文图与可复现记录')
    a.md('shape_example_selection','形态示例使用seed100，并在每条记录内按目标几何选择|x_tip|最大的帧，附逐节点误差。'
         '选择依据是目标侧向位移，不使用模型预测误差排序；该示例用于检查大侧向形态，不能代替全测试帧汇总。','figure_catalog')
    a.notes.append('Figure captions from the final catalog are authoritative; shape examples use target maximum |x_tip|, seed100, nodewise error.')
    # Use the same last-read catalog for links and recorded metadata.
    next(b for b in a.m['blocks'] if b['id']=='figures')['body']='## 论文图与可复现记录\n\n全套采用蓝、橙与中性色，并辅以符号/线型区分；具体对应关系见各图图例。报告中的原生图表和语义表格已嵌入HTML。论文SVG/PDF为可选链接，随同figures目录可一起携带。\n\n'+ '\n'.join(
        f'- [{r["id"]} · SVG](./figures/{r["id"]}.svg) · [PDF](./figures/{r["id"]}.pdf) · [PNG](./figures/{r["id"]}.png)' for r in catalog)
    next(b for b in a.m['blocks'] if b['id']=='figures')['sourceId']='figure_catalog'
    stage_presentation_queries(a)
    report_preflight(a,previous,catalog)
    a.save();print(str(REPORT/'artifact.json'),flush=True)
    if args.deliver:deliver()


if __name__=='__main__':main()
