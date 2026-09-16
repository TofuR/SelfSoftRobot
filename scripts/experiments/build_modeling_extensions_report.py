#!/usr/bin/env python3
"""Source-backed report for initialization and newly trained internal plugins."""
from pathlib import Path
import argparse
import json
from html.parser import HTMLParser
import os
import re
import shutil
import subprocess
import zipfile
import numpy as np
import pandas as pd

import build_unified20_report as shared

ROOT = shared.ROOT
OUT = ROOT/'workspace/runs/analysis/modeling_extensions_20260913_006'
REPORT = ROOT/'workspace/reports/modeling_extensions_20260913_006'
RUN = ROOT/'workspace/runs/training/modeling_internal_plugins_20260913_006'
OLD = ROOT/'workspace/runs/training/modeling_unified20_20260913_004'
OLD_ANALYSIS = ROOT/'workspace/runs/analysis/modeling_unified20_20260913_005'
shared.REPORT = REPORT
NAMES = dict(base='基础模型', path='＋路径记忆', time='＋时间记忆', both='＋双记忆', static_capacity='静态扩展（等参数）')
STAGES = dict(reference_prefit='参考预拟合', hov_joint_epoch0='完整初始化 · 0联合epoch')
read, write = shared.read, shared.write


def pm(item, metric):
    r=item[metric]
    return f"{r['mean']:.4f} ± {r['sd']:.4f}"


def stage_draft():
    """Keep the manuscript's relative figure references valid in the bundle."""
    origin=ROOT/'docs/icra2027/draft.md'
    shutil.copyfile(origin,REPORT/'draft.md')
    copied=[]
    for href in re.findall(r'\]\(([^)]+)\)',origin.read_text()):
        path=href.removeprefix('./')
        if not path.startswith('figures/'):continue
        source=origin.parent/path
        assert source.is_file(),source
        target=REPORT/path;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target);copied.append(path)
    write(REPORT/'draft_assets.json',dict(manuscript='draft.md',relative_figure_links=copied))


def focus_mlp_presentation(a):
    """Organize presentation; retain all completed families in the supplement."""
    blocks={b['id']:b for b in a.m['blocks']}
    blocks['answer']['body']=blocks['answer']['body'].rsplit('\n\n',1)[0]
    blocks['plugin_interpretation']['body']=blocks['plugin_interpretation']['body'].split('\n\n')[0]+(
        '\n\n补充的Koopman解码层接入在当前配置和预算下未带来改善，完整比较见下方补充材料。内部插件收益限定于所测试的MLP接口。')
    blocks['plugin_math']['body']=(
        '`m=[q;d]`共32维，`e=φθ(u)`为可训练驱动变换。MLP的D把32维记忆投影为64维，'
        '在第二隐藏层的Tanh之前与主网络相加。主网络、驱动变换和D共同训练，路径阈值与时间常数固定。\n\n'
        '同seed从相同基础权重和零投影开始，各epoch使用同一独立随机生成器产生批次顺序。复用的是记忆结构；各基础网络从头重新训练。')
    blocks['protocol_detail']['body']=(
        '正文MLP的3个配置先各用选型seed900001比较学习率0.001和0.003，各100 epoch；随后完成60次正式拟合。'
        'Adam、batch256、100 epoch、验证最佳骨架误差选检查点。静态容量对照用当前驱动的1–8次幂组成32维旁路，与双记忆参数量相同。'
        '补充的Koopman组另有5个配置、100次正式拟合；两组共16次选型、160次正式拟合。')
    blocks['plugin_results']['body']='## 4. MLP内部记忆分支的20次重复结果\n\n基础、双记忆和静态容量对照均使用本轮重新训练的模型，在同一2,958个测试目标上评价。'
    blocks['paired_scope']['body']='### 改善幅度与统计证据\n\n差值=同seed基础MLP误差−分支配置误差，正数表示改善。双侧精确Wilcoxon沿用原MLP两项Holm族，符号检验作敏感性分析；区间来自20,000次配对seed重抽样。补充Koopman的四项统计族保持独立。'
    blocks['capacity_scope']['body']='双记忆与静态扩展参数量相同。下面给出其描述性配对差及未校正95%区间，这项直接比较不属于预定的两项MLP检验。'
    blocks['limits']['body']=blocks['limits']['body'].replace(
        '2. 可训练记忆分支可以接入MLP隐藏层与Koopman读出层，需分别报告各分支、基础网络和容量对照的实际效果。',
        '2. 可训练记忆接入MLP隐藏层后显示预测收益；其他基础网络与接入位置的适用性需分别评价。')
    appendix=[dict(id='koopman_supplement',type='markdown',sourceId='internal',body=(
        '## 补充材料：Koopman解码层记忆接入\n\n'
        '基础模型骨架误差为1.6864 mm，路径、时间、双记忆和静态容量扩展分别为1.7200、1.7514、1.7305和1.7361 mm。'
        '四项比较沿用原Koopman Holm族，误差均升高。结果限定于本轮解码接入方式和100 epoch预算。'
        '双记忆与静态容量的直接差值区间包含零，不能据此分离退化原因。'))]
    def split_table(tid, predicate, aid, title):
        item=next(t for t in a.m['tables'] if t['id']==tid)
        rows=a.datasets[item['dataset']]
        selected=[r for r in rows if predicate(r)]
        a.datasets[item['dataset']]=[r for r in rows if not predicate(r)]
        assert selected and a.datasets[item['dataset']]
        extra=dict(item,id=aid,title=title,dataset=aid)
        a.m['tables'].append(extra);a.datasets[aid]=selected
        appendix.append(dict(id=aid+'_block',type='table',tableId=aid))
    split_table('architecture',lambda r:'Koopman' in r['模型'],'koopman_architecture','Koopman解码接入方式')
    split_table('internal_values',lambda r:r['骨干']=='KOOPMAN','koopman_values','Koopman完整测试结果：每配置20次')
    split_table('paired_values',lambda r:r['骨干']=='KOOPMAN','koopman_pairs','Koopman四项预定义配对统计')
    split_table('capacity_values',lambda r:r['骨干']=='KOOPMAN','koopman_capacity','Koopman等参数直接比较（描述性）')
    split_table('convergence_values',lambda r:r['模型'].startswith('koopman_'),'koopman_convergence','Koopman验证选择与后期改善')
    a.datasets['paired_distribution']=[r for r in a.datasets['paired_distribution'] if r['配置'].startswith('MLP')]
    for chart in a.m['charts']:
        if chart['id']=='paired_distribution':chart['title']='MLP每个seed中分支带来的改善'
    for table in a.m['tables']:
        if table['id']=='internal_values':table['title']='MLP内部插件的三种配置'
        if table['id']=='paired_values':table['title']='MLP两项预定义骨架比较'
        if table['id']=='architecture':table['title']='MLP两种记忆接入方式'
    moving={'koopman_distribution_block','koopman_reading'}
    appendix[1:1]=[b for b in a.m['blocks'] if b['id'] in moving]
    a.m['blocks']=[b for b in a.m['blocks'] if b['id'] not in moving]
    index=next(i for i,b in enumerate(a.m['blocks']) if b['id']=='figures')
    a.m['blocks'][index:index]=appendix


def create_report():
    assert (RUN/'TRAIN_VAL_COMPLETE.json').exists()
    assert (OUT/'internal_plugins/COMPLETE.json').exists()
    audit=read(OUT/'validation/validation.json')
    assert audit['checks_failed']==0 and audit['results']['status']=='independently_recomputed'
    init=read(OUT/'initialization/summary.json')
    ext=read(OUT/'internal_plugins/summary.json')
    old=read(OLD_ANALYSIS/'summary.json')
    oldmap={r['model']:r for r in old['models']}
    modelmap={r['model']:r for r in ext['models']}
    a=shared.Artifact()
    a.title='初始化与内部记忆分支：测试结果及实现解释'
    a.m.update(title=a.title, description='0联合epoch的真实测试评价、预拟合解释，以及MLP/Koopman内部记忆分支的20次重复。')
    sources=[('initialization','重新构建的初始化模型与两阶段测试',OUT/'initialization/summary.json'),
             ('init_validation','初始化验证门与保存预测复核',OUT/'initialization/validation.json'),
             ('internal','内部插件：全部20次重复汇总',OUT/'internal_plugins/summary.json'),
             ('internal_raw','内部插件：逐seed测试指标',RUN/'raw_test.csv'),
             ('internal_pairs','内部插件：配对统计',RUN/'paired_statistics.json'),
             ('internal_val','内部插件：验证选型与收敛记录',RUN/'raw_validation.csv'),
             ('internal_lr','内部插件：冻结的学习率选择',RUN/'selected_configuration.json'),
             ('protocol','训练前保存的内部插件协议',RUN/'protocol.json'),
             ('implementation','参考预拟合与既有插件的实现说明',REPORT/'implementation_explained.md'),
             ('module','可训练内部记忆模块源码快照',RUN/'source/src/benchmarks/modeling_memory_plugin.py'),
             ('old','此前统一20次重复的全部结果',OLD_ANALYSIS/'summary.json'),
             ('old_raw','此前统一20次重复的逐seed测试',OLD/'raw_test.csv'),
             ('efficiency','既有模型的验证曲线和CPU计时',OLD_ANALYSIS/'efficiency/summary.json')]
    for sid,label,path in sources:a.source(sid,label,path)
    a.md('title','# '+a.title+'\n\n5 Hz · 原时间留出划分 · 20次预定重复 · 骨架及末端误差')
    s0=next(r for r in init['stage_metrics'] if r['role']=='test' and r['stage']=='hov_joint_epoch0')
    sr=next(r for r in init['stage_metrics'] if r['role']=='test' and r['stage']=='reference_prefit')
    final=init['final_hov_summary']
    improvement=100*(s0['mean_node_mm']-final['mean_node_mm']['mean'])/s0['mean_node_mm']
    summary=[]
    for family in ['mlp','koopman']:
        base=modelmap[family+'_base'];both=modelmap[family+'_both']
        gain=100*(base['mean_node_mm']['mean']-both['mean_node_mm']['mean'])/base['mean_node_mm']['mean']
        direction='降低' if gain>=0 else '增加'
        summary.append(f"{family.upper()}内部接入双记忆后，骨架误差由{pm(base,'mean_node_mm')} mm变为{pm(both,'mean_node_mm')} mm，平均{direction}{abs(gain):.2f}%。")
    a.md('answer','## 初始化已获得较低误差，联合训练仍有明确收益\n\n'
         f"仅完成参考与记忆初始化的HOV，测试骨架误差为 **{s0['mean_node_mm']:.4f} mm**；完成联合训练后为 **{pm(final,'mean_node_mm')} mm**，进一步降低 **{improvement:.2f}%**。0联合epoch是在训练集拟合完成之后记录的阶段。\n\n"+
         '\n\n'.join(summary),'initialization')
    a.md('scope','## 评价范围与读取方式\n\n'
         '沿用三条记录内按时间6∶2∶2划分，再合并同一集合。有效训练、验证、测试目标分别为8,988、2,958、2,958；历史窗口20点。'
         '骨架误差先平均每帧15节点的欧氏距离，再合并全部测试帧，单位mm；表内±为20个训练seed的样本标准差。第一条记录贡献83.06%的测试帧。\n\n'
         '0联合epoch只重建一个确定性初始化基线；完整训练HOV来自原20次重复。内部插件为本次重新训练的8个配置，每个保留seed100–119全部20次。'
         '数据与划分此前已有分析，因此本轮评价当前固定数据上的补充结构效果。','protocol')

    a.md('init_title','## 1. 只初始化的结果如何\n\n'
         '从冻结源码与训练数据重新构建模型，先复现原验证值：参考预拟合2.045591 mm、完整初始化1.630999 mm。通过验证后，再评价同一2,958个测试目标。'
         '重建时没有使用联合训练后的权重。下图展示两个初始化阶段与完整训练结果；前两项各只有一个实际拟合解。','init_validation')
    stage_rows=[dict(阶段=STAGES[r['stage']],骨架误差=r['mean_node_mm'],末端误差=r['endpoint_mm'],独立拟合数=1,骨架SD=None,末端SD=None)
                for r in [sr,s0]]
    stage_rows.append(dict(阶段='联合训练后 · val选择',骨架误差=final['mean_node_mm']['mean'],末端误差=final['endpoint_mm']['mean'],独立拟合数=20,
                           骨架SD=final['mean_node_mm']['sd'],末端SD=final['endpoint_mm']['sd']))
    a.chart('init_stages','初始化与联合训练的测试误差','两项指标共享mm单位；完整训练的seed离散性见相邻表。',stage_rows,'阶段','骨架误差',source='initialization',
            series=[dict(field='骨架误差'),dict(field='末端误差')],grain='两个单解初始化阶段和20个最终模型的均值')
    stage_table=[dict(阶段=r['阶段'],拟合数=r['独立拟合数'],骨架=f"{r['骨架误差']:.4f}"+(f" ± {r['骨架SD']:.4f}" if r['骨架SD'] is not None else ''),
                     末端=f"{r['末端误差']:.4f}"+(f" ± {r['末端SD']:.4f}" if r['末端SD'] is not None else '')) for r in stage_rows]
    a.table('init_values','同一测试目标上的阶段结果',stage_table,[('阶段','拟合阶段','text'),('拟合数','独立模型数','number'),('骨架','骨架/mm','text'),('末端','末端/mm','text')],'initialization')
    a.md('init_interpretation',
         f"记忆读出初始化使参考骨架误差由{sr['mean_node_mm']:.4f}降至{s0['mean_node_mm']:.4f} mm，降低{100*(sr['mean_node_mm']-s0['mean_node_mm'])/sr['mean_node_mm']:.2f}%。"
         f"继续联合训练再降低{improvement:.2f}%。相对于固定初始化基线，20个完整训练模型全部改善，平均减少0.091737 mm，条件bootstrap 95%区间[0.088355, 0.094885] mm。"
         '这个区间仅刻画最终训练随机性的影响；没有把同一个初始化解当成20个独立训练结果。\n\n'
         '论文可表述为：**分阶段拟合在联合优化开始前已建立较准确的历史条件形态预测，联合优化进一步细化参考项与历史修正。** '
         '现有结果没有比较不同训练样本数量，因而还不能据此判断样本效率；同样，初始化已包含数据拟合，其成本需要计入。','initialization')

    a.md('prefit_title','## 2. 预拟合如何完成\n\n'
         '它先将一个较难的整体预测问题拆成“主要形态”和“历史修正”两个容易求初值的子问题。参考项学习当前压力能解释的平均趋势；记忆项学习同一压力趋势周围、由历史特征能够解释的残差。'
         '这套参考使用动态运动样本，不能直接解释为独立测得的静态物理平衡。','implementation')
    a.table('prefit_steps','从视觉骨架到可用初值',[
        dict(步骤='1 · 几何坐标',输入='8,988个训练窗末的4维压力、15节点骨架',计算='骨架切向差得到14个弯曲坐标；两段长度比取对数',产出='每个目标16维形状坐标'),
        dict(步骤='2 · 参考拟合',输入='当前压力与16维形状坐标',计算='线性岭初值；单调参考样条500步坐标拟合，再250步含几何损失拟合',产出='压力→参考形态；训练数据定义的几何基准'),
        dict(步骤='3 · 记忆递推',输入='同一训练集的20点动作窗口',计算='各窗口以首输入为数值起点，递推8维路径量q与24维时间量d',产出='8,988×32历史特征矩阵'),
        dict(步骤='4 · 残差岭拟合',输入='历史特征与目标减参考的16维残差',计算='按RMS缩放特征；带正则的32→16线性求解；写回记忆增益与方向',产出='完整初始化模型，即0联合epoch'),
        dict(步骤='5 · 联合优化',输入='上述初值、训练骨架与驱动窗口',计算='更新参考系数、通道耦合、记忆驱动与读出；val选择检查点',产出='本次论文中的完整HOV')],
        [('步骤','步骤','text'),('输入','使用什么','text'),('计算','做什么','text'),('产出','得到什么','text')],'implementation')
    a.md('prefit_math','### 岭拟合在这里的含义\n\n'
         '`残差 Y = 视觉形状坐标 − 参考形状坐标`\n\n'
         '`B = (ZᵀZ / n + λI)⁻¹ ZᵀY / n`，其中Z是按训练RMS缩放的历史特征，λ=0.001。\n\n'
         'B的每一列回答：为了修正一个弯曲或长度坐标，应该怎样组合32个历史特征。正则项抑制相关特征下过大的系数。'
         '实际求解还包含固定形状尺度，以及时间偏差的符号转换；这些保持线性解与模型前向一致。'
         '随后联合优化通过真实骨架位置误差，继续调整这些初值。\n\n'
         '预拟合实际包含750次全批量Adam更新，随后还有一次记忆岭求解。本次单进程重建中，参考/模型构建加记忆初始化耗时约7.72 s；'
         '该计时环境与原8任务并发训练不同，不能直接据此计算正式训练节省比例。','implementation')

    a.md('plugin_title','## 3. 之前的插件与现在的内部接入有何区别\n\n'
         '之前的MLP确实重新进行训练、验证和测试，但分支先生成固定输入特征，再拼接到MLP第一层。此次将记忆模块注册在预测网络中，'
         '记忆驱动变换、分支读出和基础网络一起学习。路径阈值与时间常数仍固定。','module')
    a.table('architecture','三种记忆使用方式',[
        dict(模型='已有：固定输入特征MLP',接入='[当前压力4；路径8；时间24] → 36→64→64→45',学习='重新训练MLP；记忆编码固定',参数='9,453'),
        dict(模型='新增：MLP隐藏层记忆',接入='h₁=tanh(L₁u)；h₂=tanh(L₂h₁ + Dm)；输出=L₃h₂',学习='基础MLP、D、记忆驱动变换共同学习',参数='9,473'),
        dict(模型='新增：Koopman解码记忆',接入='zₜ=A zₜ₋₁+B lift(uₜ)；输出=Czₜ+b+Dmₜ',学习='原潜状态模型、D、记忆驱动变换共同学习',参数='5,441')],
        [('模型','配置','text'),('接入','连接位置','text'),('学习','训练方式','text'),('参数','参数数','text')],'module')
    a.md('plugin_math','`m = [q; d]`共32维，`e = φθ(u)`为可训练驱动变换。路径量按play递推得到；时间量满足`dₜ=αdₜ₋₁−α(eₜ−eₜ₋₁)`，`α=exp(−Δt/τ)`。'
         'MLP的D把32维记忆映射到64维隐藏特征；Koopman的D把它映射到45维输出坐标。\n\n'
         '同一seed先构建相同的基础权重，再以D=0接入分支，因此各变体初始输出相同；各epoch使用同一独立随机生成器产生相同的批次顺序。'
         '每个配置从头独立优化，没有把训练好的HOV权重直接移植到基础模型。这里复用的是记忆结构。'
         'Koopman分支作用于读出，未改变线性潜状态A的递推定义。','module')
    a.md('protocol_detail','8个配置先各用独立选型seed900001比较学习率0.001和0.003，各100 epoch；随后冻结学习率，开展160次正式拟合。'
         'Adam、batch256、100 epoch、验证最佳骨架误差选检查点；所有正式训练与验证完成后才为本轮加载测试目标。'
         '静态容量对照把当前驱动的1–8次幂组成32维旁路，与双记忆具有相同的驱动变换参数和D矩阵大小。'
         '这种对照控制了参数数目，但不保证函数空间或数值条件完全相同。','protocol')

    a.md('plugin_results','## 4. 内部记忆分支的20次重复结果\n\n'
         '两种基础网络使用本轮重新训练的base作配对参照。以下所有配置都保留全部预定seed，误差先在2,958个共同目标上合并，再对seed汇总。','internal')
    a.md('plugin_interpretation','MLP双记忆在20个seed中全部改善，骨架误差平均减少0.5200 mm；等参数静态扩展的均值差接近零。'
         '这支持历史编码在当前MLP结构中的预测价值。该MLP的1.4372 mm也低于原HOV的1.4755 mm；两者分别使用9,473与760个参数或拟合系数，'
         '此处是跨两次实验的描述性数值对照。\n\n'
         'Koopman的路径、时间和双记忆版本则均比本轮base误差更高；双记忆的骨架误差增加0.0441 mm，Holm校正p=0.0107。'
         '等参数静态扩展也变差，且双记忆与静态扩展的直接差值区间包含零。当前结果未显示解码层旁路带来的额外预测收益。'
         '原潜状态已有历史、旁路与潜状态的联合优化方式等都可能影响结果，但本轮尚未分离这些原因。','internal')
    raw=pd.read_csv(RUN/'raw_test.csv')
    for family in ['mlp','koopman']:
        part=raw[raw.family==family].copy()
        part['配置']=part.variant.map(NAMES);part['骨架误差']=part.mean_node_mm
        a.chart(f'{family}_distribution',family.upper()+'内部接入的骨架误差分布','箱图展示20个训练seed；完整均值、标准差及末端误差见表。',
                part[['配置','骨架误差','seed','endpoint_mm','parameter_count']].to_dict('records'),'配置','骨架误差','boxPlot',source='internal_raw',intent='distribution')
        a.md(f'{family}_reading',summary[0 if family=='mlp' else 1], 'internal')
    rows=[dict(骨干=r['family'].upper(),配置=NAMES[r['variant']],骨架=pm(r,'mean_node_mm'),末端=pm(r,'endpoint_mm'),参数=r['parameters'],重复=r['n']) for r in ext['models']]
    a.table('internal_values','内部插件的全部配置',rows,[('骨干','基础网络','text'),('配置','配置','text'),('骨架','骨架/mm','text'),('末端','末端/mm','text'),('参数','参数数','number'),('重复','重复数','number')],'internal')
    statrows=[dict(骨干=r['family'].split('_')[0].upper(),配置=NAMES[r['alternative'].split('_',1)[1]],差值=r['mean_reference_minus_alternative_mm'],
                   下界=r['bootstrap95_lower_mm'],上界=r['bootstrap95_upper_mm'],改善seed=r['positive_pairs'],
                   p=shared.sci(r['wilcoxon_holm_p']),sign=shared.sci(r['sign_test_holm_p'])) for r in ext['statistics']]
    a.md('paired_scope','### 改善幅度与统计证据\n\n'
         '差值=同seed基础模型误差−分支配置误差，正数表示改善。双侧精确Wilcoxon采用实际秩的符号分配，MLP两项、Koopman四项分别Holm校正。'
         '同时给出双侧符号检验，便于检查结论是否依赖差值大小。置信区间来自20,000次配对seed重抽样，逐项报告。','internal_pairs')
    paired_rows=[]
    for r in ext['statistics']:
        family=r['family'].split('_')[0].upper()
        variant=NAMES[r['alternative'].split('_',1)[1]]
        paired_rows.extend(dict(配置=family+' '+variant,误差减少=float(delta),seed=seed)
                           for seed,delta in zip(range(100,120),r['seed_differences_mm']))
    a.chart('paired_distribution','每个seed中分支带来多少改善','正值：分支误差更低；负值：基础模型误差更低。每组20个配对差值。',
            paired_rows,'配置','误差减少','boxPlot',source='internal_pairs',intent='distribution',grain='基础网络内同seed的测试骨架误差差值')
    a.md('paired_chart_reading','配对差值把基础网络自身的训练波动纳入比较；盒体和须线描述20个差值的分布，下面的置信区间针对均值差。两者含义不同。','internal_pairs')
    a.table('paired_values','本轮6项预定义骨架比较',statrows,[('骨干','骨干','text'),('配置','对照base的配置','text'),('差值','平均改善/mm','number'),
            ('下界','95%CI下界','number'),('上界','95%CI上界','number'),('改善seed','改善seed数/20','number'),('p','Wilcoxon Holm p','text'),('sign','符号检验 Holm p','text')],'internal_pairs')
    capacity=[]
    boot=np.random.default_rng(20260913).integers(0,20,size=(20000,20))
    for family in ['mlp','koopman']:
        pivot=raw[raw.family==family].pivot(index='seed',columns='variant',values='mean_node_mm').sort_index()
        delta=(pivot.static_capacity-pivot.both).to_numpy();ci=np.quantile(delta[boot].mean(1),[.025,.975])
        capacity.append(dict(骨干=family.upper(),静态扩展减双记忆=float(delta.mean()),下界=float(ci[0]),上界=float(ci[1]),改善seed=int((delta>0).sum())))
    a.md('capacity_scope','等参数的双记忆与静态扩展也可直接比较。下面给出描述性配对差与未校正95%区间，未把这一额外比较混入预定义6项检验。'
         '若基础模型本身已经利用历史，额外显式记忆能否获益，仍需要结合分支种类和学习结果判断。','internal_raw')
    a.table('capacity_values','等参数旁路的直接差值（描述性）',capacity,[('骨干','骨干','text'),('静态扩展减双记忆','平均改善/mm','number'),('下界','95%CI下界','number'),('上界','95%CI上界','number'),('改善seed','双记忆较好seed数','number')],'internal_raw')

    val=pd.read_csv(RUN/'raw_validation.csv');chosen=read(RUN/'selected_configuration.json')['configurations'];convergence=[]
    for key,r in modelmap.items():
        v=val[val.model==key]
        convergence.append(dict(模型=key,学习率=chosen[key]['lr'],最佳epoch均值=float(v.best_epoch.mean()),epoch100最佳次数=int((v.best_epoch==100).sum()),
                                epoch80后改善次数=int((v.after80_best_gain_mm>0).sum()),epoch80后改善百分比=float(v.after80_best_gain_pct.mean())))
    a.md('convergence_scope','### 验证预算与收敛范围\n\n'
         '本轮按统一100 epoch预算完成；下表同时保留晚期验证改善，防止把预算终点误认为已证明完全收敛。'
         '0.001/0.003的两项学习率机会相同，各配置最终采用自己的验证选择。','internal_val')
    a.table('convergence_values','各配置的验证选择与后期改善',convergence,[('模型','配置','text'),('学习率','冻结LR','number'),('最佳epoch均值','最佳epoch均值','number'),
            ('epoch100最佳次数','最佳在100次数/20','number'),('epoch80后改善次数','80后改善次数/20','number'),('epoch80后改善百分比','80后改善/%','number')],'internal_val')

    a.md('old_plugin','## 5. 与此前固定输入插件结果如何衔接\n\n'
         '此前固定输入编码的MLP基线为1.9586±0.0203 mm，加入双记忆后为1.5105±0.0205 mm；'
         '新实验改变了接入位置、驱动是否可训练及批次随机生成方式，因此采用本轮base作比较。'
         '两个实验分别支持不同接口的使用效果，不能将同编号seed合并为40次重复。','old')
    oldrows=[dict(输入表示=NAMES[k],骨架=pm(oldmap[k],'mean_node_mm'),参数=oldmap[k]['stored_parameters']) for k in ['base','path','time','both','static_capacity']]
    a.table('old_input_values','已有固定输入特征MLP（独立实验）',oldrows,[('输入表示','表示','text'),('骨架','骨架/mm','text'),('参数','参数数','number')],'old')
    a.md('window_appendix','### 原完整对照的补充记录\n\n'
         '正文聚焦递推与紧凑形态模型，窗口MLP的已完成结果保留在补充比较中。它在同一原测试集上的骨架误差更低；'
         '这一点仍应在整体结论中保留。原主对照7项、消融3项、固定输入插件5项统计族维持原定义。','old')
    a.table('window_record','原窗口基线与HOV',[
        dict(模型=label,骨架=pm(oldmap[k],'mean_node_mm'),末端=pm(oldmap[k],'endpoint_mm'),参数=oldmap[k]['active_fitted_parameters'],重复=20)
        for k,label in [('hov','HOV'),('window','窗口MLP')]],
        [('模型','模型','text'),('骨架','骨架/mm','text'),('末端','末端/mm','text'),('参数','参数/拟合系数','number'),('重复','重复数','number')],'old')
    a.md('limits','## 可写入论文的结论与下一步\n\n'
         '1. 分阶段初始化已经给出较低的测试误差，后续联合训练仍能改善。把初始表现归于有监督拟合流程，有助于解释首epoch现象。\n'
         '2. 可训练记忆分支可以接入MLP隐藏层与Koopman读出层，需分别报告各分支、基础网络和容量对照的实际效果。\n'
         '3. 20次重复描述当前固定数据上的训练随机性；独立记录、不同训练样本量以及持续在线更新仍需要对应实验。\n\n'
         '新增模型的骨架和末端结果已评价，本轮未新增掩码与推理延迟测量。既有HOV缓存一步延迟p50约0.649 ms，只描述模型计算路径，实际闭环频率还需计入视觉、通信、状态校正和规划。', 'efficiency')
    if (OUT/'validation/validation.json').exists():
        a.source('audit','本轮独立统计与协议核验',OUT/'validation/validation.json')
        a.md('audit','本轮独立核验文件随结果保存，覆盖测试目标、配对seed、参数数目、精确检验和验证末段情况。','audit')
    figure_dir=REPORT/'figures'
    ids=sorted(p.stem for p in figure_dir.glob('*.svg'))
    assert ids, 'Generate paper figures before final report delivery.'
    links=[]
    for fid in ids:
        assert all((figure_dir/f'{fid}.{ext}').is_file() for ext in ['svg','pdf','png'])
        links.append(f'- [{fid} · SVG](./figures/{fid}.svg) · [PDF](./figures/{fid}.pdf) · [PNG](./figures/{fid}.png)')
    a.md('figures','## 结果图与实施说明\n\n'+'\n'.join(links)+'\n\n'
         '[预拟合与插件实现说明](./implementation_explained.md) · [小数值例](./prefit_numerical_example.md) · [论文草稿](./draft.md)')
    a.md('delivery_scope','结果图已作视觉检查；HTML采用统一报告组件，核验数据、结构与实际生成的文件链接。当前环境没有浏览器运行时，尚未实测浏览器中的交互与窄屏布局。')
    focus_mlp_presentation(a)
    shared.stage_presentation_queries(a)
    for source in a.m['sources']:
        if 'query' in source:
            source['query']['description']=source['query']['description'].replace('scripts/experiments/build_unified20_report.py','scripts/experiments/build_modeling_extensions_report.py')
    for kind in ['charts','tables']:
        for item in a.m[kind]:
            assert item['dataset'] in a.datasets and len(a.datasets[item['dataset']])>0
    assert len(a.datasets['internal_values'])==3 and len(a.datasets['koopman_values'])==5
    assert len(a.datasets['paired_values'])==2 and len(a.datasets['koopman_pairs'])==4
    assert len(raw)==160 and set(raw.seed)==set(range(100,120))
    assert raw.groupby('model').size().eq(20).all() and raw.test_frames.eq(2958).all()
    a.save()
    write(REPORT/'build_notes.json',dict(status='preflight_passed',counts=dict(charts=len(a.m['charts']),tables=len(a.m['tables']),sources=len(a.m['sources'])),
          figure_ids=ids,old_statistics_families_preserved=[7,3,5],new_statistics_families=[2,4],initialization_independent_fits=1,
          stochastic_fits=160,history=20,test_targets=2958,source_query_scope='Real SQLite presentation queries over Python-computed metrics; original lineage retained.',
          browser_policy='Canonical delivery structural verification; browser interactions not tested if runtime unavailable.'))
    stage_draft()


def deliver():
    cmd=['node',str(shared.DELIVER),'--input',str(REPORT/'artifact.json'),'--output',str(REPORT/'report.html')]
    result=subprocess.run(cmd,cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},text=True,capture_output=True)
    (REPORT/'build_stdout.log').write_text(result.stdout);(REPORT/'build_stderr.log').write_text(result.stderr)
    try:receipt=json.loads(result.stdout.strip() or result.stderr.strip())
    except json.JSONDecodeError:receipt=dict(ok=False,stderr=result.stderr,stdout=result.stdout)
    write(REPORT/'build_receipt.json',receipt)
    assert result.returncode==0 and receipt.get('ok'),receipt
    artifact=read(REPORT/'artifact.json')
    markdown=next(b['body'] for b in artifact['manifest']['blocks'] if b['id']=='figures')
    expected=re.findall(r'\]\((\./[^)]+)\)',markdown)
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            if tag=='a':self.links.append(dict(attrs).get('href'))
    parser=Links();parser.feed((REPORT/'report.html').read_text())
    assert len(expected)==len(set(expected))
    for href in expected:
        assert parser.links.count(href)==1, (href,parser.links.count(href))
        assert (REPORT/href).is_file(), href
    write(REPORT/'build_link_validation.json',dict(status='passed',links=expected,scope='Actual rendered HTML hrefs and local files; browser clicking not tested.'))
    notes=read(REPORT/'build_notes.json');notes['delivery_verification']=receipt['stages']['verification'];write(REPORT/'build_notes.json',notes)
    with zipfile.ZipFile(REPORT/'report_bundle.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(REPORT.rglob('*')):
            if path.is_file() and path.suffix.lower() in {'.html','.json','.md','.svg','.pdf','.png','.sqlite'}:
                archive.write(path,path.relative_to(REPORT))
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--deliver',action='store_true');args=parser.parse_args()
    create_report()
    if args.deliver:deliver()
