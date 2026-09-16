#!/usr/bin/env python3
"""Build the source-backed companion to the revised ICRA modeling manuscript."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.experiments.build_modeling_improvement_html import Report
from scripts.experiments.build_modeling_mechanisms_html import (
    source,add_chart,table,finish,save,load,custom_chart)

OUT=ROOT/'workspace/reports/modeling_paper_revision_20260913_002'


def rows(module,id):
    tables=module['tables']
    return tables[id] if isinstance(tables,dict) else next(t['rows'] for t in tables if t['id']==id)


def chart(module,id):
    return next(c for c in module['charts'] if c['id']==id)


def pick(data,fields):
    return [{f:r.get(f) for f in fields} for r in data]


def build():
    modules={k:load(OUT/(k+'.json')) for k in ['representation','efficiency','plugin']}
    p,e,r=(modules[k] for k in ['plugin','efficiency','representation'])
    stamp=datetime.now(timezone.utc).isoformat()
    sources=[]
    scripts={'representation':'analyze_modeling_representation.py','efficiency':'analyze_modeling_paper_efficiency.py',
             'plugin':'extend_modeling_plugin_repetitions.py'}
    labels={'representation':'记忆表示与几何传递','efficiency':'早期精度与计算预算','plugin':'插件重复与全身建模对照'}
    for k,m in modules.items():
        definitions=m.get('definitions',{})
        if isinstance(definitions,dict):definitions=[str(a)+': '+str(b) for a,b in definitions.items()]
        sources.append(source(k,labels[k],(OUT.relative_to(ROOT)/(k+'.json')).as_posix(),
                              'scripts/experiments/'+scripts[k],[str(x) for x in definitions],stamp))
    out=Report('历史记忆的表示、模块复用与学习效率',sources)
    both=next(x for x in p['plugin_summary'] if x['variant']=='both')
    base=next(x for x in p['plugin_summary'] if x['variant']=='base')
    con=next(x for x in p['plugin_contrasts'] if x['variant']=='both')
    out.md('intro',f'''# 历史记忆的表示、模块复用与学习效率

本报告对应论文初稿中的历史表示、插件复用与效率实验。主结果来自同一三序列划分，预测对照使用5个seed；插件在冻结配置下扩展为20个seed，完整使用全部结果。

双记忆MLP的平均骨架误差由{base['mean_node_mm_mean']:.3f}降至{both['mean_node_mm_mean']:.3f} mm，改善{con['improvement_pct']:.2f}%，Holm校正p={con['holm_p']:.3g}。新增的数学与数值分析进一步说明历史增量如何通过记忆读出形成沿臂位移。完整正文位于docs/icra2027/draft.md。''','plugin')
    out.md('main_story','''## 固定预算内的全身预测

三条5 Hz记录为172644、181044、181548，每条记录按时间6:2:2划分后合并，H20。各seed先池化全部2958个测试目标，再汇总训练重复。三条记录来自此前选定的运动子集，结果反映该子集内的时间留出预测。

主对照覆盖当前输入、完整窗口、方向特征与递推动力学。窗口MLP在验证网格内选择宽度和学习率；其批量为512，其他主要对照为256。各方法采用100 epoch预算及验证最佳检查点，比较的是这些具体配置与训练流程。''','plugin')
    m=p['main_comparison']
    custom_chart(out,'main_accuracy','八种模型的骨架误差',m,'label','mean_node_mm_mean','plugin','horizontalBar',unit='mm',x_label='模型',y_label='平均骨架误差（mm）')
    custom_chart(out,'main_capacity','模型规模与骨架精度',
                 [dict(label=x['label'],parameter_count=x['parameter_count'],error_mm=x['mean_node_mm_mean'],point_label=x['label']) for x in m],
                 'parameter_count','error_mm','plugin','scatter',unit='mm',x_label='拟合参数量',y_label='骨架误差（mm）')
    table(out,'main_table','同一全量测试集的五次结果',pick(m,['label','parameter_count','mean_node_mm_mean','mean_node_mm_sd','endpoint_mm_mean','mask_iou_mean','mask_dice_mean']),
          [('label','模型','text'),('parameter_count','参数量','number'),('mean_node_mm_mean','节点 mm','number'),
           ('mean_node_mm_sd','seed SD mm','number'),('endpoint_mm_mean','末端 mm','number'),('mask_iou_mean','IoU','number'),('mask_dice_mean','Dice','number')],'plugin')
    out.md('representation_story','''## 记忆保存什么，以及怎样进入几何预测

路径记忆可等价写为 q(t)=clip(q(t−1)+Δe(t),−r,r)：它存储输入增量的有界累积。由正饱和边界开始反转时，反向行程达到2r才到达另一边界。时间记忆满足 d(t)=α[d(t−1)−Δe(t)]：每次输入变化按各时间尺度持续衰减，因而等价于有符号增量的指数加权卷积。

下面统计有输入变化的测试窗口末步。r=0.02与r=0.5单元的内部累积区比例分别为26.48%和95.47%，边界裁剪区为72.79%和4.42%。两档阈值在当前轨迹中覆盖不同幅度的变化。这是实际模型的编码行为，更新恒等本身由结构保证。''','representation')
    c=dict(chart(r,'representation_play_regimes'));c['rows']=[x for x in c['rows'] if x['split']=='test']
    c.update(x_label='归一化驱动阈值 r',y_label='有效输入更新占比（%）')
    add_chart(out,c,'representation')
    out.md('time_kernel_story','''### 输入变化经过学习读出形成合成时间核

每个时间尺度的指数响应经几何读出组合，可得到从过去输入增量到当前形变的合成核。下图使用训练数据确定的参考输入和输入通道3，展示不同沿臂节点的横向响应。曲线单位是每单位归一化驱动增量引起的毫米位移，属于固定模型的局部响应。

时间基固定为0.600–2.000 s六尺度。状态之间具有相关性：训练24维时间记忆的99%方差子空间需要7维，并保留98.93%的测试能量；32维双记忆对应12维和99.32%。这里的子空间只解释状态分布，没有将降维后的模型重新训练或当作已辨识的材料谱。''','representation')
    c=dict(chart(r,'representation_time_kernel'));c.update(x_label='历史增量距当前的时间（s）',y_label='横向局部增益（mm / 归一化驱动单位）')
    c['rows']=[dict(x,node='节点'+str(x['node'])) for x in c['rows']]
    add_chart(out,c,'representation')
    c=dict(chart(r,'representation_state_spectrum'));c['rows']=[x for x in c['rows'] if x['family'] in ['双记忆32维','时间记忆24维']]
    c.update(x_label='训练主成分数',y_label='保留的状态能量（%）')
    add_chart(out,c,'representation')
    out.md('geometry_story','''### 显式几何解释沿臂的记忆位移

令 m=Wp·q+Wh·d，模型的记忆位移为 G(ξref+m)−G(ξref)。在参考形态处的一阶映射 Jgeo·m，将局部弯曲解释为累计角度造成的法向位移，将分段长度解释为切向伸缩。

全部测试窗口的模型记忆位移平均幅值为1.176±0.014 mm，一阶余项为0.028±0.001 mm，解释99.798%的位移能量。下图的位移来自固定模型，可用于解释内部历史量的作用；真实骨架准确性由前面的测试表评价。''','representation')
    c=dict(chart(r,'representation_geometry_profile'));c.update(y_label='节点位移/一阶余项（mm）')
    add_chart(out,c,'representation')
    out.md('plugin_story','''## 模块复用：20次固定重复及显著性

将当前输入与路径、时间特征拼接，接入普通线性映射或MLP并拟合骨架读出。基础输入4维，路径版本12维，时间版本28维，双记忆36维。静态扩展特征保持36维输入但只依赖当前压力；窗口模型直接使用80维输入。

在最初5个seed完成后，固定扩展到20个seed；这次新增15个seed×6种MLP配置，共90次训练。学习率、容量、100 epoch预算及验证选模沿用冻结配置。MLP的统计单位为训练seed，线性拟合是确定性的单次解。''','plugin')
    custom_chart(out,'plugin_accuracy','六种MLP输入表示的20次测试结果',p['plugin_summary'],'label','mean_node_mm_mean','plugin','horizontalBar',unit='mm',x_label='输入表示',y_label='骨架平均误差（mm）')
    ps=rows(p,'plugin_seeds');b={x['seed']:x['mean_node_mm'] for x in ps if x['variant']=='base'}
    gains=[dict(seed=x['seed'],label=x['label'],gain_mm=b[x['seed']]-x['mean_node_mm']) for x in ps if x['variant']!='base']
    custom_chart(out,'plugin_seed_gains','每个seed的配对改善',gains,'seed','gain_mm','plugin','scatter','label',unit='mm',y_label='基础误差 − 变体误差（mm）')
    out.md('plugin_statistics',f'''双记忆在20/20个seed中改善基础MLP，平均降低{con['improvement_mm']:.3f} mm，seed配对bootstrap的95%区间为[{con['bootstrap95_lower_mm']:.3f}, {con['bootstrap95_upper_mm']:.3f}] mm。双侧精确Wilcoxon检验在五项变体比较内作Holm校正。

窗口MLP使用更完整的显式输入历史，获得更低误差。双记忆的结果支持36维递推特征可改善基础映射，而不是所有历史表示中的精度最优。显著性只描述此固定数据与训练流程下的优化重复。''','plugin')
    statistics_rows=pick(p['plugin_contrasts'],['label','improvement_mm','improvement_pct','better_seeds','bootstrap95_lower_mm','bootstrap95_upper_mm','wilcoxon_exact_p','holm_p'])
    for item in statistics_rows:
        for field in ['wilcoxon_exact_p','holm_p']:item[field]=f'{item[field]:.6g}'
    table(out,'plugin_statistics_table','全部变体的配对效应与统计',statistics_rows,
          [('label','变体','text'),('improvement_mm','改善 mm','number'),('improvement_pct','改善 %','number'),('better_seeds','改善seed数','number'),
           ('bootstrap95_lower_mm','95%下界 mm','number'),('bootstrap95_upper_mm','95%上界 mm','number'),('wilcoxon_exact_p','原始p','text'),('holm_p','Holm p','text')],'plugin')
    linear=rows(p,'linear_descriptive')
    custom_chart(out,'linear_transfer','线性读出也可复用记忆特征',linear,'label','mean_node_mm','plugin','horizontalBar',unit='mm',x_label='输入表示')
    out.md('efficiency_story','''## 早期精度与完整拟合过程

HOV在参考形态与记忆读出初始化后，验证误差为1.631 mm，首个联合优化epoch为1.652 mm，验证最佳为1.509 mm。首轮到最佳的下降为8.53%，其他模型仍有57%–90%的下降。

这说明该结构与拟合流程能够在联合优化早期建立较准确的模型。初始化已使用完整训练集，并包含750步参考拟合；因此下面的epoch曲线比较各自流程的阶段进展，数据量效率与等计算成本效率需要相应实验。''','efficiency')
    for id in ['efficiency_hov_stages','efficiency_validation_curves','efficiency_relative_best_curve']:
        c=dict(chart(e,id));c.update(x_label='正式优化 epoch' if c['x']=='epoch' else '拟合阶段',
            y_label='验证骨架误差（mm）' if c['y']=='val_mm_mean' else '超出各自最佳的误差（%）')
        add_chart(out,c,'efficiency')
    out.md('inference_story','''### 缓存状态后的模型计算预算

同机Xeon Platinum 8336C CPU、单线程、批量1，五个seed各预热20次、计时200次。HOV缓存一步的p50/p95为0.648/0.701 ms，窗口重算为2.422/2.569 ms。缓存一步的平均p95占假设100 Hz周期的7.01%。

延迟包含输入处理、状态递推与骨架输出。测试存在同机任务；模型动力学步长仍为0.2 s。该结果表示高频框架中的模型求值预算，实际控制频率还需感知、通信、校正和规划的整体计时及闭环实验。''','efficiency')
    c=dict(chart(e,'efficiency_cpu_p95'));c.update(x_label='预测方式',y_label='p95延迟（ms）')
    add_chart(out,c,'efficiency')
    out.md('paper_use','''## 论文中的组织方式

方法部分给出有界增量累积、指数增量卷积及几何传递关系，再说明相同记忆接口可接入基础映射。实验以全身预测为起点，接续内部表示分析、插件复用、早期精度和递推效率。重训消融作为结构贡献的辅助结果，形状规划与部分观测反馈作为下游验证。

本轮完成的分析脚本、检查记录与逐seed结果均保存在同目录JSON及对应analysis运行目录。硬件规格、标定误差和实际下游任务的待补测量保留在初稿中。''','representation')
    artifact=finish(out,OUT,stamp)
    artifact['manifest']['description']='历史增量编码、显式几何传递、20次插件重复与学习及推理效率'
    save(OUT/'artifact.json',artifact)
    print(json.dumps(dict(charts=len(out.charts),tables=len(out.tables),artifact=str(OUT/'artifact.json')),ensure_ascii=False))


if __name__=='__main__':
    build()
