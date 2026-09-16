#!/usr/bin/env python3
"""Source-backed HTML supplement: completed physical trials and full MLP reuse."""
from pathlib import Path
import json
import re
import shutil
import subprocess
import sys
import zipfile
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import build_unified20_report as shared

ROOT=shared.ROOT
MLP=ROOT/'workspace/runs/analysis/modeling_internal_mlp_single_memory_20260913_007'
REAL=ROOT/'workspace/runs/analysis/real_control_paper_20260913_007'
REPORT=ROOT/'workspace/reports/completed_control_and_mlp_20260913_007'
FIG=ROOT/'docs/icra2027/figures/completion007'
NAMES=dict(base='基础MLP',path='仅路径记忆',time='仅时间记忆',both='双记忆',static_capacity='多项式特征对照')
TARGETS=dict(G01='全身左弯A',G02='全身左弯B',G03='全身右弯',G06='末端左移')
shared.REPORT=REPORT
read,write=shared.read,shared.write

def pm(row,metric):
    return f"{row[metric]['mean']:.3f} ± {row[metric]['sd']:.3f}"

def build():
    mlp=read(MLP/'summary.json');models={r['variant']:r for r in mlp['models']}
    assert read(MLP/'validation.json')['status']=='pass'
    assert read(REAL/'mlp_independent_validation.json')['status']=='pass'
    physical=read(REAL/'execution_summary.json')
    tips=pd.read_csv(REAL/'visual/endpoint_measurements.csv')
    assert len(tips)==15 and not tips.status.str.contains('provisional|pending').any()
    raw=pd.read_csv(MLP/'raw_test.csv');executions=pd.read_csv(REAL/'completed_trials.csv')
    a=shared.Artifact();a.title='实机任务结果与MLP记忆复用实验'
    a.m.update(title=a.title,description='已执行的形状规划与实物遮挡反馈，以及基础、单记忆、双记忆MLP的20次重复结果。')
    for sid,label,path in [
        ('mlp','MLP全部配置：20次重复测试',MLP/'raw_test.csv'),
        ('mlp_summary','MLP测试均值与配对统计',MLP/'summary.json'),
        ('mlp_protocol','MLP训练与统计协议',MLP/'README.md'),
        ('mlp_qa','独立重算的MLP统计核验',REAL/'mlp_independent_validation.json'),
        ('physical','真实执行完整记录',REAL/'completed_trials.csv'),
        ('physical_summary','全部尝试与完整记录统计',REAL/'execution_summary.json'),
        ('execution_evidence','部署版本及实际反馈作用核验',REAL/'execution_evidence.md'),
        ('tips','原始末帧独立提取的可见末端',REAL/'visual/endpoint_measurements.csv'),
        ('tip_method','末端提取方法与测量核验',REAL/'visual/README.md'),
        ('draft','更新后的论文初稿',ROOT/'docs/icra2027/draft.md')]:
        a.source(sid,label,path)
    a.md('title','# '+a.title+'\n\n实验补充 · 骨架预测、真实执行与视觉到位测量')
    a.md('summary','## 两个单记忆均改善MLP，双记忆继续降低误差\n\n'
         '基础、仅路径、仅时间、双记忆的测试骨架误差分别为 **1.957、1.478、1.578、1.437 mm**。'
         '双记忆相对仅路径和仅时间分别改善 **0.041 mm、0.141 mm**，校正后均显著。'
         '\n\n真实实验已有 **15次完整执行**，覆盖无遮挡、实物遮挡、开环和视觉反馈。'
         '末帧图像给出实际可见末端误差；各目标的反馈收益存在差异。','mlp_summary')
    a.md('mlp_scope','## 隐藏层融合：四种配置使用同一训练与测试条件\n\n'
         '基础MLP以当前四维压力预测15节点骨架，隐藏层宽度为64。路径和时间记忆分别处理驱动历史，'
         '在第二隐藏层激活之前投影并与基础特征相加。单记忆配置只融合对应记忆，双记忆融合两者；'
         '主网络、驱动变换及投影共同训练。\n\n'
         '每种配置100轮、20次重复，学习率由验证集选择，使用同一批2,958个测试目标。'
         '下图每个点来自一次训练，分布反映固定数据条件下的训练随机性。','mlp_protocol')
    rows=[]
    for r in raw.itertuples():
        rows.append({'配置':NAMES[r.variant],'骨架误差':r.mean_node_mm,'末端误差':r.endpoint_mm,'重复':int(r.seed)})
    a.chart('mlp_distribution','MLP骨架误差分布','每种配置20次；小值表示骨架预测更准确。',rows,'配置','骨架误差',
            kind='boxPlot',source='mlp',intent='distribution',grain='每次独立训练的合并测试误差')
    a.md('mlp_result','仅路径记忆已经带来较大改善；双记忆在此基础上进一步降低骨架和末端误差。'
         '多项式特征对照的骨架误差仍为1.957 mm，说明当前压力的特征扩展在该设置下未获得相应收益。','mlp_summary')
    a.table('mlp_values','MLP记忆复用结果',[
        dict(model=NAMES[v],skeleton=pm(m,'mean_node_mm'),tip=pm(m,'endpoint_mm'),n=m['n']) for v,m in models.items()],
        [('model','配置','text'),('skeleton','骨架/mm：均值±SD','text'),('tip','末端/mm：均值±SD','text'),('n','重复数','number')],'mlp_summary')
    effects=[]
    for c in mlp['statistics']:
        effects.append(dict(comparison=NAMES[c['reference'].removeprefix('mlp_')]+' → '+NAMES[c['alternative'].removeprefix('mlp_')],
            delta=f"{c['mean_reference_minus_alternative_mm']:+.6f}",
            ci=f"[{c['bootstrap95_lower_mm']:.6f}, {c['bootstrap95_upper_mm']:.6f}]",
            p=f"{c['wilcoxon_holm_p']:.6g}",sign=f"{c['sign_test_holm_p']:.6g}",positive=f"{c['positive_pairs']}/20"))
    a.md('paired','### 双记忆相对单记忆的收益大小\n\n'
         '双记忆相对仅路径的平均改善为0.041 mm，95%区间[0.026, 0.056] mm，Holm校正p=0.000168；'
         '相对仅时间的改善为0.141 mm，区间[0.126, 0.157] mm，校正p=0.00000381。'
         '分别有16/20和20/20次训练支持这一方向，符号检验也支持改善。'
         '\n\n下表差值为箭头左侧误差减右侧误差。基础模型的四项扩展比较与双记忆对两个单记忆的比较分别校正。'
         '区间由20,000次配对重抽样获得，描述训练重复的不确定性。','mlp_qa')
    a.table('paired_values','配对改善与显著性',effects,[('comparison','比较','text'),('delta','改善/mm','text'),
        ('ci','95%区间/mm','text'),('p','Wilcoxon校正p','text'),('sign','符号检验校正p','text'),('positive','改善次数','text')],'mlp_summary')
    a.md('real_scope','## 真实形状规划与实物遮挡反馈已有完整执行\n\n'
         '实机采用冻结的部署模型评估候选压力序列，比较初始计划开环执行与图像反馈校正。'
         '黑色挡板遮住部分臂身，命令频率设为5 Hz。15次完整执行中，10次为全身形状目标、5次为末端目标；'
         '9次无遮挡、6次实物遮挡，8次启用反馈。\n\n'
         '总计22次尝试还包含6次因连续反馈超时触发的保护停止，以及1次末端反馈未确认的执行。'
         '完成次数表示执行记录完整，不等同于按预设精度阈值判定的成功次数。','physical_summary')
    visual=[]
    for r in tips.itertuples():
        visual.append({'条件':TARGETS[r.target_group]+(' · 遮挡' if r.occlusion=='occluded' else ' · 无遮挡'),
            '反馈':'视觉反馈' if r.correction=='closed' else '开环','末端误差':float(r.tip_error_px),'试验':f'T{int(r.trial):02d}'})
    a.md('tip_scope','### 从原始图像独立测量实际到位误差\n\n'
         '在每次完整执行的最后一张原始RGB图像中提取可见臂端端面中心，提取过程不读取目标或预测。'
         '随后将实际执行使用的目标末端投影到图像，计算二维欧氏距离，单位为像素。'
         '下图保留每次执行；相同目标下的不同遮挡与反馈条件分别显示。','tip_method')
    a.chart('tip_errors','各目标与观测条件下的末端误差','每点为一次完成执行的原始末帧测量。',visual,'条件','末端误差',
            kind='scatter',source='tips',color='反馈',unit='px',intent='comparison',grain='每次实际执行的可见末端')
    grouped=[]
    for group in ['G01','G02','G03','G06']:
        values={'target':TARGETS[group]}
        for name,occ,corr in [('clear_open','clear','open'),('clear_closed','clear','closed'),('occluded_open','occluded','open'),('occluded_closed','occluded','closed')]:
            subset=tips[(tips.target_group==group)&(tips.occlusion==occ)&(tips.correction==corr)]
            values[name]=' / '.join(f'{v:.1f}' for v in subset.tip_error_px) if len(subset) else '—'
        grouped.append(values)
    a.md('tip_interpretation','全身左弯A的反馈末端误差小于开环；全身右弯的开环与反馈误差接近，遮挡反馈未显示更低误差。'
         '无遮挡末端目标的反馈误差也低于开环，遮挡条件下则未显示改善。'
         '这些结果说明系统能够在实物遮挡下执行模型规划与视觉反馈，但当前有限试验尚未给出一致的控制精度收益。'
         '各次初始历史及反馈间隔存在差异，因此按目标展示描述结果，不把它们当作配对训练重复进行显著性检验。','tips')
    a.table('tip_values','实际末帧可见末端误差（px）',grouped,[('target','目标','text'),('clear_open','无遮挡·开环','text'),
        ('clear_closed','无遮挡·反馈','text'),('occluded_open','遮挡·开环','text'),('occluded_closed','遮挡·反馈','text')],'tips')
    a.md('timing','## 5 Hz命令执行与图像反馈分别计时\n\n'
         '完整执行的命令间隔中位数为203–204 ms，各次95%分位为218–219 ms。'
         '初始规划耗时0.813–2.515 s；反馈任务的中位耗时为125–203 ms，95%分位为166–315 ms。'
         '反馈按每1–3个命令周期触发，并存在未能按时提交的校正。因此，5 Hz描述命令频率，实际反馈更新取决于图像与求解耗时。','physical_summary')
    timing=[dict(试验=f'T{r.trial:02d}',命令间隔中位_ms=r.command_interval_median_ms,命令间隔p95_ms=r.command_interval_p95_ms,
                 反馈中位_ms=r.job_median_ms if np.isfinite(r.job_median_ms) else None,
                 反馈p95_ms=r.job_p95_ms if np.isfinite(r.job_p95_ms) else None) for r in executions.itertuples()]
    a.chart('command_timing','完整执行的命令间隔','各试验的中位数与95%分位。',timing,'试验','命令间隔中位_ms',source='physical',unit='ms',
        series=[dict(field='命令间隔中位_ms'),dict(field='命令间隔p95_ms')],grain='每次真实执行的命令间隔分位数')
    a.md('timing_reading','命令间隔在各次试验中较为接近，反馈耗时的变化更大。完整结果图同时列出两者，便于区分模型求值速度与系统闭环速度。','physical_summary')
    a.table('timing_table','真实执行计时',timing,[('试验','试验','text'),('命令间隔中位_ms','命令中位/ms','number'),
        ('命令间隔p95_ms','命令p95/ms','number'),('反馈中位_ms','反馈中位/ms','number'),('反馈p95_ms','反馈p95/ms','number')],'physical')
    a.md('limits','## 结果的适用范围与论文表述\n\n'
         'MLP四种记忆配置支持“记忆结构可在隐藏层复用、双记忆在该结构中进一步改善预测”的结论。'
         '这些检验针对固定且已被前期分析查看的数据划分，不能替代新采集数据的泛化检验。'
         '\n\n实机使用较早训练并冻结的部署模型，不能将当前20次建模误差直接作为该部署模型的控制精度。'
         '图像评价覆盖可见末端；挡板后的完整骨架没有独立真值。相机尺度缺少独立精度核验，因此保留像素单位。'
         '所有末帧均随最后命令采集，未额外等待稳定；T21在最后两个采样间仍移动约9.3 px，其7.2 px误差是末帧结果。'
         '\n\n后续若要量化闭环收益，应统一目标与初始驱动历史，固定反馈间隔，并在更多实机重复中评价实际到位误差。'
         '模型预测、视觉测量与控制时序之间的偏差是进一步分析的重点。','tip_method')
    figure_dir=REPORT/'figures/completion007';figure_dir.mkdir(parents=True,exist_ok=True)
    for p in FIG.glob('*'):
        if p.is_file():shutil.copyfile(p,figure_dir/p.name)
    links=[]
    for name,label in [('internal_memory_mlp','MLP记忆融合'),('real_control_endpoints','实机末帧与到位误差'),('real_control_timing','执行覆盖与反馈时序')]:
        assert (figure_dir/f'{name}.svg').is_file()
        links.append(f'- {label}：[SVG](./figures/completion007/{name}.svg) · [PDF](./figures/completion007/{name}.pdf) · [PNG](./figures/completion007/{name}.png)')
    a.md('deliverables','## 论文与结果图\n\n[查看更新后的论文初稿](./draft.md)\n\n'+'\n'.join(links),'draft')
    a.md('rendering_scope','图件已检查排版，HTML的数据与结构已核验。当前环境缺少浏览器运行时，交互及窄屏布局尚未实测。')
    shared.stage_presentation_queries(a)
    for source in a.m['sources']:
        if 'query' in source:
            source['query']['description']=source['query']['description'].replace('build_unified20_report.py','build_completed_control_and_mlp_report.py')
    a.save()
    write(REPORT/'source_notes.json',dict(delivery='portable html',audience='technical',chart_map=a.chart_map,
        structure=['summary','MLP definitions and evidence','physical endpoint measurements','execution timing','limits and next experiments'],
        rendering='Scientific figures inspected separately; portable report uses canonical chart renderer.',
        material_caveats=['fixed previously inspected test data','20 training repetitions differ from physical trials','image-only pixel endpoint measurement','unmatched initial histories','earlier deployed model']))
    draft=ROOT/'docs/icra2027/draft.md';shutil.copyfile(draft,REPORT/'draft.md')
    for href in re.findall(r'\]\(([^)]+)\)',draft.read_text()):
        if href.startswith('figures/'):
            source=draft.parent/href;assert source.is_file(),source
            dest=REPORT/href;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    result=subprocess.run(['node',str(shared.DELIVER),'--input',str(REPORT/'artifact.json'),'--output',str(REPORT/'report.html')],capture_output=True,text=True)
    (REPORT/'build_stdout.log').write_text(result.stdout);(REPORT/'build_stderr.log').write_text(result.stderr)
    receipt=json.loads(result.stdout.strip() or result.stderr.strip())
    write(REPORT/'build_receipt.json',receipt)
    assert result.returncode==0 and receipt.get('ok'),receipt
    with zipfile.ZipFile(REPORT/'report_bundle.zip','w',compression=zipfile.ZIP_DEFLATED) as bundle:
        for p in REPORT.rglob('*'):
            if p.is_file() and p.name!='report_bundle.zip':bundle.write(p,p.relative_to(REPORT))
    print(json.dumps(receipt,ensure_ascii=False))
    print(REPORT/'report.html')

if __name__=='__main__':build()
