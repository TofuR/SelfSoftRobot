#!/usr/bin/env python3
"""Post-hoc, input-defined history matching and frozen-readout interventions.

Uses the completed study's fixed test slices and all five saved repetitions.
No model fitting, test-dependent hyperparameter selection, or label changes.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.benchmarks.modeling_models import make_model

MODELS = ['hov', 'chen_direction', 'bezier_gru', 'mlp', 'hov_no_play',
          'hov_no_maxwell', 'hov_no_memory']
LABEL = dict(hov='HOV', chen_direction='Chen 方向网络', bezier_gru='Yu Bézier–GRU',
             mlp='静态 MLP', hov_no_play='重训：无路径记忆',
             hov_no_maxwell='重训：无时间记忆', hov_no_memory='重训：无记忆')
CATEGORY = dict(opposite='当前输入近似相同、方向不同',
                same_direction='当前输入及方向近似相同、历史不同',
                same_recent_two='最近两步输入近似相同、较早历史不同')
VARIANTS = {'full': '完整模型', 'zero_play': '路径项置零',
            'zero_time': '时间项置零', 'zero_both': '两类记忆置零',
            'shuffle_play': '路径项跨帧置换', 'shuffle_time': '时间项跨帧置换'}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        out = csv.DictWriter(stream, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)


def chart(id, title, rows, *, kind='bar', x='model', y='value', color=None,
          unit='mm', description=''):
    return dict(id=id, title=title, rows=rows, kind=kind, x=x, y=y,
                color=color, unit=unit, description=description)


def load_groups(study):
    groups = []
    for file in sorted((study/'data/test').glob('*.npz')):
        with np.load(file, allow_pickle=False) as data:
            actions = data['actions'].astype(np.float32)
            windows = np.stack([actions[t-19:t+1] for t in range(19, len(actions))])
            item = dict(group=file.stem, actions=actions, windows=windows,
                        target=data['positions'][19:].astype(np.float64),
                        frames=data['frame_ids'][19:], times=data['timestamps'][19:])
        item['predictions'] = {}
        for model in MODELS:
            saved = []
            for seed in range(5):
                p = study/'evaluations'/model/f'seed_{seed}'/f'{file.stem}_predictions.npz'
                with np.load(p, allow_pickle=False) as d:
                    assert np.array_equal(item['frames'], d['frame_ids'])
                    assert np.allclose(item['target'], d['target_mm'], atol=1e-5)
                    saved.append(d['prediction_mm'].astype(np.float64))
            item['predictions'][model] = np.stack(saved)
        groups.append(item)
    assert sum(len(g['windows']) for g in groups) == 2958
    return groups


def select_pairs(group, tolerance):
    w = group['windows']
    active = np.ptp(group['actions'], axis=0) > .01
    current = w[:, -1, active].astype(np.float64)*150
    # Matching uses pressure only; disjoint greedy selection uses current-input
    # distance and chronology, never any measured or predicted shape.
    pairs = cKDTree(current).query_pairs(tolerance, p=np.inf, output_type='ndarray')
    if not len(pairs):
        return {k: [] for k in CATEGORY}, {k: 0 for k in CATEGORY}
    pairs = pairs[np.abs(pairs[:, 1]-pairs[:, 0]) >= 20]
    candidates = {k: [] for k in CATEGORY}
    for i, j in pairs:
        hist_gap = float(np.sqrt(np.mean((w[i, :-1, active]-w[j, :-1, active])**2))*150)
        if hist_gap < 20:
            continue
        now_gap = float(np.max(np.abs(current[i]-current[j])))
        delta_i = (w[i, -1, active]-w[i, -2, active])*150
        delta_j = (w[j, -1, active]-w[j, -2, active])*150
        direction_i = np.where(np.abs(delta_i) >= .3, np.sign(delta_i), 0)
        direction_j = np.where(np.abs(delta_j) >= .3, np.sign(delta_j), 0)
        row = dict(i=int(i), j=int(j), current_gap_kpa=now_gap,
                   history_rms_kpa=hist_gap,
                   recent_two_gap_kpa=float(np.max(np.abs(w[i,-2:]-w[j,-2:]))*150),
                   time_gap_s=float(abs(group['times'][j]-group['times'][i])))
        if np.any(direction_i*direction_j < 0):
            candidates['opposite'].append(row)
        if np.array_equal(direction_i, direction_j):
            candidates['same_direction'].append(row)
        if row['recent_two_gap_kpa'] <= tolerance:
            candidates['same_recent_two'].append(row)
    selected = {}
    for key, rows in candidates.items():
        used, selected[key] = set(), []
        for row in sorted(rows, key=lambda r: (r['current_gap_kpa'], r['i'], r['j'])):
            if row['i'] not in used and row['j'] not in used:
                used.update((row['i'], row['j']))
                selected[key].append(row)
    return selected, {k: len(v) for k, v in candidates.items()}


def history_analysis(groups, output):
    detailed, coverage, seed_metrics, summaries, examples, profiles = [], [], [], [], [], []
    for tol in (1., 2., 5., 10.):
        selected = {}
        for group in groups:
            rows_by_category, counts = select_pairs(group, tol)
            for category, rows in rows_by_category.items():
                selected[group['group'], category] = rows
                measured = []
                for r in rows:
                    gap = np.linalg.norm(group['target'][r['i']]-group['target'][r['j']],axis=-1)
                    measured.append(float(gap.mean()))
                    detailed.append(dict(group=group['group'],tolerance_kpa=tol,category=category,
                        frame_i=int(group['frames'][r['i']]), frame_j=int(group['frames'][r['j']]),
                        **r, observed_shape_gap_mm=float(gap.mean())))
                coverage.append(dict(group=group['group'],tolerance_kpa=tol,category=category,
                    category_label=CATEGORY[category], candidates=counts[category], pairs=len(rows),
                    observed_shape_gap_mm=float(np.mean(measured)) if measured else None))
        for category in CATEGORY:
            total = sum(len(selected[g['group'], category]) for g in groups)
            if not total:
                continue
            for model in MODELS:
                per_seed = []
                for seed in range(5):
                    errors, observations = [], []
                    for g in groups:
                        for row in selected[g['group'], category]:
                            i,j=row['i'],row['j']
                            target_delta=g['target'][i]-g['target'][j]
                            pred=g['predictions'][model][seed]
                            pred_delta=pred[i]-pred[j]
                            errors.append(float(np.linalg.norm(pred_delta-target_delta,axis=-1).mean()))
                            observations.append(float(np.linalg.norm(target_delta,axis=-1).mean()))
                    value=float(np.mean(errors))
                    per_seed.append(value)
                    seed_metrics.append(dict(tolerance_kpa=tol,category=category,model=model,
                        model_label=LABEL[model],seed=seed,pairs=total,delta_error_mm=value,
                        observed_gap_mm=float(np.mean(observations))))
                summaries.append(dict(tolerance_kpa=tol,category=category,category_label=CATEGORY[category],
                    model=model,model_label=LABEL[model],pairs=total,
                    delta_error_mm=float(np.mean(per_seed)),seed_sd_mm=float(np.std(per_seed,ddof=1))))
        if tol==5.:
            for category in CATEGORY:
                for g in groups:
                    rows=selected[g['group'],category]
                    if not rows:
                        continue
                    # A chronological example, selected without consulting its errors.
                    row=min(rows,key=lambda r:(r['i'],r['j']))
                    i,j=row['i'],row['j']
                    for source in ['观测', 'hov', 'chen_direction', 'hov_no_memory']:
                        arr=g['target'] if source=='观测' else g['predictions'][source][0]
                        for node in range(15):
                            examples.append(dict(category=category, group=g['group'],node=node,
                                model='观测' if source=='观测' else LABEL[source],
                                lateral_delta_mm=float(arr[i,node,0]-arr[j,node,0]),
                                frame_i=int(g['frames'][i]),frame_j=int(g['frames'][j]),
                                pressure_gap_kpa=row['current_gap_kpa'],history_rms_kpa=row['history_rms_kpa']))
    charts=[]
    charts.append(chart('history_pair_coverage','不同匹配容差下的帧对数量',
        [dict(tolerance=tol,category=CATEGORY[category],
              pairs=sum(r['pairs'] for r in coverage if r['tolerance_kpa']==tol and r['category']==category),
              sequences=3)
         for tol in (1.,2.,5.,10.) for category in CATEGORY],kind='bar',x='tolerance',y='pairs',
         color='category',unit='帧对',description='每个序列和类别内以压力距离优先、不重复使用帧；两窗口至少间隔20帧'))
    for category in CATEGORY:
        rows=[r for r in summaries if r['tolerance_kpa']==5. and r['category']==category]
        if rows:
            charts.append(chart('history_delta_'+category,CATEGORY[category]+'：形态差异预测误差',
                rows,kind='horizontalBar',x='model_label',y='delta_error_mm',
                description=f'当前压力各通道差≤5kPa；{rows[0]["pairs"]}对；误差先按帧对合并，再对五seed平均'))
    sensitivity=[r for r in summaries if r['model'] in ('hov','chen_direction','hov_no_memory')
                 and r['category']=='same_direction']
    if sensitivity:
        charts.append(chart('history_tolerance_sensitivity','同方向历史匹配的容差敏感性',sensitivity,
            kind='bar',x='tolerance_kpa',y='delta_error_mm',color='model_label',
            description='容差改变会同时改变帧对数量与输入匹配程度；各阈值为同一数据的敏感性分析'))
    for category in ('opposite','same_direction'):
        applicable=sorted({r['group'] for r in examples if r['category']==category})
        if applicable:
            group=applicable[0]
            rows=[r for r in examples if r['category']==category and r['group']==group]
            charts.append(chart('history_example_'+category,'历史配对示例：沿臂横向形变差异',rows,
                kind='line',x='node',y='lateral_delta_mm',color='model',
                description=f'{group}中按时间选首个合格帧对；seed0；正负号表示两帧的横向坐标差'))
    write_csv(output/'matched_pairs.csv',detailed)
    write_csv(output/'matched_pair_coverage.csv',coverage)
    write_csv(output/'matched_pair_seed_metrics.csv',seed_metrics)
    return dict(coverage=coverage,summary=summaries,seed_metrics=seed_metrics,examples=examples,charts=charts)


def interventions(study, groups, output):
    seed_results,group_results,profile_rows,agreement=[],[],[],[]
    for seed in range(5):
        checkpoint=torch.load(study/'formal/hov'/f'seed_{seed}'/'best_eval_model.pt',
                              map_location='cpu',weights_only=True)
        model,_=make_model(checkpoint['model'],checkpoint['config'],
            normalization=(checkpoint['center'],checkpoint['scale']),geometry_config=checkpoint['geometry_config'])
        model.load_state_dict(checkpoint['state_dict']);model.eval()
        core=model.core
        center=np.asarray(checkpoint['center']);scale=checkpoint['scale']
        pooled={key:[] for key in VARIANTS}
        for g in groups:
            actions=torch.from_numpy(g['windows'])
            pi,time,base=[],[],[]
            with torch.inference_mode():
                for chunk in actions.split(256):
                    out=core(chunk)
                    pi.append(out['pi_generalized'])
                    time.append(out['maxwell_generalized'])
                    base.append(out['skeleton'])
                pi,time=torch.cat(pi),torch.cat(time)
                actual=torch.cat(base).numpy()*scale+center
                saved=g['predictions']['hov'][seed]
                difference=float(np.max(np.abs(actual-saved)))
                assert difference<.002,(seed,g['group'],difference)
                agreement.append(dict(seed=seed,group=g['group'],max_coordinate_difference_mm=difference))
                rng=np.random.default_rng(4500+seed)
                permutation=torch.as_tensor(rng.permutation(len(actions)))
                memory={'full':pi+time,'zero_play':time,'zero_time':pi,'zero_both':torch.zeros_like(pi),
                        'shuffle_play':pi[permutation]+time,'shuffle_time':pi+time[permutation]}
                saved_arrays={}
                for variant,features in memory.items():
                    prediction=core._decode_generalized(actions[:,-1],features).numpy()*scale+center
                    error=np.linalg.norm(prediction-g['target'],axis=-1).mean(-1)
                    pooled[variant].append(error)
                    group_results.append(dict(seed=seed,group=g['group'],variant=variant,
                        variant_label=VARIANTS[variant],frames=len(error),mean_node_mm=float(error.mean())))
                    saved_arrays[variant]=prediction.astype(np.float32)
                if seed==0:
                    np.savez_compressed(output/f'{g["group"]}_interventions_seed0.npz',
                        target_mm=g['target'],frame_ids=g['frames'],pi=pi.numpy(),time=time.numpy(),**saved_arrays)
                for branch,features in [('路径记忆',pi),('时间记忆',time)]:
                    for node,v in enumerate(torch.sqrt(features[:,:14].square().mean(0)).tolist(),1):
                        profile_rows.append(dict(seed=seed,group=g['group'],branch=branch,node=node,
                            rms_bend_deg=float(v*180/np.pi),frames=len(actions)))
            print('interventions',seed,g['group'],flush=True)
        for variant,arrays in pooled.items():
            values=np.concatenate(arrays)
            seed_results.append(dict(seed=seed,variant=variant,variant_label=VARIANTS[variant],
                frames=len(values),mean_node_mm=float(values.mean()),p95_frame_mm=float(np.quantile(values,.95))))
    summary=[]
    for variant,label in VARIANTS.items():
        values=[r['mean_node_mm'] for r in seed_results if r['variant']==variant]
        baseline=[r['mean_node_mm'] for r in seed_results if r['variant']=='full']
        summary.append(dict(variant=variant,variant_label=label,mean_node_mm=float(np.mean(values)),
            seed_sd_mm=float(np.std(values,ddof=1)),increase_mm=float(np.mean(np.asarray(values)-baseline))))
    profile=[]
    for branch in ('路径记忆','时间记忆'):
        for node in range(1,15):
            rows=[r for r in profile_rows if r['branch']==branch and r['node']==node]
            # Pool squared RMS by frame count, then average the five seeds.
            values=[np.sqrt(sum(r['rms_bend_deg']**2*r['frames'] for r in rows if r['seed']==seed)/2958)
                    for seed in range(5)]
            profile.append(dict(branch=branch,node=node,rms_bend_deg=float(np.mean(values)),
                                seed_sd_deg=float(np.std(values,ddof=1))))
    charts=[chart('frozen_branch_interventions','固定权重后的记忆项干预',summary,
                  kind='horizontalBar',x='variant_label',y='mean_node_mm',
                  description='同一完整模型、固定参考与读出；五seed全部2958帧；置换在各序列内进行'),
            chart('branch_geometry_profile','两类记忆对局部弯曲坐标的实际贡献',profile,
                  kind='line',x='node',y='rms_bend_deg',color='branch',unit='度',
                  description='测试帧上的局部弯曲修正RMS；节点顺序从基座至末端；幅值不等同预测收益')]
    write_csv(output/'intervention_seed_metrics.csv',seed_results)
    write_csv(output/'intervention_group_metrics.csv',group_results)
    write_csv(output/'branch_profiles.csv',profile_rows)
    write_json(output/'checkpoint_agreement.json',agreement)
    return dict(summary=summary,seed_metrics=seed_results,group_metrics=group_results,
                profile=profile,agreement=agreement,charts=charts)


def streaming_timing(study, groups):
    """Bounded CPU timing; cached forward prediction is not online learning."""
    previous_threads=torch.get_num_threads()
    torch.set_num_threads(1)
    checkpoint=torch.load(study/'formal/hov/seed_0/best_eval_model.pt',map_location='cpu',weights_only=True)
    model,_=make_model(checkpoint['model'],checkpoint['config'],
        normalization=(checkpoint['center'],checkpoint['scale']),geometry_config=checkpoint['geometry_config'])
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    windows=torch.from_numpy(groups[0]['windows'][:250])
    values={'重算20步历史窗口':[], '缓存状态递推一步':[]}
    with torch.inference_mode():
        state=model.core(windows[:1])['latent_z']
        for index in range(220):
            window=windows[index:index+1]
            start=time.perf_counter_ns();out=model.core(window);end=time.perf_counter_ns()
            start2=time.perf_counter_ns();step=model.core.step_state(window[:,-1],state);end2=time.perf_counter_ns()
            state=step['latent_z']
            if index>=20:
                values['重算20步历史窗口'].append((end-start)/1e6)
                values['缓存状态递推一步'].append((end2-start2)/1e6)
    torch.set_num_threads(previous_threads)
    rows=[dict(mode=mode,p50_ms=float(np.median(v)),p95_ms=float(np.quantile(v,.95)),samples=len(v))
          for mode,v in values.items()]
    return dict(rows=rows,device='CPU, one PyTorch thread',warmup=20,
        definition='B1，输入已在内存；含状态处理与骨架输出；不含相机、分割、通信、规划或参数更新。',
        limitation='与其他分析任务同机执行，受CPU调度影响；仅衡量这份实现的在线预测计算量。',
        source='scripts/experiments/analyze_modeling_history_mechanisms.py::streaming_timing')


def history_horizon(study,groups):
    rows=[]
    for seed in range(5):
        checkpoint=torch.load(study/'formal/hov'/f'seed_{seed}'/'best_eval_model.pt',map_location='cpu',weights_only=True)
        model,_=make_model(checkpoint['model'],checkpoint['config'],
            normalization=(checkpoint['center'],checkpoint['scale']),geometry_config=checkpoint['geometry_config'])
        model.load_state_dict(checkpoint['state_dict']);model.eval()
        with torch.inference_mode():
            for history in (2,3,5,10,20):
                errors,changes=[],[]
                for g in groups:
                    x=torch.from_numpy(g['windows'][:,-history:].copy())
                    prediction=np.concatenate([model(chunk).numpy() for chunk in x.split(256)])
                    prediction=prediction*checkpoint['scale']+np.asarray(checkpoint['center'])
                    errors.extend(np.linalg.norm(prediction-g['target'],axis=-1).mean(-1))
                    changes.extend(np.linalg.norm(prediction-g['predictions']['hov'][seed],axis=-1).mean(-1))
                rows.append(dict(seed=seed,history_steps=history,history_span_s=(history-1)*.2,
                    mean_node_mm=float(np.mean(errors)),prediction_change_mm=float(np.mean(changes)),frames=len(errors)))
    summary=[]
    for history in (2,3,5,10,20):
        selected=[r for r in rows if r['history_steps']==history]
        summary.append(dict(history_steps=history,history_span_s=(history-1)*.2,
            mean_node_mm=float(np.mean([r['mean_node_mm'] for r in selected])),
            seed_sd_mm=float(np.std([r['mean_node_mm'] for r in selected],ddof=1)),
            prediction_change_mm=float(np.mean([r['prediction_change_mm'] for r in selected]))))
    return dict(rows=rows,summary=summary,
        definition='保持H20训练所得权重和目标帧不变，仅截短可用输入前缀；按各窗口首输入初始化；步数包含当前输入。',
        limitation='同时改变可用历史及初始化位置，衡量有限窗口敏感性；不代表重新训练后的最优历史长度。',
        charts=[chart('history_horizon','固定模型对历史窗口截短的敏感性',summary,
                      kind='bar',x='history_steps',y='mean_node_mm',
                      description='H20权重冻结；每种长度覆盖同样2958个目标，五seed平均')])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'workspace/runs/training/modeling_three_seq_20260913_001928')
    parser.add_argument('--output',type=Path,default=ROOT/'workspace/runs/analysis/modeling_mechanisms_20260913_001/history')
    parser.add_argument('--report',type=Path,default=ROOT/'workspace/reports/modeling_mechanisms_20260913_001/history_mechanisms.json')
    parser.add_argument('--timing-only',action='store_true')
    parser.add_argument('--horizon-only',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    groups=load_groups(args.study)
    if args.horizon_only:
        result=json.loads(args.report.read_text())
        result['history_horizon']=history_horizon(args.study,groups)
        result['charts']=[r for r in result['charts'] if r['id']!='history_horizon']+result['history_horizon']['charts']
        write_json(args.report,result)
        print(result['history_horizon']['summary'])
        return
    if args.timing_only:
        result=json.loads(args.report.read_text())
        result['streaming_timing']=streaming_timing(args.study,groups)
        write_json(args.report,result)
        print(result['streaming_timing'])
        return
    history=history_analysis(groups,args.output)
    effects=interventions(args.study,groups,args.output)
    horizon=history_horizon(args.study,groups)
    result=dict(status='complete',history=history,interventions=effects,
        streaming_timing=streaming_timing(args.study,groups),
        history_horizon=horizon,
        charts=history['charts']+effects['charts']+horizon['charts'],
        definitions=[
            '所有数据为既有三序列6:2:2划分中的2958个测试目标；五个保存seed均纳入。',
            '形态差异预测误差：对每对帧计算预测坐标差与观测坐标差之差，再对15个节点的欧氏距离取均值。',
            '当前输入L∞差≤5kPa为主匹配条件，1/2/10kPa为敏感性条件；窗口至少相隔20帧，前19步驱动历史RMS差≥20kPa。',
            '每个序列、阈值及类别内，按当前压力距离贪心选不重复使用帧的配对；类别和阈值之间并不独立。',
            '方向由最后两步差分得到，绝对变化小于0.3kPa记为零。相同方向并不等同相同最近输入幅值。',
            '统计中的±为五个训练seed的样本标准差；配对帧来自单次采集，未把每帧视为独立实验重复。'],
        limitations=[
            '相似输入匹配仍存在压力残差；模型间使用相同帧对，较严格的最近两步匹配可能样本稀少。',
            '形状差异还可包含测量噪声和漂移；同日数据不能单独识别材料迟滞的物理来源。',
            '冻结权重下的置零/置换证明模型依赖该记忆项，不等同其状态唯一对应物理内部变量。',
            '原正式模型已在这些测试片段上报告过结果，本报告为冻结模型后的探索性机制分析。'],
        provenance=[str(args.study.relative_to(ROOT)),str(args.output.relative_to(ROOT)),
                    'scripts/experiments/analyze_modeling_history_mechanisms.py'])
    write_json(args.report,result)
    print('saved',args.report,flush=True)


if __name__=='__main__':
    main()
