#!/usr/bin/env python3
"""Convert image-measured endpoints using each physical trial's saved registration."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'workspace/runs/analysis/real_control_paper_20260913_007/visual'
OUT=ROOT/'workspace/runs/analysis/real_control_registered_mm_20260914_008'

def main():
    source=pd.read_csv(SOURCE/'endpoint_measurements.csv')
    assert len(source)==15 and source.tip_valid.all()
    rows=[]
    for row in source.to_dict('records'):
        with np.load(ROOT/row['target_source'],allow_pickle=False) as plan:
            matrix=plan['camera_matrix'].astype(float)
            target=plan['goal_mm'][-1].astype(float)
        a,b=matrix[:2,:2],matrix[:2,2]
        scale=float(np.linalg.norm(a[:,0]))
        np.testing.assert_allclose(a.T@a,scale**2*np.eye(2),atol=1e-12,rtol=0)
        observed_px=np.array([row['measured_tip_x_px'],row['measured_tip_y_px']])
        target_px=np.array([row['target_tip_x_px'],row['target_tip_y_px']])
        observed=np.linalg.solve(a,observed_px-b)
        np.testing.assert_allclose(a@target+b,target_px,atol=1e-10,rtol=0)
        delta=observed-target
        error=float(np.linalg.norm(delta))
        np.testing.assert_allclose(error,row['tip_error_px']/scale,atol=1e-12,rtol=0)
        rows.append(dict(row,scale_px_per_registered_mm=scale,
            measured_tip_x_registered_mm=float(observed[0]),measured_tip_y_registered_mm=float(observed[1]),
            target_tip_x_registered_mm=float(target[0]),target_tip_y_registered_mm=float(target[1]),
            delta_x_registered_mm=float(delta[0]),delta_y_registered_mm=float(delta[1]),
            tip_error_registered_mm=error,
            sensitivity_min_registered_mm=row['tip_error_sensitivity_min_px']/scale,
            sensitivity_max_registered_mm=row['tip_error_sensitivity_max_px']/scale))
    result=pd.DataFrame(rows);OUT.mkdir(parents=True,exist_ok=True)
    result.to_csv(OUT/'endpoint_measurements.csv',index=False)
    summary=dict(n_trials=15,nominal_diameter_mm=16.0,training_pixels_per_mm=1.25,
        registration_scale_range_px_per_mm=[float(result.scale_px_per_registered_mm.min()),float(result.scale_px_per_registered_mm.max())],
        error_range_registered_mm=[float(result.tip_error_registered_mm.min()),float(result.tip_error_registered_mm.max())],
        source=str(SOURCE.relative_to(ROOT)/'endpoint_measurements.csv'),
        formula='x_observed = inverse(A) (p_observed - b); e = norm(x_observed - x_goal) = e_px / scale, A = scale R',
        unit_basis='Millimeters on the nominal 16 mm diameter scale inherited from training and transferred through each saved similarity registration',
        independent_physical_scale_validation=False,
        validation=dict(status='passed',similarity_matrices=15,target_reprojection=15,norm_matches_pixel_error_over_scale=15),
        sources=['docs/real_data/general_6ch_postprocess.md','docs/real_data/camera_pose_robot_frame.md',
                 'scripts/experiments/train_full_hereditary_deployment.py',
                 'workspace/runs/training/modeling_three_seq_20260913_001928/data/dataset_manifest.json',
                 'Hereditary_workbench/real_validation/runtime/hereditary_deployment.py'])
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    names=dict(G01='全身左弯A',G02='全身左弯B',G03='全身右弯',G06='末端左移')
    table=['| 目标 | 无遮挡·开环 | 无遮挡·反馈 | 实物遮挡·开环 | 实物遮挡·反馈 |','|---|---:|---:|---:|---:|']
    for group,name in names.items():
        values=[]
        for occ,corr in [('clear','open'),('clear','closed'),('occluded','open'),('occluded','closed')]:
            sub=result[(result.target_group==group)&(result.occlusion==occ)&(result.correction==corr)]
            values.append(' / '.join(f'{v:.1f}' for v in sub.tip_error_registered_mm) if len(sub) else '—')
        table.append('| '+name+' | '+' | '.join(values)+' |')
    (OUT/'paper_table.md').write_text('\n'.join(table)+'\n')
    (OUT/'README.md').write_text('''# 实机末端误差的毫米换算

评价对象是原始末帧独立提取的可见末端，与实际执行目标的距离。保留15次原始像素测量，不重新提取图像或改变终止帧。

对每次试验读取 initial_plan.npz 的 camera_matrix=[A,b]，以 inverse(A)(p-b) 得到实际末端在机器人平面坐标的位置。目标使用同一计划保存的 goal_mm[-1]。保存了实际位置、目标位置、二维有符号差值和欧氏距离。

矩阵是相似变换 A=sR，所以距离也等于像素距离除以s。两种算法逐试次一致，目标重投影与原图目标一致。

已有离线处理以名义16 mm外径与图像主体宽度建立尺度；早期示例为16/20.542≈0.778874 mm/px，当前合并训练统一为16/20=0.8 mm/px。实机相机位置改变后，通过初始中心线与模型的相似配准转移该尺度，本批s约1.816–1.828 px/mm。8 mm还作为几何轮廓半径，但反馈中的半径乘以配准尺度是向图像投影，不是独立反求尺度的测量。

因此表中毫米是沿用名义外径标尺的平面距离估计。其绝对尺度精度取决于外径参考、平面假设和初始配准；当前并未追加独立尺规或NDI目标坐标标定。旧日志的planned/estimated误差来自模型预测，本表来自原图测得位置，两者不能混用。

所有试次的最终位置是最后一条命令对应的保存帧。T21在最后两个采样间仍有运动；图像数据并未额外记录持压稳定后的精度。

复算：`/Data5/ddf/environments/conda_envs/selfsr/bin/python scripts/experiments/convert_real_control_endpoint_units.py`。
''')
    print(result[['trial','scale_px_per_registered_mm','tip_error_px','tip_error_registered_mm']].to_string(index=False))
    print('\n'.join(table))

if __name__=='__main__':main()
