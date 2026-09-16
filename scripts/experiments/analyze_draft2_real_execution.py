#!/usr/bin/env python3
"""Read-only reconstruction of sparse camera endpoint trajectories for draft2 §3.6.

All generated files stay in the explicitly assigned real/ directory. No model is
loaded. Saved registration and endpoint measurements are reused unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d
from skimage.morphology import skeletonize

from analyze_real_control_image_endpoints import (
    measure_endpoint, sensitivity_configs, read_csv, read_json, write_csv, write_json,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'workspace/runs/analysis/draft2_experiment_extensions_20260914_009/real'
SOURCE = ROOT / 'workspace/runs/analysis/real_control_registered_mm_20260914_008/endpoint_measurements.csv'
BLUE, GOLD, INK = '#2873A6', '#B17A12', '#30363B'
# Image-only localizers refined after inspecting magnified raw cap/shaft crops.
# These are localization seeds and axes, not target-derived measurements.
RAW_REVIEW_LOCALIZERS = {
    (5,10): ((338,356),55.00798),
    (19,5): ((338,354),55.00798),
    (18,44): ((273,350),110.),
    (21,29): ((272,351),110.),
}


def rel(p):
    return str(Path(p).relative_to(ROOT))


def context():
    rows, _ = read_csv(SOURCE)
    for r in rows:
        r['trial'] = int(r['trial'])
        r['directory'] = (ROOT / r['raw_frame']).parents[2]
        samples, _ = read_csv(r['directory'] / 'samples.csv')
        r['samples'] = {int(s['step']): s for s in samples if int(s['camera']) == 0}
        r['t0'] = min(float(s['t_command']) for s in r['samples'].values())
        last=max(r['samples'])
        # Preserve all six already-reviewed points, adding early/middle/late frames.
        r['selected'] = sorted(set(np.rint(np.linspace(0,last,6)).astype(int).tolist()
                                   + np.rint(np.array([.1,.5,.9])*last).astype(int).tolist()))
        assert len(r['selected']) == 9
    return rows


def locate_tip(im):
    """Locate a thick silicone cap from the image, without a plan or target.

    The reviewed common distal ROI excludes the fixture. Opening removes tubes.
    A skeleton path gives only a localizer and outward axis, not the measurement.
    """
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    mask[290:407, 210:440] = (gray[290:407, 210:440] >= 150)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    candidates = [k for k in range(1, count) if stats[k, cv2.CC_STAT_AREA] > 300
                  and stats[k, cv2.CC_STAT_TOP] < 350]
    if not candidates:
        raise ValueError('No thick distal shaft in reviewed ROI')
    k = max(candidates, key=lambda j: stats[j, cv2.CC_STAT_AREA])
    mask = (labels == k).astype(np.uint8)
    sk = skeletonize(mask > 0)
    yy, xx = np.nonzero(sk)
    pts = set(zip(xx.tolist(), yy.tolist()))
    start = min(pts, key=lambda p: p[1])
    stack, previous = [start], {start: None}
    for p in stack:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                q = (p[0] + dx, p[1] + dy)
                if q in pts and q not in previous:
                    previous[q] = p
                    stack.append(q)
    end = stack[-1]
    path = [end]
    while previous[path[-1]] is not None:
        path.append(previous[path[-1]])
    if len(path) < 20:
        # A blocker can leave a short but directly visible vertical cap. Fit the
        # observed row midpoints, rather than extrapolating a hidden centerline.
        ys, xs = np.nonzero(mask)
        sections = []
        for y in range(int(ys.min())+3, int(ys.max())-5):
            x = np.flatnonzero(mask[y])
            if len(x) >= 14:
                sections.append((y, (x[0]+x[-1])/2))
        if len(sections) < 8:
            raise ValueError('Insufficient visible short cap')
        sections = np.asarray(sections)
        fit = np.polyfit(sections[:,0], sections[:,1], 1)
        normal = np.array([fit[0], 1.])
        seed = np.array([np.polyval(fit, ys.max()), float(ys.max())])
        angle = np.rad2deg(np.arctan2(normal[1], normal[0])-np.arctan2(.7,1))
        return seed, angle
    path = np.asarray(path, float)
    normal = path[2] - path[min(23, len(path)-1)]
    normal /= np.linalg.norm(normal)
    seed = path[0].copy()
    for distance in np.arange(0, 24, .25):
        q = path[0] + distance * normal
        x, y = np.rint(q).astype(int)
        if mask[y, x] == 0:
            seed = q - normal * .5
            break
    angle = np.rad2deg(np.arctan2(normal[1], normal[0]) - np.arctan2(.7, 1))
    return seed, angle


def reconstruct(rows):
    detail = []
    for r in rows:
        final = max(r['samples'])
        plan = np.load(r['directory'] / 'initial_plan.npz')
        matrix, goal = plan['camera_matrix'], plan['goal_mm'][-1]
        A, b = matrix[:2, :2], matrix[:2, 2]
        scale = np.linalg.norm(A[:, 0])
        assert np.allclose(A.T @ A, np.eye(2)*scale**2)
        assert np.allclose(goal @ A.T + b, [float(r['target_tip_x_px']), float(r['target_tip_y_px'])])
        for step in r['selected']:
            s = r['samples'][step]
            path = r['directory'] / s['raw_path']
            im = cv2.imread(str(path))
            d = dict(trial=r['trial'], name=r['name'], execution_id=r['execution_id'],
                     target_group=r['target_group'], target_kind=r['target_kind'],
                     occlusion=r['occlusion'], correction=r['correction'], step=step,
                     frame_timestamp=float(s['frame_timestamp']),
                     time_s=float(s['frame_timestamp'])-r['t0'],
                     raw_frame=rel(path), sample_source=rel(r['directory']/'samples.csv'),
                     after_command_ms=float(s['after_command_ms']),
                     fresh_after_command=s['fresh_after_command'], scale_px_per_mm=float(scale),
                     target_x_px=float(goal @ A[0] + b[0]), target_y_px=float(goal @ A[1] + b[1]),
                     target_x_mm=float(goal[0]), target_y_mm=float(goal[1]),
                     valid=False, is_final=step == final)
            try:
                if step == final:
                    point = np.array([float(r['measured_tip_x_px']), float(r['measured_tip_y_px'])])
                    d.update(method='existing_reviewed_final_endpoint',
                             sensitivity_min_mm=float(r['sensitivity_min_registered_mm']),
                             sensitivity_max_mm=float(r['sensitivity_max_registered_mm']))
                else:
                    seed, angle = locate_tip(im)
                    if (r['trial'],step) in RAW_REVIEW_LOCALIZERS:
                        auto_point,_,_=measure_endpoint(im,seed,1,angle_deg=angle)
                        d.update(provisional_auto_tip_x_px=float(auto_point[0]),
                                 provisional_auto_tip_y_px=float(auto_point[1]),
                                 quality_refinement='Raw crop review: local skeleton branch biased the cap axis; image-local seed/axis refined')
                        seed,angle=RAW_REVIEW_LOCALIZERS[(r['trial'],step)]
                        seed=np.asarray(seed,float)
                    point, _, diagnostics = measure_endpoint(im, seed, 1, angle_deg=angle)
                    variants = []
                    for config in sensitivity_configs():
                        config = dict(config)
                        config['angle_deg'] = angle + config.get('angle_deg', 0)
                        try:
                            alternative, _, _ = measure_endpoint(im, seed, 1, **config)
                            variants.append(float(np.linalg.norm(np.linalg.solve(A, alternative-b)-goal)))
                        except ValueError:
                            pass
                    d.update(method='image_silhouette_local_axis_and_cap', seed_x=float(seed[0]),
                             seed_y=float(seed[1]), axis_angle_deg=float(angle),
                             sensitivity_min_mm=min(variants) if variants else None,
                             sensitivity_max_mm=max(variants) if variants else None,
                             sensitivity_valid_variants=len(variants),
                             shaft_width_px=diagnostics['median_shaft_width_px'])
                observed = np.linalg.solve(A, point-b)
                error = float(np.linalg.norm(observed-goal))
                if d.get('sensitivity_min_mm') is not None:
                    d['sensitivity_min_mm']=min(error,d['sensitivity_min_mm'])
                    d['sensitivity_max_mm']=max(error,d['sensitivity_max_mm'])
                d.update(valid=True, tip_x_px=float(point[0]), tip_y_px=float(point[1]),
                         tip_x_mm=float(observed[0]), tip_y_mm=float(observed[1]), tip_error_mm=error)
                if step == final:
                    assert np.isclose(error, float(r['tip_error_registered_mm']), atol=1e-10)
            except ValueError as exc:
                d['failure_reason'] = str(exc)
            detail.append(d)
    write_csv(OUT/'detail.csv', detail)
    write_json(OUT/'measurements.json', detail)
    return detail


def review_sheets(rows, detail):
    for batch in range(3):
        selected = rows[batch*5:(batch+1)*5]
        canvas = Image.new('RGB', (9*255, len(selected)*156), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, r in enumerate(selected):
            for j, d in enumerate(x for x in detail if x['trial'] == r['trial']):
                im = Image.open(ROOT/d['raw_frame']).convert('RGB')
                dr = ImageDraw.Draw(im)
                if d['valid']:
                    x, y = d['tip_x_px'], d['tip_y_px']
                    dr.ellipse((x-4,y-4,x+4,y+4), outline='#00bfff', width=2)
                crop = im.crop((205, 275, 445, 410))
                canvas.paste(crop, (j*255, i*156+20))
                draw.text((j*255+3,i*156+3),f"T{r['trial']:02d} step {d['step']:02d}  {d['time_s']:.2f}s " + ('OK' if d['valid'] else 'FAIL'), fill='black')
        canvas.save(OUT/f'trajectory_review_{batch+1}.png')
    clear = [r for r in rows if r['target_kind']=='full' and r['occlusion']=='clear']
    fig, axes = plt.subplots(2, 4, figsize=(12, 9))
    for ax, r in zip(axes.flat, clear):
        im = Image.open(ROOT/r['raw_frame'])
        plan = np.load(r['directory']/'initial_plan.npz')
        target = plan['goal_mm'] @ plan['camera_matrix'][:2,:2].T + plan['camera_matrix'][:2,2]
        ax.imshow(im)
        ax.plot(target[:,0],target[:,1],'o--',color=GOLD,ms=2,lw=1)
        ax.set(xlim=(245,435),ylim=(382,18),title=f"T{r['trial']:02d}  {r['correction']}")
        ax.axis('off')
    for ax in list(axes.flat)[len(clear):]:
        ax.axis('off')
    fig.suptitle('Final clear-body frames and saved targets | fixed registration')
    fig.tight_layout()
    fig.savefig(OUT/'full_body_visibility_review.png',dpi=170)
    plt.close(fig)


def representative_review(rows, detail):
    # All nine stages of one clear and one occluded execution, with full shaft
    # context rather than the small localization ROI used by the estimator.
    selected=[d for trial in [1,21] for d in detail if d['trial']==trial]
    canvas=Image.new('RGB',(3*260,6*370),'white')
    draw=ImageDraw.Draw(canvas)
    for k,d in enumerate(selected):
        im=Image.open(ROOT/d['raw_frame']).convert('RGB')
        dr=ImageDraw.Draw(im)
        if d['valid']:
            x,y=d['tip_x_px'],d['tip_y_px']
            dr.ellipse((x-4,y-4,x+4,y+4),outline='#00bfff',width=2)
        x0,y0=(k%3)*260,(k//3)*370
        canvas.paste(im.crop((200,50,445,395)),(x0,y0+22))
        draw.text((x0+3,y0+4),f"T{d['trial']:02d} step {d['step']:02d} | {d['time_s']:.2f}s",fill='black')
    canvas.save(OUT/'representative_raw_review.png')


def left_a_figure(rows,body):
    points,_=read_csv(OUT/'visible_centerline_points.csv')
    by_trial={b['trial']:b for b in body}
    fig,axes=plt.subplots(1,3,figsize=(8,5.6))
    for ax,trial in zip(axes,[2,1,5]):
        r=next(v for v in rows if v['trial']==trial)
        b=by_trial[trial]
        im=Image.open(ROOT/r['raw_frame'])
        plan=np.load(r['directory']/'initial_plan.npz')
        target=plan['goal_mm']@plan['camera_matrix'][:2,:2].T+plan['camera_matrix'][:2,2]
        obs=np.array([[float(v['x_px']),float(v['y_px'])] for v in points if int(v['trial'])==trial])
        ax.imshow(im)
        ax.plot(target[:,0],target[:,1],'--',color=GOLD,lw=1.5)
        ax.plot(obs[:,0],obs[:,1],color=BLUE,lw=1.5)
        ax.axhline(70,color='white',ls=':',lw=.8)
        ax.set(xlim=(248,387),ylim=(372,22),title=f"T{trial:02d} {'open' if trial==2 else 'feedback'}\n{b['mean_mm']:.2f} mm")
        ax.axis('off')
    fig.suptitle('Full left A: final visible centerline and target',fontsize=14,y=.99)
    fig.text(.5,.04,'Blue: observed visible curve; gold: saved target; fixed registration.\nMean nearest-curve distance below y = 70 px; limited visible-body evidence.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.1,1,.94))
    fig.savefig(OUT/'left_a_visible_geometry.png',dpi=180)
    fig.savefig(OUT/'left_a_visible_geometry.svg')
    plt.close(fig)


def polyline_distance(points, curve):
    a, v = curve[:-1], np.diff(curve, axis=0)
    u = np.clip(np.sum((points[:,None]-a)*v,axis=2)/np.sum(v*v,axis=1),0,1)
    return np.linalg.norm(points[:,None]-(a+u[:,:,None]*v),axis=2).min(axis=1)


def visible_centerline(im, tip, threshold=150, kernel=9):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape,np.uint8)
    mask[70:395,240:440] = gray[70:395,240:440] >= threshold
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel,kernel)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask,8)
    k = max(range(1,n),key=lambda j:stats[j,cv2.CC_STAT_AREA])
    mask = (labels==k).astype(np.uint8)
    y,x = np.nonzero(skeletonize(mask>0))
    pts = set(zip(x.tolist(),y.tolist()))
    start = min(pts,key=lambda p:p[1])
    queue, prev = [start],{start:None}
    for p in queue:
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                q = (p[0]+dx,p[1]+dy)
                if q in pts and q not in prev:
                    prev[q]=p
                    queue.append(q)
    end = min(prev,key=lambda p:np.linalg.norm(np.asarray(p)-tip))
    if np.linalg.norm(np.asarray(end)-tip)>20 or start[1]>90:
        raise ValueError('Visible centerline does not connect the inspected shaft to cap')
    path=[end]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    path=np.asarray(path[::-1],float)
    path=gaussian_filter1d(path,2,axis=0)
    # Add measured cross-section center at the top of the common visible ROI.
    top_x=np.flatnonzero(mask[70])
    path=np.vstack(([np.mean(top_x),70.],path,tip))
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    grid=np.linspace(0,arc[-1],300)
    sampled=np.column_stack([np.interp(grid,arc,path[:,i]) for i in range(2)])
    return sampled,mask,float(arc[-1])


def body_analysis(rows):
    results,points=[],[]
    fig,axes=plt.subplots(2,4,figsize=(12,9))
    clear=[r for r in rows if r['target_kind']=='full' and r['occlusion']=='clear']
    for ax,r in zip(axes.flat,clear):
        im=cv2.imread(str(ROOT/r['raw_frame']))
        tip=np.array([float(r['measured_tip_x_px']),float(r['measured_tip_y_px'])])
        plan=np.load(r['directory']/'initial_plan.npz')
        A,b=plan['camera_matrix'][:2,:2],plan['camera_matrix'][:2,2]
        scale=float(np.linalg.norm(A[:,0]))
        goal=plan['goal_mm']@A.T+b
        observed,mask,length=visible_centerline(im,tip)
        distance=polyline_distance(observed,goal)/scale
        variants=[float(distance.mean())]
        for threshold,kernel in [(130,9),(170,9),(150,7),(150,11)]:
            v,_,_=visible_centerline(im,tip,threshold,kernel)
            variants.append(float(np.mean(polyline_distance(v,goal)/scale)))
        result=dict(trial=r['trial'],target_group=r['target_group'],correction=r['correction'],
                    raw_frame=r['raw_frame'],metric='mean_visible_centerline_to_target_polyline_distance',
                    visible_roi_y_min_px=70,visible_length_mm=length/scale,n_arclength_samples=300,
                    mean_mm=float(distance.mean()),p95_mm=float(np.percentile(distance,95)),
                    max_mm=float(distance.max()),validated_for_paper=r['trial'] in [1,2,5],
                    quality_issue=('Limited visible-curve evidence; no complete-node correspondence' if r['trial'] in [1,2,5]
                                   else 'Small differences not robust; boundary skeleton artifacts, especially T10/T11'),sensitivity_min_mm=min(variants),
                    sensitivity_max_mm=max(variants),full_body_error=False,
                    correspondence='nearest target segment; not material-node correspondence')
        results.append(result)
        for j,(p,e) in enumerate(zip(observed,distance)):
            points.append(dict(trial=r['trial'],sample=j,x_px=float(p[0]),y_px=float(p[1]),distance_mm=float(e)))
        cv2.imwrite(str(OUT/f't{r["trial"]:02d}_visible_body_mask.png'),mask*255)
        ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
        ax.plot(goal[:,0],goal[:,1],'--',color=GOLD,lw=1.4)
        ax.plot(observed[:,0],observed[:,1],color=BLUE,lw=1.4)
        ax.axhline(70,color='white',lw=.8,ls=':')
        ax.set(xlim=(245,435),ylim=(382,20),title=f"T{r['trial']:02d} {r['correction']} | {distance.mean():.2f} mm")
        ax.axis('off')
    ax=axes.flat[-1]
    ax.axis('off')
    ax.text(0, .8, 'Gold: saved target\nBlue: observed visible centerline\n\nMean nearest-curve distance\nOnly visible body below y = 70 px\nFixed camera registration\n\nNot a 15-node full-body error',va='top',fontsize=11)
    fig.suptitle('Visible centerline extraction | T01/T02/T05: limited evidence; other trials: diagnostic',fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT/'visible_body_geometry.png',dpi=180)
    fig.savefig(OUT/'visible_body_geometry.svg')
    plt.close(fig)
    write_csv(OUT/'visible_body_metrics.csv',results)
    write_csv(OUT/'visible_centerline_points.csv',points)
    return results


def feedback_events(rows):
    by_id={r['execution_id']:r for r in rows}
    source=ROOT/'Hereditary_workbench/real_validation/runs/hereditary/20260912_002251_76c5e1/events.jsonl'
    result=[]
    for line,text in enumerate(source.read_text(encoding='utf-8-sig').splitlines(),1):
        e=json.loads(text)
        r=by_id.get(e.get('execution_id'))
        if r is None or e.get('event') not in ['feedback','delayed_feedback_application'] or not e.get('state_committed'):
            continue
        result.append(dict(trial=r['trial'],execution_id=r['execution_id'],event=e['event'],
                           event_time_s=float(e['t'])-r['t0'],event_timestamp=e['t'],
                           source_frame_timestamp=e.get('source_frame_timestamp',e.get('frame_timestamp')),
                           source_step=e.get('source_step',e.get('step')),apply_step=e.get('apply_step',e.get('step')),
                           observer_accepted=e.get('observer',{}).get('accepted',False),
                           source=rel(source),source_line=line))
    # A delayed commit is repeated in the subsequent feedback event. Keep the
    # first actual commit record, not both log messages for the same update.
    unique={}
    for e in result:
        key=(e['execution_id'],e['apply_step'])
        if key not in unique:
            unique[key]=e
    result=list(unique.values())
    for e in result:
        r=by_id[e['execution_id']]
        e['after_command_sequence']=int(e['apply_step'])>max(r['samples'])
    motion=[e for e in result if not e['after_command_sequence']]
    assert len(motion)==68 and sum(e['observer_accepted'] for e in motion)==67
    write_csv(OUT/'visual_updates.csv',result)
    return result


def plot_trajectories(rows,detail,events,body):
    panels=[('G01','clear','Full left A | clear'),('G02','clear','Full left B | clear'),
            ('G02','occluded','Full left B | occluded'),('G03','clear','Full right | clear'),
            ('G03','occluded','Full right | occluded'),('G06','clear','Tip left | clear'),
            ('G06','occluded','Tip left | occluded')]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,4,figsize=(15.5,8.5))
    for ax,(group,occ,title) in zip(axes.flat,panels):
        selected=[r for r in rows if r['target_group']==group and r['occlusion']==occ]
        for k,r in enumerate(selected):
            d=[v for v in detail if v['trial']==r['trial']]
            valid=[v for v in d if v['valid']]
            color=BLUE if r['correction']=='closed' else GOLD
            marker=['o','s','^'][k]
            style='-' if k<2 else ':'
            x=[v['time_s'] for v in d]
            y=[v['tip_error_mm'] if v['valid'] else np.nan for v in d]
            ax.plot(x,y,style,marker=marker,color=color,lw=1.5,ms=4,
                    markerfacecolor=color if r['correction']=='closed' else 'white',
                    label=f"T{r['trial']:02d} {'feedback' if r['correction']=='closed' else 'open'}")
            lo=[v['sensitivity_min_mm'] for v in valid]
            hi=[v['sensitivity_max_mm'] for v in valid]
            ax.vlines([v['time_s'] for v in valid],lo,hi,color=color,lw=1.2)
            updated=[e['event_time_s'] for e in events if e['trial']==r['trial'] and e['observer_accepted'] and not e['after_command_sequence']]
            # Logs record commit time; rug marks do not imply a sampled response.
            ax.plot(updated,[.025+.025*k]*len(updated),'|',transform=ax.get_xaxis_transform(),color=color,ms=5)
        ax.set(title=title,xlabel='Time since first command (s)',ylabel='Observed tip error (mm)',
               xlim=(-.1,10.8),ylim=(0,52))
        ax.grid(axis='y',color='#e4e4e4',lw=.7)
        ax.legend(fontsize=8,loc='upper right',frameon=False)
    ax=axes.flat[-1]
    for i,r in enumerate(rows):
        value=float(r['tip_error_registered_mm'])
        ax.plot(value,i,'o',ms=4,color=BLUE if r['correction']=='closed' else GOLD)
        ax.text(value+.25,i,f'{value:.2f}',va='center',fontsize=8)
    ax.set(yticks=range(len(rows)),yticklabels=[f"T{r['trial']:02d}" for r in rows],
           xlabel='Final observed tip error (mm)',title='All 15 executions | final frames',xlim=(0,15))
    ax.tick_params(axis='y',labelsize=8)
    ax.invert_yaxis()
    ax.grid(axis='x',color='#e4e4e4',lw=.7)
    ax.set_axisbelow(True)
    fig.suptitle('Real-robot visual measurements',fontsize=18,y=.98)
    fig.text(.5,.94,'15 executions; nine selected RGB frames each; saved registration and nominal 16 mm scale',ha='center',fontsize=11)
    fig.text(.5,.025,'Points: observations; connectors: visual guides; bars: extraction sensitivity (not confidence intervals).\nRug ticks: 67 accepted visual state updates during command execution. Sparse samples cannot resolve individual update effects.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.07,1,.91),w_pad=2,h_pad=2)
    fig.savefig(OUT/'real_execution_analysis.png',dpi=180)
    fig.savefig(OUT/'real_execution_analysis.svg')
    plt.close(fig)


def report(rows,detail,events,body):
    motion=[e for e in events if not e['after_command_sequence']]
    trials=[]
    for r in rows:
        d=[v for v in detail if v['trial']==r['trial'] and v['valid']]
        trials.append(dict(trial=r['trial'],execution_id=r['execution_id'],target_group=r['target_group'],
                           correction=r['correction'],occlusion=r['occlusion'],n_observations=len(d),
                           first_observation_s=d[0]['time_s'],last_observation_s=d[-1]['time_s'],
                           first_error_mm=d[0]['tip_error_mm'],final_error_mm=d[-1]['tip_error_mm'],
                           sampled_min_error_mm=min(v['tip_error_mm'] for v in d),
                           accepted_updates=sum(e['trial']==r['trial'] and e['observer_accepted'] for e in motion)))
    full_status=dict(status='complete_corresponding_15_node_error_not_available',
                     n_clear_full_task_final_frames=7,n_occluded_full_task_final_frames=3,
                     reason=['No saved complete observed root-to-tip skeleton and node correspondences.',
                             'Root merges into fixture; visible-body measurement starts at image y=70.',
                             'Saved side-edge detection excludes root attachment and terminal segment.',
                             'Opaque blockers hide body regions in occluded executions.'])
    summary=dict(schema='draft2_real_execution_extensions_v1',n_trials=len(rows),
                 n_selected_frames=len(detail),n_valid_observations=sum(d['valid'] for d in detail),
                 all_existing_final_measurements_retained=True,final_endpoint_matches_source=True,
                 original_six_stage_observations_retained=True,
                 unit='registered planar mm inherited from nominal 16 mm outer diameter',
                 independent_physical_scale_validation=False,model_predictions_used_as_truth=False,
                 time_origin='first motion command per execution; camera frame_timestamp in samples.csv',
                 full_body_target_error=full_status,visible_body_alternative=body,
                 visible_body_trials_suitable_for_limited_discussion=[1,2,5],
                 visible_body_trials_not_supporting_improvement_claim=[7,8,10,11],
                 visual_state_commits_during_commands=len(motion),
                 observer_accepted_commits_during_commands=sum(e['observer_accepted'] for e in motion),
                 additional_post_sequence_commits=sum(e['after_command_sequence'] for e in events),
                 total_deduplicated_commits=len(events),trials=trials,source_endpoint_csv=rel(SOURCE),
                 extraction_review='Selected RGB contact sheets inspected by AI; four cap localizers refined from magnified raw crops; not blinded or independently annotated',
                 refined_localizers=[list(k) for k in RAW_REVIEW_LOCALIZERS],
                 sampling='Nine stages: six original uniformly spaced command indices plus 10/50/90 percent indices; no interpolated data',
                 sensitivity='Nominal plus successful extraction variants; parameter sensitivity, not confidence interval')
    write_json(OUT/'summary.json',summary)
    names={'G01':'全身左弯A','G02':'全身左弯B','G03':'全身右弯','G06':'末端左移'}
    lines=['# 实机执行补充分析：结果与3.6加入方式','',
           f'完成15次有效执行的9阶段抽帧，共{len(detail)}帧，得到{sum(d["valid"] for d in detail)}个实际末端观测。原6阶段点全部保留；15个末帧误差与008版逐点一致。',
           '', '**正文建议只补两项：左弯A的有限可见中心线证据，以及末端执行趋势。** 不报告完整15节点全身误差，也不将T07/T08、T10/T11的小差值解释为可靠改善。',
           '', '**实际末端轨迹结果。** 时间为相机时间戳减该次第一条运动命令时间；每点均为RGB轮廓测量。连接线仅提示阶段顺序。新增阶段位于原序列约10%、50%、90%，因此9点不是等时间间隔。',
           '', '| 执行 | 物理任务 | 条件 | 首个抽帧→末帧误差 (mm) | 末帧时间 (s) | 观测点 | 接受视觉校正 |',
           '|---|---|---|---:|---:|---:|---:|']
    for t in trials:
        lines.append(f"| T{t['trial']:02d} | {names[t['target_group']]} | {'无遮挡' if t['occlusion']=='clear' else '实物遮挡'}·{'反馈' if t['correction']=='closed' else '开环'} | {t['first_error_mm']:.2f}→{t['final_error_mm']:.2f} | {t['last_observation_s']:.3f} | {t['n_observations']} | {t['accepted_updates']} |")
    lines += ['', 'T01/T05是相同物理目标的不同反馈执行，反馈间隔分别为1/2条命令；T17/T22是相同物理目标的不同遮挡开环执行。全部保留并标注执行ID，不择优；这些执行条件并不完全相同，不据此估计重复实验方差。',
              '', '**可见臂身指标。** 7次无遮挡全身任务末帧采用固定公共ROI（原图y≥70 px），阈值分割厚硅胶体、骨架化和平滑，连接已测末端，按可见弧长取300点。计算每点到保存目标折线的最近距离，再取均值并通过原相似变换尺度换算mm；最终形状没有重新平移、旋转或缩放。300点是单张图像内的几何采样，不是300次实验。该指标描述可见曲线接近程度，不是逐材料点对应误差，也不评价完整根部或独立轴向伸长。',
              '', '| 执行 | 名义均值 (mm) | 参数敏感性范围，含名义配置 (mm) | 使用范围 |',
              '|---|---:|---:|---|']
    for b in body:
        use='左弯A：有限可见中心线证据' if b['trial'] in [1,2,5] else ('差值小，不支持可靠改善' if b['trial'] in [7,8] else '根部/末端分叉折角，仅供诊断')
        lines.append(f"| T{b['trial']:02d} | {b['mean_mm']:.3f} | {b['sensitivity_min_mm']:.3f}–{b['sensitivity_max_mm']:.3f} | {use} |")
    lines += ['', '敏感性范围已包括名义配置，以及阈值130/170、形态学核7/11的扰动；不是置信区间，未覆盖配准和物理标尺误差。T10/T11的骨架在近基座与末端出现分叉/折角，相关数值保留为诊断，不能用于可靠精度提升结论。左弯A的差异可作为本组执行的有限几何证据，不推广为一般反馈收益。',
              '', '**为何不能给出完整全身目标误差。** 7个无遮挡末帧的主要自由臂身可见，但根部轮廓与固定夹具相接，当前未形成独立的完整根部—末端节点对应。执行目录唯一NPZ为计划文件，其中reference_mm是参考预测、goal_mm是目标，不是实测骨架。既有掩码是末端局部掩码；反馈JSON保存的是部分侧边缘与prediction_px，后者是预测。检测代码明确跳过根部和终端段，coverage=1也只表示所搜索侧边缘段齐全；保存pixels未附全部segment对应。遮挡区域没有图像真值。',
              '', '**可直接加入3.6的论文段落（建议接表4分析之后）：**',
              '', '> 为补充末端终点指标，我们对无遮挡全身左弯A的最终图像提取可见自由臂身中心线，并在各次实验的固定相机配准下计算其到目标折线的平均距离。开环T02的可见中心线距离为3.94 mm，两次反馈执行T01和T05分别为2.82 mm和3.11 mm，表明在该组执行中可见臂身更接近目标。该指标评价可见曲线的几何偏差，尚不等同于具有完整节点对应的全身目标误差。',
              '', '> 进一步从15次有效执行各抽取9个阶段的RGB图像，依据可见硅胶末端轮廓重建实际末端目标误差随时间的变化。无遮挡全身左弯A中，反馈T01的误差由首个抽帧的36.80 mm降至末帧的5.08 mm，开环T02由37.51 mm降至8.44 mm；另一反馈执行T05最终为6.10 mm。无遮挡末端定位中，反馈T18和开环T19的最终误差分别为1.79 mm和5.39 mm。遮挡末端定位T21的最终误差为3.92 mm，反映出不同观测条件下的执行结果差异。图中同时标出日志确认接受的视觉状态校正时刻。抽帧结果描述了实际执行趋势，但其时间分辨率不足以解析单次校正效果。',
              '', '**图与加入方式。** `real_execution_analysis.png/.svg`前7子图按物理目标和遮挡条件分面，保留全部15条轨迹；第8子图汇总全部有效末帧。`left_a_visible_geometry.png/.svg`仅展示T01/T02/T05的可见几何证据，适合作为图6补充。`visible_body_geometry.png/.svg`保留7次全身任务的提取诊断，正文不采用后4次的细微优劣。',
              '', '建议图注：“实机执行的分阶段视觉测量。点为原始RGB图像提取的末端目标误差，连接线提示阶段顺序，竖线为提取参数敏感性范围；底部短划线表示执行期间接受的视觉状态校正时刻。末帧可见臂身距离采用固定配准下的最近曲线距离。”',
              '', '**证据与检查。**',
              '', '- `detail.csv`和`measurements.json`逐点保存执行ID、原图路径、时间戳、像素及mm坐标、定位参数、质量状态、敏感性范围。所有最终误差与008版一致，未剔除较差有效执行。',
              '- `visual_updates.csv`按执行ID和应用step去重，保留原始事件行号、源帧时间与实际提交时间。运动命令阶段68次提交、67次接受视觉修正；另外7次为命令序列结束后的提交，保留在CSV但不混入图中的执行期更新。T07有一次提交但观察器未接受。',
              '- `trajectory_review_1/2/3.png`提供135个抽帧点的末端定位叠图；代表轨迹另见`representative_raw_review.png`。图像测量函数不接收模型预测或目标，目标只参与测量完成后的误差计算。',
              '- 质检对T05/step10、T19/step5、T18/step44、T21/step29的局部骨架轴偏移进行了原图定位修正；原自动坐标保留在detail.csv的provisional_auto字段，纠正后的轮廓测量用于作图。不能将修正前的尖峰或回升当作真实运动证据。',
              '- `visible_body_metrics.csv`、`visible_centerline_points.csv`和各可见体掩码保留几何量复算依据。名义指标落在含名义配置的敏感性范围内。',
              '- 配准与标尺来源：008版README/summary和每次initial_plan.npz。原执行路径由008版endpoint_measurements.csv追溯；相机矩阵相似性、目标重投影、毫米距离与像素距离除尺度均已检查。',
              '- 侧边缘覆盖定义：`Hereditary_workbench/real_validation/perception/partial_edges.py`的extract_edges_vectorized循环从节点1开始并跳过最后两节点；`runtime/hereditary_deployment.py`的feedback函数说明检测由预测引导。',
              '', '**不能支持的解释与限制。**',
              '', '- 不支持全身15节点真值、遮挡后区域真值、反馈普遍改善、单调收敛、或每次更新的因果增益。9阶段点无法识别帧间峰值、精确到达时间或连续收敛速度。',
              '- 第一采样在首条命令之后；末帧是最后命令对应的新帧，不是额外持压稳定后的精度。T21末端仍可能运动。',
              '- mm沿用16 mm名义外径与平面假设，未增加独立尺规/NDI绝对尺度验证；提取敏感性不是测量置信区间。',
              '- 原图由AI复核，未做盲法人工重复标注；原部署版本与论文最新离线模型的区别参见007版execution_evidence.md，不将最新模型的预测误差当成本次控制误差。',
              '', '复算：`MPLCONFIGDIR=/tmp/draft2_real_mpl /Data5/ddf/environments/conda_envs/selfsr/bin/python -B scripts/experiments/analyze_draft2_real_execution.py`。仅写指定real目录，不修改主稿或原始数据。']
    (OUT/'findings.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'body_mean_mm':[(b['trial'],round(b['mean_mm'],3)) for b in body],
                      'accepted_motion_updates':67,'post_sequence_updates':7},ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['measure', 'report', 'all'], default='all')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = context()
    if args.stage in ['measure', 'all']:
        detail = reconstruct(rows)
        review_sheets(rows, detail)
        representative_review(rows,detail)
    else:
        detail = json.loads((OUT/'measurements.json').read_text())
    print(json.dumps(dict(measurements=len(detail), valid=sum(d['valid'] for d in detail),
                         failures=[(d['trial'],d['step'],d.get('failure_reason')) for d in detail if not d['valid']]),indent=2))
    if args.stage in ['report','all']:
        body=body_analysis(rows)
        events=feedback_events(rows)
        plot_trajectories(rows,detail,events,body)
        left_a_figure(rows,body)
        report(rows,detail,events,body)


if __name__ == '__main__':
    main()
