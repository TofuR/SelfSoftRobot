#!/usr/bin/env python3
"""Pre-hardware sequential replay with fixed image occlusion and suffix feedback.

The hypothetical track issues revised pressures but receives prerecorded images,
as requested for integration debugging. A separate factual track always consumes
the recorded pressures for interpretable same-input prediction comparisons.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.control.hereditary_feedback import (
    ActionBounds, correct_state, correct_suffix, physical_shape, plan_initial, rollout, batched_rollout)
from src.control.hereditary_fast import FrozenHereditary, CachedSuffixA, fast_suffix_b
from src.control.partial_image import extract_edges, extract_edges_vectorized, project_camera
from src.evaluation.partial_replay_data import load_replay_data
from src.utils.model_loader import load_model


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")


def write_csv(path,rows):
    if not rows:
        return
    with Path(path).open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def select_motion_window(positions,first,length):
    """Select before feedback, solely by mean temporal node spread within dev."""
    if length<2 or first+length>len(positions):
        raise ValueError("requested window exceeds available development frames")
    candidates=[]
    for start in range(first,len(positions)-length+1):
        shape=positions[start:start+length,1:]
        score=float(np.sqrt(np.var(shape,axis=0).sum(-1)).mean())
        candidates.append({"start":start,"motion_spread_mm":score})
    chosen=max(candidates,key=lambda x:x["motion_spread_mm"])
    return chosen["start"],candidates


def fixed_occlusion(camera_positions,side,image_shape):
    """Choose mid/lower arm on first task frame once; never follow its motion."""
    node=int(round(0.65*(len(camera_positions)-1)))
    center=camera_positions[node]
    height,width=image_shape[:2]
    if side<1 or side>min(height,width):
        raise ValueError("invalid occlusion square size")
    x=int(np.clip(round(center[0]-side/2),0,width-side))
    y=int(np.clip(round(center[1]-side/2),0,height-side))
    return [x,y,side,side]


def hidden_nodes(camera,rectangle):
    x,y,w,h=rectangle
    return ((camera[...,0]>=x)&(camera[...,0]<x+w)&
            (camera[...,1]>=y)&(camera[...,1]<y+h))


def mean_error(prediction,reference,selection=None):
    error=np.linalg.norm(prediction-reference,axis=-1)
    if selection is not None:
        error=error[selection]
    return float(error.mean()) if error.size else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--data",required=True,help="development NPZ with split_manifest.json")
    parser.add_argument("--raw",required=True)
    parser.add_argument("--out",required=True,help="new, nonexisting run directory")
    parser.add_argument("--frames",type=int,default=80)
    parser.add_argument("--start",type=int,default=None,help="local NPZ index; default selects largest motion spread")
    parser.add_argument("--occluder-size",type=int,default=56)
    parser.add_argument("--occluder-bgr",type=int,nargs=3,default=[35,35,35])
    parser.add_argument("--measurement",choices=["edges","nodes"],default="edges")
    parser.add_argument("--oracle-visibility",action="store_true")
    parser.add_argument("--blocks",type=int,default=8)
    parser.add_argument("--initial-iterations",type=int,default=8)
    parser.add_argument("--prior-std",type=float,default=0.08)
    parser.add_argument("--state-max-delta",type=float,default=0.12)
    parser.add_argument("--search-px",type=int,default=12)
    parser.add_argument("--threads",type=int,default=1)
    parser.add_argument("--controller", choices=["torch_b", "batched_b", "fast_b", "cached_a"], default="fast_b")
    parser.add_argument("--detector", choices=["scalar", "vectorized"], default="vectorized")
    parser.add_argument("--cache-limit-mm", type=float, default=.5)
    args=parser.parse_args()
    if args.frames<2 or args.blocks<1 or args.initial_iterations<1 or args.threads<1:
        parser.error("frames>=2; blocks/iterations/threads must be positive")
    if args.prior_std<=0 or args.state_max_delta<=0 or args.search_px<1:
        parser.error("observer scales and search width must be positive")
    if not np.isfinite(args.cache_limit_mm) or args.cache_limit_mm<=0:
        parser.error("cache-limit-mm must be finite and positive")
    if any(c<0 or c>255 for c in args.occluder_bgr):
        parser.error("occluder color must be in [0,255]")
    output=Path(args.out)
    output.mkdir(parents=True,exist_ok=False)
    (output/"status.txt").write_text("running\n")
    try:
        cv2.setNumThreads(args.threads)
        with threadpool_limits(args.threads):
            run(args,output)
    except Exception:
        (output/"status.txt").write_text("failed\n")
        raise


def run(args,output):
    torch.set_num_threads(args.threads)
    torch.manual_seed(42);np.random.seed(42)
    info=load_model(args.checkpoint,device="cpu")
    if info["model_type"]!="hereditary_geometry":
        raise ValueError("expected hereditary geometry checkpoint")
    model=info["model"].eval();model.requires_grad_(False)
    cfg=info["saved_config"]
    norm=float(model.action_norm_factor)
    data=load_replay_data(args.data,args.raw,cfg,model_dt=float(model.dt),norm_factor=norm)
    selected,candidates=select_motion_window(data.positions,data.evaluation_start,args.frames)
    start=selected if args.start is None else args.start
    stop=start+args.frames
    if start<data.evaluation_start or stop>len(data.actions):
        raise ValueError("task window must lie wholly in development scoring rows")
    # State is propagated only from preceding actions. Future shape reference is
    # an explicitly prescribed offline task; the image detector never reads it.
    actions=torch.as_tensor(data.actions)
    z=model.init_z_from_action(actions[:1][None])[0]
    with torch.no_grad():
        for u in actions[:start]:
            z=model.step_state(u[None],z[None])["latent_z"][0]
    initial_state=z.detach().clone();previous=actions[start-1]
    reference=torch.as_tensor(data.positions[start:stop])
    channels=cfg["action_view"]["model_action_channels"]
    scale=data.scale_kpa*norm
    meta=data.meta
    bounds=ActionBounds(np.asarray(meta["lo6"])[channels]/scale,
                        np.asarray(meta["hi6"])[channels]/scale,
                        np.asarray(meta["rise_rates6"])[channels]*float(model.dt)/scale,
                        np.asarray(meta["fall_rates6"])[channels]*float(model.dt)/scale)
    lower=torch.tensor(bounds.lower,dtype=torch.float32)
    upper=torch.tensor(bounds.upper,dtype=torch.float32)
    first_image=cv2.imread(str(data.images[start]))
    if first_image is None:
        raise ValueError("cannot decode first image")
    rectangle=fixed_occlusion(data.camera_positions[start],args.occluder_size,first_image.shape)
    x,y,w,h=rectangle
    hidden_pixels=np.zeros(first_image.shape[:2],dtype=bool);hidden_pixels[y:y+h,x:x+w]=True
    visibility=~hidden_nodes(data.camera_positions[start:stop],rectangle)
    visibility[:,0]=False
    matrix=torch.tensor(data.model_to_camera)
    selected_score=next(r["motion_spread_mm"] for r in candidates if r["start"]==start)
    audit={**data.audit,"task_local_slice":[start,stop],
           "raw_frame_slice":[int(data.raw_indices[start]),int(data.raw_indices[stop-1])+1],
           "window_selection":"max mean temporal node spread" if args.start is None else "explicit start",
           "selected_motion_spread_mm":selected_score,
           "candidate_motion_spread_median_mm":float(np.median([r["motion_spread_mm"] for r in candidates])),
           "occluder_xywh":rectangle,"occluder_bgr":args.occluder_bgr,
           "occluder_policy":"fixed pixel square centered at first-frame mid/lower reference node",
           "occluder_follows_robot":False,
           "task_reference":"recorded future full shapes prescribed before planning; not withheld target generalization",
           "feedback_reference_access":args.measurement=="nodes",
           "feedback_visibility_access":args.oracle_visibility or args.measurement=="nodes",
           "scope":"counterfactual image-injection integration replay plus factual same-input prediction"}
    write_json(output/"data_audit.json",audit)
    write_csv(output/"pairing.csv",data.pairing[start:stop])
    write_csv(output/"window_candidates.csv",candidates)
    write_json(output/"config.json",vars(args))
    (output/"commands.sh").write_text(shlex.join([sys.executable,*sys.argv])+"\n")
    write_json(output/"run_manifest.json",{
        "schema":"partial_replay_run_v1","created_at":datetime.now(timezone.utc).isoformat(),
        "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "git_dirty":bool(subprocess.check_output(["git","status","--porcelain"],text=True).strip()),
        "checkpoint":str(Path(args.checkpoint).resolve()),"model_selection":cfg["phases"][0].get("validation_selection"),
        "artifacts":["data_audit.json","pairing.csv","steps.json","metrics.csv","traces.npz","report.html","overview.png"]})
    print(json.dumps({"selection":audit["raw_frame_slice"],"motion_mm":selected_score,"occluder":rectangle}),flush=True)
    print("Optimizing initial full trajectory from recorded-action seed...",flush=True)
    initial,planning_trace=plan_initial(model,initial_state,actions[start:stop],previous,reference,bounds,
                                        iterations=args.initial_iterations,blocks=args.blocks)
    write_json(output/"initial_planning.json",planning_trace)
    engine = FrozenHereditary(model) if args.controller in {"fast_b", "cached_a"} else None
    cache = (CachedSuffixA(engine, initial_state.numpy(), initial.numpy(), reference.numpy(),
                            blocks=args.blocks, max_model_discrepancy_mm=args.cache_limit_mm)
             if args.controller == "cached_a" else None)
    if cache is not None:
        write_json(output/"cache_precompute.json", {"time_ms": cache.precompute_ms,
                                                    "megabytes": cache.cache_bytes/1e6,
                                                    "stale_limit_mm": args.cache_limit_mm})
    detector = extract_edges if args.detector == "scalar" else extract_edges_vectorized
    with torch.no_grad():
        initial_prediction=rollout(model,initial_state,initial).numpy()
    work=initial.clone();z_sim=z.clone();z_factual=z.clone();z_open=z.clone()
    histories=[];metrics=[];panels=[]
    n=args.frames;nodes=model.n_nodes
    traces={name:np.full((n,n,nodes,2),np.nan,dtype=np.float32) for name in
            ("future_before_state","future_after_state","future_after_control")}
    traces.update({name:np.zeros((n,nodes,2),dtype=np.float32) for name in
                   ("sim_prior","sim_corrected","factual_prior","factual_corrected","factual_open")})
    traces["suffix_before"]=np.full((n,n,model.action_dim),np.nan,dtype=np.float32)
    traces["suffix_after"]=np.full_like(traces["suffix_before"],np.nan)
    traces["sim_state_before"]=np.zeros((n,model.operator_state_dim),np.float32)
    traces["sim_state_after"]=np.zeros_like(traces["sim_state_before"])
    traces["factual_state_before"]=np.zeros_like(traces["sim_state_before"])
    traces["factual_state_after"]=np.zeros_like(traces["sim_state_before"])
    issued=[];future_scores={1:[[],[]],5:[[],[]],10:[[],[]]}

    def update(state,action,image,local_index,frame_offset):
        prior=physical_shape(model,state,action).detach()
        predicted=project_camera(prior,matrix).numpy()
        edges=detector(image,predicted,radius=data.radius_px,search=args.search_px,
                            oracle_hidden=hidden_pixels if args.oracle_visibility else None)
        if not data.pairing[local_index]["pair_valid"]:
            return state,{"accepted":False,"reason":"invalid_time_pair","count":0,"delta_norm":0.0,"time_ms":0.0},edges
        if args.measurement=="edges":
            def residual(s):
                return edges.residual(project_camera(physical_shape(model,s,action),matrix),data.radius_px)
        else:
            visible=torch.tensor(visibility[frame_offset])
            target=reference[frame_offset,visible].clone()
            def residual(s):
                return ((physical_shape(model,s,action)[visible]-target)/1.0).reshape(-1)
        corrected,diagnostics=correct_state(model,state,action,residual,lower,upper,
                                           prior_std=args.prior_std,max_delta=args.state_max_delta)
        return corrected,diagnostics,edges

    for k,t in enumerate(range(start,stop)):
        frame_started=time.perf_counter()
        raw=cv2.imread(str(data.images[t]))
        if raw is None or raw.shape!=first_image.shape:
            raise ValueError("missing image or changing image dimensions")
        image=raw.copy();image[y:y+h,x:x+w]=np.asarray(args.occluder_bgr,np.uint8)
        issued_action=work[0].clone();remaining=work[1:].clone();issued.append(issued_action.numpy())
        with torch.no_grad():
            z_sim=model.step_state(issued_action[None],z_sim[None])["latent_z"][0]
            z_factual=model.step_state(actions[t:t+1],z_factual[None])["latent_z"][0]
            z_open=model.step_state(actions[t:t+1],z_open[None])["latent_z"][0]
            traces["sim_prior"][k]=physical_shape(model,z_sim,issued_action).numpy()
            traces["factual_prior"][k]=physical_shape(model,z_factual,actions[t]).numpy()
            traces["factual_open"][k]=physical_shape(model,z_open,actions[t]).numpy()
            traces["sim_state_before"][k]=z_sim.numpy();traces["factual_state_before"][k]=z_factual.numpy()
            if len(remaining):
                traces["future_before_state"][k,k+1:]=rollout(model,z_sim,remaining).numpy()
        factual_prior=z_factual.clone()
        z_factual,factual_info,_=update(z_factual,actions[t],image,t,k)
        z_sim,state_info,edges=update(z_sim,issued_action,image,t,k)
        with torch.no_grad():
            traces["sim_corrected"][k]=physical_shape(model,z_sim,issued_action).numpy()
            traces["factual_corrected"][k]=physical_shape(model,z_factual,actions[t]).numpy()
            traces["sim_state_after"][k]=z_sim.numpy();traces["factual_state_after"][k]=z_factual.numpy()
            for horizon in future_scores:
                if t+horizon>=stop:
                    continue
                same_actions=actions[t+1:t+horizon+1]
                prior_future=rollout(model,factual_prior,same_actions)[-1].numpy()
                corrected_future=rollout(model,z_factual,same_actions)[-1].numpy()
                future_scores[horizon][0].append(mean_error(prior_future[1:],data.positions[t+horizon,1:]))
                future_scores[horizon][1].append(mean_error(corrected_future[1:],data.positions[t+horizon,1:]))
            if len(remaining):
                traces["future_after_state"][k,k+1:]=rollout(model,z_sim,remaining).numpy()
        traces["suffix_before"][k,k+1:]=remaining.numpy()
        if len(remaining) and state_info["count"]>0:
            if args.controller == "fast_b":
                result,control_info=fast_suffix_b(engine,z_sim.numpy(),remaining.numpy(),issued_action.numpy(),
                                                  reference[k+1:].numpy(),bounds,blocks=args.blocks)
                work=torch.tensor(result,dtype=remaining.dtype)
            elif args.controller == "cached_a":
                result,control_info=cache.correct(k+1,z_sim.numpy(),remaining.numpy(),issued_action.numpy(),bounds)
                work=torch.tensor(result,dtype=remaining.dtype)
            else:
                work,control_info=correct_suffix(model,z_sim,remaining,issued_action,reference[k+1:],bounds,
                                                 blocks=args.blocks,rollout_fn=batched_rollout if args.controller=="batched_b" else rollout)
        else:
            work=remaining
            control_info={"accepted":False,"reason":"no_remaining_or_evidence","time_ms":0.0}
        traces["suffix_after"][k,k+1:]=work.numpy()
        with torch.no_grad():
            if len(work):
                traces["future_after_control"][k,k+1:]=rollout(model,z_sim,work).numpy()
        hidden=~visibility[k];hidden[0]=False
        row={"step":k,"raw_frame":int(data.raw_indices[t]),
             "factual_open_mm":mean_error(traces["factual_open"][k,1:],data.positions[t,1:]),
             "factual_prior_mm":mean_error(traces["factual_prior"][k,1:],data.positions[t,1:]),
             "factual_corrected_mm":mean_error(traces["factual_corrected"][k,1:],data.positions[t,1:]),
             "factual_hidden_prior_mm":mean_error(traces["factual_prior"][k],data.positions[t],hidden),
             "factual_hidden_corrected_mm":mean_error(traces["factual_corrected"][k],data.positions[t],hidden),
             "sim_image_discrepancy_prior_mm":mean_error(traces["sim_prior"][k,1:],data.positions[t,1:]),
             "sim_image_discrepancy_corrected_mm":mean_error(traces["sim_corrected"][k,1:],data.positions[t,1:]),
             "action_mismatch_kpa":float(np.max(abs((issued_action-actions[t]).numpy()*scale))),
             "edge_count":int(len(edges.pixels)),"state_accepted":state_info["accepted"],
             "suffix_accepted":control_info["accepted"],
             "factual_state_accepted":factual_info["accepted"],
             "state_ms":state_info["time_ms"],"suffix_ms":control_info["time_ms"]}
        histories.append({"step":k,"raw_frame":int(data.raw_indices[t]),
                          "issued_assumed_kpa":(issued_action.numpy()*scale).tolist(),
                          "recorded_kpa":(actions[t].numpy()*scale).tolist(),
                          "state_update":state_info,"factual_state_update":factual_info,
                          "suffix_update":control_info,
                          "edge_pixels":edges.pixels.tolist(),"edge_segments":edges.segments.tolist()})
        # JPEGs retain the original and the actual masked feedback image for inspection.
        panel=np.hstack([raw,image])
        for name,color in (("sim_prior",(0,165,255)),("sim_corrected",(0,255,0))):
            shape=project_camera(torch.tensor(traces[name][k]),matrix).numpy()
            shape[:,0]+=raw.shape[1]
            cv2.polylines(panel,[np.rint(shape).astype(np.int32)],False,color,2,cv2.LINE_AA)
        for q in edges.pixels:
            cv2.circle(panel,(int(round(q[0]))+raw.shape[1],int(round(q[1]))),2,(255,255,0),-1)
        cv2.putText(panel,f"Original frame {data.raw_indices[t]}",(15,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,220,255),2)
        cv2.putText(panel,"Masked feedback: orange prior / green corrected",(raw.shape[1]+10,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,220,255),1)
        ok,encoded=cv2.imencode('.jpg',panel,[cv2.IMWRITE_JPEG_QUALITY,80])
        if not ok:
            raise RuntimeError("image encoding failed")
        panels.append(encoded.tobytes())
        row["replay_step_ms"]=(time.perf_counter()-frame_started)*1000
        metrics.append(row)
        if k%10==0 or k==n-1:
            print(f"step {k+1}/{n}: edges={len(edges.pixels)}, state={state_info['accepted']}, suffix={control_info['accepted']}, mismatch={row['action_mismatch_kpa']:.2f} kPa",flush=True)
    assumed=np.asarray(issued)
    if not bounds.valid(assumed,previous.numpy()):
        raise AssertionError("issued sequence violates pressure/rate contract")
    traces.update({"initial_state":initial_state.numpy(),"initial_plan":initial.numpy(),
                   "initial_prediction":initial_prediction,"reference":reference.numpy(),
                   "recorded_actions":actions[start:stop].numpy(),"assumed_actions":assumed,
                   "visible_node_mask":visibility,"raw_indices":data.raw_indices[start:stop],
                   "action_unit_to_kpa":scale})
    np.savez_compressed(output/"traces.npz",**traces)
    write_json(output/"steps.json",histories);write_csv(output/"metrics.csv",metrics)
    def average(key):
        values=[r[key] for r in metrics if r[key] is not None]
        return float(np.mean(values)) if values else None
    summary={"schema":"partial_replay_summary_v1","frames":n,"raw_frames":audit["raw_frame_slice"],
             "controller":args.controller,"detector":args.detector,
             "measurement":args.measurement,"oracle_visibility":args.oracle_visibility,
             "occluder_xywh":rectangle,"motion_spread_mm":selected_score,
             "factual_same_input":{key:average(key) for key in
                                    ("factual_open_mm","factual_prior_mm","factual_corrected_mm",
                                     "factual_hidden_prior_mm","factual_hidden_corrected_mm")},
             "factual_future":{str(h):{"samples":len(v[0]),"before_mm":float(np.mean(v[0])) if v[0] else None,
                                       "after_mm":float(np.mean(v[1])) if v[1] else None} for h,v in future_scores.items()},
             "hypothetical_injection":{
                 "state_updates":sum(r["state_accepted"] for r in metrics),
                 "suffix_updates":sum(r["suffix_accepted"] for r in metrics),
                 "action_mismatch_mean_kpa":average("action_mismatch_kpa"),
                 "action_mismatch_max_kpa":max(r["action_mismatch_kpa"] for r in metrics),
                 "mean_edge_count":average("edge_count"),"pressure_contract_passed":True,
                 "suffix_time_percentiles_ms":np.percentile([r["suffix_ms"] for r in metrics[:-1]],[50,95,99,100]).tolist()},
             "limitations":["Revised actions did not generate the injected recorded images; no physical closed-loop accuracy claim.",
                            "Factual scores use offline SAM2 pseudo-reference and a within-sequence development window.",
                            "Original sequence-derived coordinate transform and fixed model time grid are replay approximations.",
                            "Image detector is a local white-side-edge prototype; no global relocalization or hardware deadline guarantee.",
                            "A and accelerated B are offline controllers; asynchronous execution and physical deployment remain pending.",
                            "replay_step_ms includes factual scoring and visualization; use benchmark_partial_feedback.py for control-path timing."]}
    write_json(output/"summary.json",summary)
    pressure_rows=[]
    for k in range(n):
        r={"step":k,"raw_frame":int(data.raw_indices[start+k])}
        for label,values in (("recorded",actions[start+k].numpy()*scale),
                             ("initial",initial[k].numpy()*scale),("assumed",assumed[k]*scale)):
            for c,v in enumerate(values[data.expansion6]):r[f"{label}{c}_kpa"]=float(v)
        pressure_rows.append(r)
    write_csv(output/"pressures.csv",pressure_rows)
    from src.evaluation.partial_replay_report import write_report
    write_report(output,summary,audit,metrics,histories,traces,panels)
    (output/"status.txt").write_text("complete\n")
    (output/"COMPLETE").touch(exist_ok=False)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
