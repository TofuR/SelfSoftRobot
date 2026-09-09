"""Strict development NPZ -> raw command -> original image pairing for replay."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


def read_csv(path):
    with Path(path).open(newline="",encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


@dataclass
class ReplayData:
    actions: np.ndarray
    positions: np.ndarray
    camera_positions: np.ndarray
    raw_indices: np.ndarray
    images: list[Path]
    pairing: list[dict]
    model_to_camera: np.ndarray
    scale_kpa: np.ndarray
    meta: dict
    audit: dict
    evaluation_start: int
    radius_px: float
    expansion6: np.ndarray


def load_replay_data(npz_path, raw_root, saved_config, *, model_dt, norm_factor):
    if not np.isfinite([model_dt, norm_factor]).all() or min(model_dt, norm_factor) <= 0:
        raise ValueError("invalid checkpoint time or action normalization")
    npz_path,raw_root = Path(npz_path),Path(raw_root)
    if npz_path.parent.name != "dev":
        raise ValueError("this development runner only accepts a dev NPZ")
    manifest_path = npz_path.parent.parent/"split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    matches = [x for x in manifest["roles"]["dev"]["files"] if Path(x["source"]).name==npz_path.name]
    if len(matches)!=1:
        raise ValueError("ambiguous or missing development lineage")
    source = matches[0]
    start,stop = source["source_slice"]
    # Legacy train NPZ is a raw prefix. Verify all commands below instead of
    # trusting that historical naming convention as an alignment guarantee.
    indices = np.arange(start,stop)
    with np.load(npz_path,allow_pickle=False) as payload:
        data = {k:payload[k] for k in payload.files}
    meta = json.loads((raw_root/"meta.json").read_text())
    raw_actions = read_csv(raw_root/"actions6.csv")
    samples = read_csv(raw_root/"samples.csv")
    commands = read_csv(raw_root/"commands.csv")
    sample_map = {int(x["frame_idx"]):x for x in samples}
    command_map = {int(x["command_id"]):x for x in commands}
    if len(sample_map)!=len(samples) or len(command_map)!=len(commands):
        raise ValueError("duplicate frame/command identity")
    if len(indices)!=len(data["actions"]) or stop>len(raw_actions):
        raise ValueError("raw/derived length mismatch")
    channels = data["model_action_channels"].astype(int)
    if list(channels)!=saved_config["action_view"]["model_action_channels"]:
        raise ValueError("checkpoint action channel mismatch")
    if str(data["node_order"].item()) != saved_config["node_order"]:
        raise ValueError("checkpoint node order mismatch")
    if str(data["state_coordinate_frame"].item()) != saved_config["state_view"]["state_coordinate_frame"]:
        raise ValueError("checkpoint coordinate frame mismatch")
    if str(data["state_length_unit"].item()) != "mm":
        raise ValueError("expected physical millimetres")
    if not np.isclose(meta["action_interval_s"],model_dt):
        raise ValueError("nominal command interval differs from checkpoint")
    raw_six = np.asarray([[float(raw_actions[i][f"c{c}"]) for c in range(6)] for i in indices])
    reconstructed = data["actions"]*data["raw_action_scale6_kpa"]
    difference = float(np.max(abs(reconstructed-raw_six)))
    if not np.isfinite(reconstructed).all() or difference>0.002:
        raise ValueError(f"raw pressure alignment failed: {difference} kPa")
    if np.any(raw_six < np.asarray(meta["lo6"])-1e-5) or np.any(raw_six > np.asarray(meta["hi6"])+1e-5):
        raise ValueError("recorded pressure exceeds declared range")
    expansion = data["action_expansion6"].astype(int)
    if not np.allclose(data["raw_action_scale6_kpa"][channels],data["action_scale_kpa"]):
        raise ValueError("raw/model action scale mismatch")
    expected_expansion = saved_config["action_view"].get("action_expansion6")
    if expected_expansion is not None and list(expansion) != expected_expansion:
        raise ValueError("checkpoint six-channel expansion mismatch")
    if not np.allclose(raw_six[:,channels][:,expansion],raw_six,atol=0.002):
        raise ValueError("six-channel expansion mismatch")
    transform = json.loads(data["skeleton_frame_transform"].item())
    matrix = np.asarray(transform["model_to_camera_matrix"],dtype=np.float32)
    positions = data["positions"].transpose(0,2,1)[:,:,:2].astype(np.float32)
    camera = data["positions_camera_px"].transpose(0,2,1)[:,:,:2].astype(np.float32)
    projected = np.concatenate([positions,np.ones((*positions.shape[:2],1))],axis=-1)@matrix.T
    projection_error = float(abs(projected[:,:,:2]/projected[:,:,2:]-camera).max())
    if not np.isfinite(positions).all() or not np.isfinite(camera).all() or projection_error>0.01:
        raise ValueError("image/model coordinates are inconsistent")
    singular = np.linalg.svd(matrix[:2,:2],compute_uv=False)
    if not np.allclose(matrix[2],[0,0,1]) or not np.isclose(singular[0],singular[1],rtol=1e-5):
        raise ValueError("fixed pixel radius requires an affine similarity transform")
    pairing=[]; images=[]
    ordered_commands = sorted(commands,key=lambda x:float(x["t_command"]))
    next_times={int(x["command_id"]):float(y["t_command"]) for x,y in zip(ordered_commands,ordered_commands[1:])}
    for local,i in enumerate(indices):
        s=sample_map[int(i)]; c=command_map[int(s["command_id"])]
        applied=np.array([float(c[f"action_command{j}"]) for j in range(6)])
        if not np.allclose(applied,raw_six[local],atol=0.002) or c["communication_status"]!="ack":
            raise ValueError(f"command receipt mismatch at frame {i}")
        grab=float(s["t_grab"]); age=float(s["frame_age0"]); sent=float(c["t_command"])
        received=grab-age
        if not np.isfinite([grab,age,sent]).all() or age<0:
            raise ValueError("invalid frame times")
        if abs(float(raw_actions[i]["t_sec"])-grab)>0.002:
            raise ValueError("sample/action clock mismatch")
        valid=(sent<=received<next_times.get(int(c["command_id"]),float("inf"))
               and age<=meta.get("max_frame_age",0.5))
        image=raw_root/"cam0"/f"{i:05d}.png"
        if not image.is_file():
            raise FileNotFoundError(image)
        images.append(image)
        pairing.append({"local_index":local,"raw_frame":int(i),"command_id":int(s["command_id"]),
                        "t_command_s":sent,"t_grab_s":grab,"host_frame_receive_s":received,
                        "frame_age_s":age,"receive_after_command_s":received-sent,
                        "pair_valid":bool(valid),"image":str(image.resolve())})
    command_times=np.array([r["t_command_s"] for r in pairing])
    if np.any(np.diff(command_times)<=0):
        raise ValueError("command sequence is not strictly chronological")
    evaluation=data["evaluation_mask"]
    if evaluation.dtype!=np.bool_ or not evaluation.any():
        raise ValueError("missing development scoring mask")
    first=int(np.flatnonzero(evaluation)[0])
    if not evaluation[first:].all() or first!=source["evaluation_start"]:
        raise ValueError("evaluation mask/lineage mismatch")
    audit={"schema":"partial_replay_pairing_v1","data_role":"within-sequence development",
           "npz":str(npz_path.resolve()),"raw_root":str(raw_root.resolve()),
           "split_manifest":str(manifest_path.resolve()),"source_slice":[start,stop],
           "frames":len(indices),"pressure_pair_max_kpa":difference,"projection_max_px":projection_error,
           "invalid_time_pairs":sum(not r["pair_valid"] for r in pairing),
           "command_interval_percentiles_s":np.percentile(np.diff(command_times),[0,50,95,99,100]).tolist(),
           "model_dt_s":model_dt,"time_policy":"one recorded command per fixed checkpoint step; measured timing retained as approximation",
           "image_time":"host receive estimate = t_grab - frame_age0; not exposure",
           "transform_source":transform.get("source"),"calibration_causal":False,
           "test_read":False,"reference_kind":"offline SAM2 centerline pseudo-reference"}
    return ReplayData(data["actions"][:,channels].astype(np.float32)/norm_factor,positions,camera,
                      indices,images,pairing,matrix,data["action_scale_kpa"].astype(np.float32),meta,audit,
                      first,float(data["robot_diameter_px"])/2,expansion)
