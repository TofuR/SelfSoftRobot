"""write_data_readme.py — 扫描 data/real_seq/*/ 自动生成每个数据集的参数 README + 总索引。

动机: 实物数据有多个 mask 来源(raw / repaired / sam2) × 节点数(15/31) × 是否 clean,
组合出多个 npz 变体, 容易混淆。本脚本读每个 npz 的真实元数据(n_points/tip_fix/T/A)
+ 从文件夹名解析(mask 源/clean/nodes), 在每个文件夹写 README.md, 并写总索引
data/real_seq/README.md(含命名约定 + 当前默认 + 归档建议)。

命名约定(文件夹后缀):
  _n15  = 15 节点 (否则 31)
  _rep  = mask 来自 masks_repaired (启发式修复)
  _sam2 = mask 来自 sam2/masks/<seq>_full (SAM2 视频)
  (无 _rep/_sam2) = raw masks
  _clean = 经 clean_transition_npz 全清洗 (否则仅 masks_to_transition 内的 clean_outlier)

用法:
  python scripts/real/write_data_readme.py            # 扫所有 data/real_seq/*
  python scripts/real/write_data_readme.py --current seq_20260627_163921_n15_rep_clean
"""
import argparse
import glob
import os

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEQ_DEFAULT = "seq_20260627_163921"


def parse_variant(name):
    """从文件夹名解析 (mask_source, n15, cleaned)。"""
    mask = "raw"
    cleaned = False
    n15 = False
    base = name
    if base.endswith("_clean"):
        cleaned = True; base = base[:-len("_clean")]
    if base.endswith("_sam2"):
        mask = "sam2"; base = base[:-len("_sam2")]
    elif base.endswith("_rep"):
        mask = "repaired"; base = base[:-len("_rep")]
    if base.endswith("_n15"):
        n15 = True; base = base[:-len("_n15")]
    return mask, n15, cleaned, base


def read_npz_meta(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    pos = z["positions"]; act = z["actions"]
    n = int(pos.shape[2])
    npts = int(z["n_points"].item()) if "n_points" in z else n
    tip = bool(z["tip_fix"].item()) if "tip_fix" in z else None
    return dict(T=int(pos.shape[0]), N=n, A=int(act.shape[1]),
                n_points=npts, tip_fix=tip)


def status_of(mask, n15, cleaned, name, current):
    tags = []
    if name == current:
        tags.append("★ 当前默认训练数据")
    if not n15:
        tags.append("旧版(31节点, 已弃用: 用户验证 15 节点误差不大)")
    if not cleaned:
        tags.append("中间产物(仅 clean_outlier, 未全清洗)")
    if mask == "raw":
        tags.append("raw mask(有腐败, 仅对比用)")
    if mask == "sam2" and cleaned:
        tags.append("SAM2+clean (新, 可与 rep_clean 对比)")
    if mask == "sam2" and not cleaned:
        tags.append("SAM2 未清洗 (新)")
    if mask == "repaired" and cleaned and n15 and name != current:
        tags.append("rep+clean (与当前同管线, 备选)")
    return " / ".join(tags) if tags else "—"


def mask_dir_hint(seq, mask):
    return {"raw": f"real_capture/data/derived/{seq}/masks",
            "repaired": f"real_capture/data/derived/{seq}/masks_repaired",
            "sam2": f"sam2/masks/{seq}_full"}[mask]


def write_folder_readme(d, name, meta, mask, n15, cleaned, seq, current):
    tip_s = "True" if meta["tip_fix"] is True else ("False" if meta["tip_fix"] is False else "(无, 旧版)")
    clean_s = "是 (clean_transition_npz 全清洗)" if cleaned else "否 (仅 masks_to_transition 内 clean_outlier)"
    md = f"""# {name}

实物 transition 训练数据(免标定 2D)。骨架 state = 图像像素 `[col,row,0]`。

| 参数 | 值 |
|---|---|
| mask 来源 | {mask} ({mask_dir_hint(seq, mask)}) |
| 节点数 N | {meta['N']} |
| tip_fix | {tip_s} |
| 清洗 | {clean_s} |
| 帧数 T (train) | {meta['T']} |
| action_dim | {meta['A']} |
| npz 文件 | `train/{seq}_train.npz`, `val/{seq}_val.npz` |

生成命令:
```
python scripts/real/masks_to_transition_npz.py \\
    --seq real_capture/data/raw/{seq} \\
    --masks-dir {mask_dir_hint(seq, mask)} \\
    --out-root data/real_seq/{name} --n-points {meta['N']}
```
"""
    if cleaned:
        pre = name[:-len("_clean")] if name.endswith("_clean") else name
        md += f"""```
python scripts/real/clean_transition_npz.py \\
    --seq {pre} --in-root data/real_seq/{pre} --out-root data/real_seq/{name}
```
"""
    md += f"""
状态: {status_of(mask, n15, cleaned, name, current)}

训练: `--data_dir data/real_seq/{name}/train`
"""
    with open(os.path.join(d, "README.md"), "w") as f:
        f.write(md)


def main():
    pa = argparse.ArgumentParser(description="生成 data/real_seq/*/ README + 总索引")
    pa.add_argument("--root", default=os.path.join(PROJECT_ROOT, "data", "real_seq"))
    pa.add_argument("--seq", default=SEQ_DEFAULT, help="原图序列名(写 mask 路径用)")
    pa.add_argument("--current", default=f"{SEQ_DEFAULT}_n15_rep_clean",
                    help="当前默认训练数据集名(在索引里标★)")
    args = pa.parse_args()

    rows = []
    for d in sorted(glob.glob(os.path.join(args.root, "*/"))):
        name = os.path.basename(d.rstrip("/"))
        if name.startswith("_") or name == "_archive":
            continue
        npzs = sorted(glob.glob(os.path.join(d, "train", "*.npz")))
        if not npzs:
            continue
        meta = read_npz_meta(npzs[0])
        mask, n15, cleaned, base = parse_variant(name)
        if base != args.seq:
            continue
        write_folder_readme(d, name, meta, mask, n15, cleaned, args.seq, args.current)
        rows.append((name, mask, meta["N"], meta["tip_fix"], cleaned, meta["T"], meta["A"],
                     status_of(mask, n15, cleaned, name, args.current)))
        print(f"  写 {name}/README.md  (mask={mask} N={meta['N']} clean={cleaned})")

    lines = [
        f"# data/real_seq/ — 实物 transition 训练数据索引\n",
        f"原图序列: `{args.seq}` (cam0 10214 帧)。骨架 state = 图像像素 `[col,row,0]` (免标定)。\n",
        "## 命名约定\n",
        "`<seq>[_n15][_rep|_sam2][_clean]`\n",
        "- `_n15` = 15 节点 (否则 31; 用户验证 15 节点误差不大, 当前默认)\n",
        "- `_rep` = mask 来自 `masks_repaired` (启发式修复) / `_sam2` = SAM2 视频 / (无) = raw masks\n",
        "- `_clean` = 经 `clean_transition_npz` 全清洗 (否则仅 `masks_to_transition` 内的 `clean_outlier`)\n\n",
        "## mask 来源(3 种, 都 10214 帧)\n",
        f"- `real_capture/data/derived/{args.seq}/masks` — RAW (white_on_blue 分割, 有腐败)\n",
        f"- `real_capture/data/derived/{args.seq}/masks_repaired` — 启发式修复 (hand+static+actuated)\n",
        f"- `sam2/masks/{args.seq}_full` — SAM2 视频 (分块双向传播, 最干净; area std 1.7%)\n\n",
        "## 骨架化(src/utils/skeleton_2d.py)\n",
        "逐行质心 + 弧长重采样 + **tip_fix**(末端垂直切片修 corner 偏移)。\n",
        "实测 7 法对比(SAM2 mask, n=31): tip_fix 末端误差 0.80px **最优**; medial_axis 7.50px **最差**\n",
        "(corner 问题=倾斜 cap 水平切片, 非细化伪影, medial_axis 治不了且抖 body)。故不换通用骨架化。\n\n",
        "## 数据集列表\n",
        "| 目录 | mask源 | N | tip_fix | 清洗 | T(train) | A | 状态 |\n",
        "|---|---|---|---|---|---|---|---|\n",
    ]
    for name, mask, n, tip, cleaned, T, A, status in rows:
        tip_s = "True" if tip is True else ("False" if tip is False else "?")
        clean_s = "是" if cleaned else "否"
        star = "★ " if name == args.current else ""
        lines.append(f"| {star}`{name}` | {mask} | {n} | {tip_s} | {clean_s} | {T} | {A} | {status} |\n")
    lines += [
        "\n## 当前默认\n",
        f"训练: `--data_dir data/real_seq/{args.current}/train`\n\n",
        "## 归档建议(待用户确认)\n",
        "- 31 节点变体(无 `_n15`): 已弃用, 可移至 `_archive/`。\n",
        "- 未清洗中间产物(`_n15`, `_n15_rep`, `_n15_sam2`): 仅 clean_outlier, 训练用其 `_clean` 版。\n",
        "- 保留: `*_n15_rep_clean`(当前) + `*_n15_sam2_clean`(SAM2 对比) + `*_n15_sam2`(SAM2 未clean, 用户备选)。\n",
    ]
    with open(os.path.join(args.root, "README.md"), "w") as f:
        f.writelines(lines)
    print(f"\n→ 总索引 {args.root}/README.md  ({len(rows)} 个数据集, 当前默认={args.current})")


if __name__ == "__main__":
    main()
