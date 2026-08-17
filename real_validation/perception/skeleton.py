"""2D 骨架提取（唯一实现）。

逐行质心 → 弧长均匀重采样 → 可选 tip_fix。只依赖 numpy，供在线部署与离线数据
准备共用；src/utils/skeleton_2d.py 是本模块的薄壳。

节点顺序：node0 = tip（图像底部、运动末端），node N-1 = base（图像顶部、固定基座）。
"""

import numpy as np

TIP_FIX_APPLIED = "applied"
TIP_FIX_NOT_REQUESTED = "not_requested"
TIP_FIX_SKIP_FEW_POINTS = "n_points_lt_5"
TIP_FIX_SKIP_ZERO_SKELETON = "zero_skeleton"
TIP_FIX_SKIP_FEW_FOREGROUND = "foreground_lt_10"
TIP_FIX_SKIP_DEGENERATE_AXIS = "local_axis_degenerate"
TIP_FIX_SKIP_THIN_SLAB = "tip_slab_lt_3"


def _perpendicular_tip_fix_with_reason(skeleton, binary_img, n_points):
    """与 _perpendicular_tip_fix 相同的计算，同时返回 (skeleton, 生效/跳过原因)。

    原因取值见模块顶部 TIP_FIX_* 常量。供在线质量门控消费 —— 原实现的门控是
    静默跳过，调用方无从得知末端 node0 可能落在 cap 角落(B13)。
    """
    sk = skeleton.astype(np.float64)
    if n_points < 5:
        return skeleton, TIP_FIX_SKIP_FEW_POINTS
    if np.abs(sk).max() == 0:
        return skeleton, TIP_FIX_SKIP_ZERO_SKELETON
    ys, xs = np.where(binary_img > 0.5)
    if len(xs) < 10:
        return skeleton, TIP_FIX_SKIP_FEW_FOREGROUND
    pts = np.column_stack([xs.astype(float), ys.astype(float)])  # (col, row)
    far = sk[min(max(2, int(0.25 * n_points)), n_points - 1)]    # body 节点(偏 base, ~25%处)
    near = sk[min(max(1, int(0.10 * n_points)), n_points - 1)]   # body 节点(偏 tip, ~10%处)
    seg = near - far                      # 指向 tip 的局部轴方向
    L = float(np.hypot(*seg))
    if L < 1e-6:
        return skeleton, TIP_FIX_SKIP_DEGENERATE_AXIS
    d = seg / L
    proj = (pts - far) @ d
    w = float(binary_img.sum(1).max())    # 管径估计(最大行宽)
    slab = proj >= proj.max() - 0.4 * w   # 尖端垂直切片
    if int(slab.sum()) < 3:
        return skeleton, TIP_FIX_SKIP_THIN_SLAB
    node0 = pts[slab].mean(0)             # 垂直切片质心 = cap 中心线中点
    sk[0] = node0
    a = sk[min(3, n_points - 1)]          # 沿 body→node0 重布 node1,2 消折角
    sk[1] = node0 + (a - node0) / 3.0
    sk[2] = node0 + (a - node0) * 2.0 / 3.0
    return sk.astype(np.float32), TIP_FIX_APPLIED


def _perpendicular_tip_fix(skeleton, binary_img, n_points):
    """末端 node0 的"垂直于局部轴切片质心"修正（修倾斜 cap 的 corner 偏移 + node0-1-2 折角）。

    根因: 逐行质心对倾斜管的末端 cap 做**水平**切片, 最底行落在 cap 角落而非中点
    (弯管 cap 倾斜时, 底部几行变窄且偏向一侧→node0 落角落, node0-1-2 形成非物理尖折角)。
    修法: body 段保留(直管段水平切片本来就对), 仅重算 tip——从 body 节点估**局部轴方向**,
    在尖端做**垂直于轴**的切片(对管左右对称)→质心=局部中心线中点=cap 中点, 与倾斜无关;
    再沿 body→node0 重布 node1-2 消折角。body(node3+)不动。

    实测(实物 10116 帧): 34% 帧(M0 末端误差>4px)从 mean 6.94px→2.01px(-71%); body 不变;
    0 失败; 仅 1.3% 易帧小幅回退(≤3.5px)。详见 scripts/real/compare_skeleton_methods.py。

    仅在 n_points>=5 且 mask 非空足够时生效, 否则原样返回(skeleton 不变)。
    行为与迁移前完全一致；需要"是否生效"信号时改用 _perpendicular_tip_fix_with_reason。
    """
    return _perpendicular_tip_fix_with_reason(skeleton, binary_img, n_points)[0]


def extract_skeleton_2d(binary_img, n_points=31, tip_fix=False, return_info=False):
    """从二值图像提取 2D 中心线骨架。

    对图像每一行（从底到顶）计算白色像素的质心列坐标，
    然后沿弧长均匀重采样到 n_points 个点。

    Args:
        binary_img: (H, W) 二值图像，1=前景。
        n_points: 采样点数。
        tip_fix: 是否对末端 node0 做"垂直于局部轴切片质心"修正。默认 False
            (保持原有行为, 供 sim 等已验证管线)。实物管在弯曲时逐行质心会把 node0
            落到倾斜 cap 的角落, 置 True 可修正(见 _perpendicular_tip_fix)。
        return_info: True 时返回 (skeleton, info)；info 含 tip_fix_requested /
            tip_fix_applied / tip_fix_reason / n_foreground_px / n_valid_rows。
            默认 False，返回值与迁移前完全一致。

    Returns:
        skeleton_2d: (n_points, 2) 像素坐标 [col, row]，从底部到顶部排列。
                     若图像无前景，返回全零。
        (仅 return_info=True) info: dict，见上。
    """
    H, W = binary_img.shape
    n_foreground = int((binary_img > 0.5).sum())
    coords = []

    for row in range(H - 1, -1, -1):
        white_cols = np.where(binary_img[row] > 0.5)[0]
        if len(white_cols) > 0:
            center_col = white_cols.mean()
            coords.append([center_col, float(row)])

    def _wrap(skeleton, reason, n_valid_rows):
        if not return_info:
            return skeleton
        return skeleton, {
            "tip_fix_requested": bool(tip_fix),
            "tip_fix_applied": reason == TIP_FIX_APPLIED,
            "tip_fix_reason": reason,
            "n_foreground_px": n_foreground,
            "n_valid_rows": int(n_valid_rows),
        }

    if len(coords) < 2:
        return _wrap(np.zeros((n_points, 2), dtype=np.float32),
                     TIP_FIX_SKIP_ZERO_SKELETON, len(coords))

    coords = np.array(coords, dtype=np.float32)

    # 沿弧长均匀重采样
    diffs = np.diff(coords, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]

    if total_len < 1e-6:
        return _wrap(np.zeros((n_points, 2), dtype=np.float32),
                     TIP_FIX_SKIP_ZERO_SKELETON, len(coords))

    target_lens = np.linspace(0, total_len, n_points)

    resampled = np.zeros((n_points, 2), dtype=np.float32)
    resampled[:, 0] = np.interp(target_lens, cum_len, coords[:, 0])
    resampled[:, 1] = np.interp(target_lens, cum_len, coords[:, 1])

    if not tip_fix:
        return _wrap(resampled, TIP_FIX_NOT_REQUESTED, len(coords))
    fixed, reason = _perpendicular_tip_fix_with_reason(resampled, binary_img, n_points)
    return _wrap(fixed, reason, len(coords))


def batch_extract_skeleton_2d(images, n_points=31, tip_fix=False):
    """批量提取 2D 骨架。

    Args:
        images: (T, H, W) 二值图像序列。
        n_points: 采样点数。
        tip_fix: 末端 node0 垂直切片修正(见 extract_skeleton_2d), 默认 False。

    Returns:
        skeletons: (T, n_points, 2) 像素坐标。
    """
    T = images.shape[0]
    skeletons = np.zeros((T, n_points, 2), dtype=np.float32)
    for t in range(T):
        skeletons[t] = extract_skeleton_2d(images[t], n_points, tip_fix=tip_fix)
    return skeletons
