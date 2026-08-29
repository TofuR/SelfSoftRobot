"""2D 骨架提取（唯一实现）。

新通用接口：mask细化/中轴 → 最长主路径 → 分段弧长重采样；不假设机器人逐行单值或
某一段静止。旧逐行质心 + tip_fix接口原样保留，供在线兼容和历史实验复现。
src/utils/skeleton_2d.py 是旧公开接口的薄壳；细化方法额外依赖scikit-image。

节点顺序：node0 = base（图像顶部、固定基座），node N-1 = tip（图像底部、运动末端）。
"""

import numpy as np


def allocate_segment_intervals(n_points, segment_lengths):
    """按物理长度为各段分配节点区间；相邻段共享关节节点。"""
    lengths = np.asarray(segment_lengths, dtype=np.float64)
    if lengths.ndim != 1 or len(lengths) == 0 or not np.isfinite(lengths).all():
        raise ValueError("segment_lengths 必须是一维有限正数")
    if (lengths <= 0).any():
        raise ValueError("segment_lengths 必须全部为正")
    total = int(n_points) - 1
    if total < len(lengths):
        raise ValueError("n_points-1 不能小于机器人段数")
    intervals = np.ones(len(lengths), dtype=np.int64)
    remaining = total - len(lengths)
    raw = remaining * lengths / lengths.sum()
    intervals += np.floor(raw).astype(np.int64)
    for index in np.argsort(-(raw - np.floor(raw)))[:total - int(intervals.sum())]:
        intervals[index] += 1
    return tuple(int(value) for value in intervals)


def _resample_path_segmented(path, n_points, segment_lengths):
    """沿有序 base→tip 路径采样，并把物理段边界固定到明确节点。"""
    path = np.asarray(path, dtype=np.float64)
    delta = np.diff(path, axis=0)
    seg = np.sqrt((delta ** 2).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    if cumulative[-1] < 1e-6:
        return np.zeros((n_points, 2), np.float32), (), ()
    lengths = tuple(float(value) for value in (segment_lengths or (1.0,)))
    intervals = allocate_segment_intervals(n_points, lengths)
    physical_bounds = np.concatenate([[0.0], np.cumsum(lengths)])
    physical_bounds /= physical_bounds[-1]
    targets = []
    joint_nodes = []
    node_offset = 0
    for index, n_interval in enumerate(intervals):
        local = np.linspace(physical_bounds[index], physical_bounds[index + 1],
                            n_interval + 1)
        if index:
            local = local[1:]
        targets.extend(local.tolist())
        node_offset += n_interval
        if index < len(intervals) - 1:
            joint_nodes.append(node_offset)
    targets = np.asarray(targets, dtype=np.float64) * cumulative[-1]
    out = np.zeros((n_points, 2), np.float32)
    out[:, 0] = np.interp(targets, cumulative, path[:, 0])
    out[:, 1] = np.interp(targets, cumulative, path[:, 1])
    return out, intervals, tuple(joint_nodes)


def _medial_longest_path(binary_img, algorithm="skeletonize", anchor_xy=None):
    """提取主路径；有base锚点时从锚点到最远tip，否则使用图直径。"""
    try:
        from skimage.morphology import medial_axis, skeletonize
    except ImportError as error:  # pragma: no cover - 依赖缺失应明确暴露
        raise RuntimeError("medial_axis骨架方法需要scikit-image") from error
    from collections import deque

    full_mask = np.asarray(binary_img) > 0
    fg_y, fg_x = np.where(full_mask)
    if not len(fg_x):
        return None, 0
    # 机器人通常只占整幅图像的一条窄ROI；裁剪后中轴速度提升一个数量级。
    x0, x1 = int(fg_x.min()), int(fg_x.max()) + 1
    y0, y1 = int(fg_y.min()), int(fg_y.max()) + 1
    pad = 2
    cropped = np.pad(full_mask[y0:y1, x0:x1], pad, mode="constant")
    medial = medial_axis(cropped) if algorithm == "medial_axis" else skeletonize(cropped)
    ys, xs = np.where(medial)
    xs = xs + x0 - pad
    ys = ys + y0 - pad
    points = set(zip(xs.tolist(), ys.tolist()))
    if len(points) < 2:
        return None, int(len(points))
    adjacency = {}
    for x, y in points:
        adjacency[(x, y)] = [
            (x + dx, y + dy)
            for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            if (dx or dy) and (x + dx, y + dy) in points
        ]

    def bfs(source):
        previous = {source: None}
        distance = {source: 0}
        queue = deque([source])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor not in previous:
                    previous[neighbor] = current
                    distance[neighbor] = distance[current] + 1
                    queue.append(neighbor)
        farthest = max(distance, key=distance.get)
        return farthest, previous

    if anchor_xy is not None:
        anchor = np.asarray(anchor_xy, dtype=np.float64).reshape(2)
        base = min(points, key=lambda point: float(np.linalg.norm(
            np.asarray(point, dtype=np.float64) - anchor)))
        tip, previous = bfs(base)
        ordered = []
        current = tip
        while current is not None:
            ordered.append(current)
            current = previous[current]
        ordered.reverse()  # base → tip
        path = np.asarray(ordered, dtype=np.float64) if len(ordered) > 1 else None
        return path, int(len(points))

    unseen = set(points)
    best_path = []
    while unseen:
        seed = next(iter(unseen))
        component = {seed}
        queue = deque([seed])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor not in component:
                    component.add(neighbor); queue.append(neighbor)
        unseen.difference_update(component)
        first, _ = bfs(seed)
        second, previous = bfs(first)
        ordered = []
        current = second
        while current is not None:
            ordered.append(current)
            current = previous[current]
        if len(ordered) > len(best_path):
            best_path = ordered
    path = np.asarray(best_path, dtype=np.float64) if len(best_path) > 1 else None
    return path, int(len(points))


ENDPOINT_FIX_APPLIED = "applied"
ENDPOINT_FIX_NOT_REQUESTED = "not_requested"
ENDPOINT_FIX_SKIP_SHORT_PATH = "path_too_short"
ENDPOINT_FIX_SKIP_WIDTH = "width_estimate_failed"
ENDPOINT_FIX_SKIP_TANGENT = "local_tangent_failed"
ENDPOINT_FIX_SKIP_SECTION = "full_width_section_not_found"
ENDPOINT_FIX_BASE_ANCHORED = "base_anchor"


def _path_cumulative(path):
    delta = np.diff(np.asarray(path, dtype=np.float64), axis=0)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(delta, axis=1))])


def _endpoint_width(mask, path):
    """用中轴上的距离变换估计局部管径，避开细化端点自身的零宽分支。"""
    from scipy.ndimage import distance_transform_edt

    distance = distance_transform_edt(np.asarray(mask) > 0)
    path = np.asarray(path, dtype=np.float64)
    xy = np.rint(path).astype(np.int64)
    valid = ((xy[:, 0] >= 0) & (xy[:, 0] < distance.shape[1]) &
             (xy[:, 1] >= 0) & (xy[:, 1] < distance.shape[0]))
    radii = np.full(len(path), np.nan, dtype=np.float64)
    radii[valid] = distance[xy[valid, 1], xy[valid, 0]]
    finite = np.isfinite(radii) & (radii > 0)
    radii_all = radii[finite]
    if not len(radii_all):
        return 0.0
    global_width = float(2.0 * np.quantile(radii_all, 0.75))
    cumulative = _path_cumulative(path)
    local_span = min(0.35 * cumulative[-1], max(2.5 * global_width, 12.0))
    local = finite & (cumulative <= local_span)
    radii_local = radii[local]
    if len(radii_local) >= 3:
        radii_all = radii_local
    # 端部的细化分支可能贴着边界；上四分位更接近主体半径，又不会被个别鼓包主导。
    return float(2.0 * np.quantile(radii_all, 0.75))


def _fix_one_endcap(mask, path_from_end, width_ratio=0.85):
    """把一个细化端点移到端帽宽边中心，并返回替换主路径时的内侧连接点。"""
    path = np.asarray(path_from_end, dtype=np.float64)
    cumulative = _path_cumulative(path)
    if len(path) < 5 or cumulative[-1] < 6.0:
        return None, ENDPOINT_FIX_SKIP_SHORT_PATH

    width = _endpoint_width(mask, path)
    if not np.isfinite(width) or width < 3.0:
        return None, ENDPOINT_FIX_SKIP_WIDTH

    # 不用细化端点直接估切向：旋转矩形的细化端点可能沿端帽分叉到角点。
    # 在约 0.65~1.65 个管径的内侧窗口取弦方向，可跳过分支又保持端部局部性。
    start_s = min(0.65 * width, 0.18 * cumulative[-1])
    end_s = min(1.65 * width, 0.38 * cumulative[-1])
    if end_s <= start_s + 2.0:
        start_s = 0.12 * cumulative[-1]
        end_s = 0.42 * cumulative[-1]
    start = np.array([np.interp(start_s, cumulative, path[:, dim]) for dim in range(2)])
    end = np.array([np.interp(end_s, cumulative, path[:, dim]) for dim in range(2)])
    tangent = end - start                         # 端部 → 机器人内部
    tangent_length = float(np.linalg.norm(tangent))
    if tangent_length < 1e-6:
        return None, ENDPOINT_FIX_SKIP_TANGENT
    tangent /= tangent_length
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)

    ys, xs = np.where(np.asarray(mask) > 0)
    points = np.column_stack([xs, ys]).astype(np.float64)
    origin = path[0]
    relative = points - origin
    axial = relative @ tangent
    transverse = relative @ normal
    # 限制到端部邻域，避免自靠近/自交的另一段前景污染横截面。
    local = ((axial >= -1.75 * width) & (axial <= 0.85 * width) &
             (np.abs(transverse) <= 1.75 * width))
    axial = axial[local]
    transverse = transverse[local]
    if len(axial) < 6:
        return None, ENDPOINT_FIX_SKIP_SECTION

    slab_half = 0.8
    scan = np.arange(float(axial.min()), min(float(axial.max()), 0.6 * width) + 0.5,
                     0.5)
    candidate = None
    for section_s in scan:                       # 最外侧 → 机器人内部
        section = np.abs(axial - section_s) <= slab_half
        if int(section.sum()) < 3:
            continue
        lo = float(transverse[section].min())
        hi = float(transverse[section].max())
        if hi - lo >= width_ratio * width:
            candidate = (section_s, 0.5 * (lo + hi), hi - lo)
            break
    if candidate is None:
        return None, ENDPOINT_FIX_SKIP_SECTION

    section_s, center_n, section_width = candidate
    # 先用“足够宽的横截面”可靠求出法向中心，再只沿该中心附近向外追到mask边界。
    # 这样平/斜端帽不会取角点，圆端帽也不会仍停在85%宽度截面而遗漏末端长度。
    center_band = max(1.5, 0.10 * width)
    central = np.abs(transverse - center_n) <= center_band
    cap_s = section_s
    if int(central.sum()) >= 3:
        central_axial = axial[central]
        central_normal = transverse[central]
        bin_id = np.floor((central_normal - central_normal.min()) / 0.75).astype(int)
        edge_samples = np.array([
            central_axial[bin_id == value].min() for value in np.unique(bin_id)
        ])
        if len(edge_samples):
            cap_s = max(float(np.median(edge_samples)), section_s - 0.80 * width)
    center = origin + cap_s * tangent + center_n * normal
    # 从修正点直连已经脱离端帽分支的内侧路径，不能保留“中心→角点→主体”的折线。
    join_s = min(max(0.65 * width, 4.0), 0.30 * cumulative[-1])
    join_index = int(np.searchsorted(cumulative, join_s, side="left"))
    join_index = min(max(join_index, 1), len(path) - 2)
    return {
        "center": center,
        "raw": origin.copy(),
        "join_index": join_index,
        "width_px": width,
        "section_width_px": section_width,
        "axial_shift_px": float(-cap_s),
        "shift_px": float(np.linalg.norm(center - origin)),
    }, ENDPOINT_FIX_APPLIED


def _fix_path_endcaps(mask, path, enabled=True, fix_base=True):
    """修正 base→tip 路径的两个端帽，并保持节点方向。"""
    path = np.asarray(path, dtype=np.float64)
    if not enabled:
        return path, None, None, ENDPOINT_FIX_NOT_REQUESTED, ENDPOINT_FIX_NOT_REQUESTED
    tip, tip_reason = _fix_one_endcap(mask, path[::-1])
    if fix_base:
        base, base_reason = _fix_one_endcap(mask, path)
    else:
        base, base_reason = None, ENDPOINT_FIX_BASE_ANCHORED

    base_join = base["join_index"] if base is not None else 0
    tip_join = tip["join_index"] if tip is not None else 0
    stop = len(path) - tip_join
    if base_join >= stop:
        return path, None, None, ENDPOINT_FIX_SKIP_SHORT_PATH, ENDPOINT_FIX_SKIP_SHORT_PATH
    pieces = []
    if base is not None:
        pieces.append(base["center"][None])
    pieces.append(path[base_join:stop])
    if tip is not None:
        pieces.append(tip["center"][None])
    return np.concatenate(pieces, axis=0), tip, base, tip_reason, base_reason


def extract_centerline_2d(binary_img, n_points=15, method="skeletonize",
                          segment_lengths=(1.0, 1.0), base_anchor_xy=None,
                          tip_fix=True, endpoint_fix=True, return_info=False):
    """通用整臂中心线；不假设静态段，也不依赖动作通道。

    ``skeletonize``（默认、快速）和 ``medial_axis``（较慢）均支持S形/局部水平形态；
    ``row_centroid`` 保留旧单段流程。
    细化算法会让长条mask的端点向内收缩或分叉到端帽角点；``endpoint_fix`` 默认在
    tip/base 两端估计局部切向、主体宽度和宽边中心，再替换端帽分支后统一重采样。
    输出始终为 ``node0=base``、``nodeN-1=tip``。若提供 ``base_anchor_xy``，用它
    确定路径方向；否则沿用当前实验中基座靠图像顶部的约定。
    """
    mask = np.asarray(binary_img)
    segment_lengths = tuple(float(v) for v in (segment_lengths or (1.0,)))
    info = {
        "method_requested": method,
        "method_used": method,
        "success": False,
        "reason": "empty",
        "arc_length_px": 0.0,
        "n_medial_pixels": 0,
        "segment_lengths": segment_lengths,
        "segment_intervals": (),
        "joint_node_indices": (),
        "endpoint_fix_requested": bool(endpoint_fix),
        "tip_endpoint_fix_reason": ENDPOINT_FIX_NOT_REQUESTED,
        "base_endpoint_fix_reason": ENDPOINT_FIX_NOT_REQUESTED,
    }
    if not np.any(mask):
        result = np.zeros((n_points, 2), np.float32)
        return (result, info) if return_info else result
    if method == "row_centroid":
        result, old_info = extract_skeleton_2d(
            mask, n_points=n_points, tip_fix=tip_fix, return_info=True)
        intervals = allocate_segment_intervals(n_points, segment_lengths)
        joints = tuple(np.cumsum(intervals)[:-1].astype(int).tolist())
        info.update({
            "success": bool(np.abs(result).max() > 0),
            "reason": old_info.get("tip_fix_reason", "row_centroid"),
            "segment_intervals": intervals,
            "joint_node_indices": joints,
            "arc_length_px": float(np.linalg.norm(np.diff(result, axis=0), axis=1).sum()),
        })
        return (result, info) if return_info else result
    if method not in ("skeletonize", "medial_axis"):
        raise ValueError(f"未知中心线方法: {method}")

    path, n_medial = _medial_longest_path(
        mask, algorithm=method, anchor_xy=base_anchor_xy)
    info["n_medial_pixels"] = n_medial
    if path is None or len(path) < 2:
        result = np.zeros((n_points, 2), np.float32)
        info["reason"] = "medial_path_too_short"
        return (result, info) if return_info else result
    if base_anchor_xy is None:
        first_is_base = path[0, 1] <= path[-1, 1]
    else:
        anchor = np.asarray(base_anchor_xy, dtype=np.float64)
        first_is_base = np.linalg.norm(path[0] - anchor) <= np.linalg.norm(path[-1] - anchor)
    if not first_is_base:
        path = path[::-1]
    if base_anchor_xy is not None:
        # 显式base锚点是物理固定端合同：细化中轴会在圆帽内收若干像素，直接把
        # 权威锚点接回路径，避免node0随细化端点抖动或停在管体内部。
        anchor = np.asarray(base_anchor_xy, dtype=np.float64).reshape(2)
        if np.linalg.norm(path[0] - anchor) > 1e-6:
            path = np.concatenate([anchor[None], path], axis=0)
    raw_path = path.copy()
    path, tip_cap, base_cap, tip_reason, base_reason = _fix_path_endcaps(
        mask, path, enabled=endpoint_fix, fix_base=base_anchor_xy is None)
    result, intervals, joints = _resample_path_segmented(
        path, n_points=n_points, segment_lengths=segment_lengths)
    info.update({
        "success": bool(np.abs(result).max() > 0),
        "reason": "ok",
        "segment_intervals": intervals,
        "joint_node_indices": joints,
        "arc_length_px": float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
        "raw_arc_length_px": float(np.linalg.norm(
            np.diff(raw_path, axis=0), axis=1).sum()),
        "raw_base_xy": tuple(float(v) for v in raw_path[0]),
        "raw_tip_xy": tuple(float(v) for v in raw_path[-1]),
        "fixed_base_xy": tuple(float(v) for v in path[0]),
        "fixed_tip_xy": tuple(float(v) for v in path[-1]),
        "tip_endpoint_fix_reason": tip_reason,
        "base_endpoint_fix_reason": base_reason,
        "tip_endpoint_fix_applied": tip_reason == ENDPOINT_FIX_APPLIED,
        "base_endpoint_fix_applied": base_reason == ENDPOINT_FIX_APPLIED,
        "tip_endpoint_shift_px": float(tip_cap["shift_px"] if tip_cap else 0.0),
        "base_endpoint_shift_px": float(base_cap["shift_px"] if base_cap else 0.0),
        "tip_axial_extension_px": float(tip_cap["axial_shift_px"] if tip_cap else 0.0),
        "base_axial_extension_px": float(base_cap["axial_shift_px"] if base_cap else 0.0),
        "tip_width_px": float(tip_cap["width_px"] if tip_cap else 0.0),
        "base_width_px": float(base_cap["width_px"] if base_cap else 0.0),
    })
    return (result, info) if return_info else result

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
    静默跳过，调用方无从得知末端 nodeN-1 可能落在 cap 角落。
    """
    # 公开合同为 base→tip；端帽几何在局部使用 tip→base
    # 视图计算，完成后再恢复公开节点顺序。
    sk = skeleton[::-1].astype(np.float64).copy()
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
    tip_point = pts[slab].mean(0)          # 垂直切片质心 = cap 中心线中点
    sk[0] = tip_point
    a = sk[min(3, n_points - 1)]          # 局部视图中沿 body→tip 重布相邻点
    sk[1] = tip_point + (a - tip_point) / 3.0
    sk[2] = tip_point + (a - tip_point) * 2.0 / 3.0
    return sk[::-1].astype(np.float32), TIP_FIX_APPLIED


def _perpendicular_tip_fix(skeleton, binary_img, n_points):
    """末端 nodeN-1 的"垂直于局部轴切片质心"修正。

    根因: 逐行质心对倾斜管的末端 cap 做**水平**切片, 最底行落在 cap 角落而非中点
    (弯管 cap 倾斜时,底部几行变窄且偏向一侧→末端落角落并形成非物理尖折角)。
    修法: body 段保留(直管段水平切片本来就对), 仅重算 tip——从 body 节点估**局部轴方向**,
    在尖端做**垂直于轴**的切片(对管左右对称)→质心=局部中心线中点=cap 中点, 与倾斜无关;
    再沿局部 body→tip 重布相邻点消折角。

    实测(实物 10116 帧): 34% 帧(M0 末端误差>4px)从 mean 6.94px→2.01px(-71%); body 不变;
    0 失败; 仅 1.3% 易帧小幅回退(≤3.5px)。详见 scripts/real/compare_skeleton_methods.py。

    仅在 n_points>=5 且 mask 非空足够时生效, 否则原样返回(skeleton 不变)。
    行为与迁移前完全一致；需要"是否生效"信号时改用 _perpendicular_tip_fix_with_reason。
    """
    return _perpendicular_tip_fix_with_reason(skeleton, binary_img, n_points)[0]


def extract_skeleton_2d(binary_img, n_points=31, tip_fix=False, return_info=False):
    """从二值图像提取 2D 中心线骨架。

    对图像每一行（从顶到底）计算白色像素的质心列坐标，
    然后沿弧长均匀重采样到 n_points 个点。

    Args:
        binary_img: (H, W) 二值图像，1=前景。
        n_points: 采样点数。
        tip_fix: 是否对末端 nodeN-1 做"垂直于局部轴切片质心"修正。默认 False
            实物管在弯曲时逐行质心会把末端
            落到倾斜 cap 的角落, 置 True 可修正(见 _perpendicular_tip_fix)。
        return_info: True 时返回 (skeleton, info)；info 含 tip_fix_requested /
            tip_fix_applied / tip_fix_reason / n_foreground_px / n_valid_rows。
            默认 False，返回值与迁移前完全一致。

    Returns:
        skeleton_2d: (n_points, 2) 像素坐标 [col, row]，从 base 到 tip 排列。
                     若图像无前景，返回全零。
        (仅 return_info=True) info: dict，见上。
    """
    H, W = binary_img.shape
    n_foreground = int((binary_img > 0.5).sum())
    coords = []

    for row in range(H):
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
        tip_fix: 末端 nodeN-1 垂直切片修正(见 extract_skeleton_2d), 默认 False。

    Returns:
        skeletons: (T, n_points, 2) 像素坐标。
    """
    T = images.shape[0]
    skeletons = np.zeros((T, n_points, 2), dtype=np.float32)
    for t in range(T):
        skeletons[t] = extract_skeleton_2d(images[t], n_points, tip_fix=tip_fix)
    return skeletons
