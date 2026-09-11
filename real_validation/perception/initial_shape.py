"""Complete-shape initialization from pixels, requiring operator review.

No model state or synthetic occlusion coordinates enter this extractor. Do not
bridge separated components: partial observations belong to the edge observer.
"""
import cv2
import numpy as np
from .skeleton import extract_centerline_2d, _fix_path_endcaps


def _foreground(image, polarity):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('需要 BGR 相机图像')
    if polarity not in ('bright', 'dark'):
        raise ValueError('未知明暗方向')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flag = cv2.THRESH_BINARY if polarity == 'bright' else cv2.THRESH_BINARY_INV
    _, mask = cv2.threshold(gray, 0, 255, flag | cv2.THRESH_OTSU)
    if polarity == 'bright':
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask[(hsv[..., 1] > 135) | (hsv[..., 2] < 95)] = 0
    return mask


def _grabcut(image, mask, domain):
    """Color refinement within an operator ROI/corridor, never using model shape."""
    labels = np.full(mask.shape, cv2.GC_BGD, np.uint8)
    labels[domain > 0] = cv2.GC_PR_BGD
    labels[(mask > 0) & (domain > 0)] = cv2.GC_PR_FGD
    core = cv2.erode(mask, np.ones((3, 3), np.uint8))
    labels[(core > 0) & (domain > 0)] = cv2.GC_FGD
    if np.count_nonzero(labels == cv2.GC_FGD) < 10:
        return mask
    # Initialization may be slower than control, but bound the graph size.
    scale = min(1., 800/max(mask.shape))
    size = (max(2, round(mask.shape[1]*scale)), max(2, round(mask.shape[0]*scale)))
    pixels = cv2.resize(np.asarray(image), size)
    small = cv2.resize(labels, size, interpolation=cv2.INTER_NEAREST)
    if not (np.any(small == cv2.GC_FGD) and np.any(small == cv2.GC_BGD)):
        return mask
    cv2.grabCut(pixels, small, None, np.zeros((1,65)), np.zeros((1,65)), 3, cv2.GC_INIT_WITH_MASK)
    result = cv2.resize(np.uint8((small == cv2.GC_FGD) | (small == cv2.GC_PR_FGD))*255,
                        (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    result[domain == 0] = 0
    return result


def _local_foreground(image, domain, polarity):
    if polarity not in ('bright','dark'):raise ValueError('未知明暗方向')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    values = gray[domain > 0]
    if not len(values) or np.ptp(values) < 12:
        raise ValueError('选区内缺少图像对比，请调整选区或直接手绘')
    threshold, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mask = np.uint8((gray > threshold) if polarity == 'bright' else (gray <= threshold))*255
    mask[domain == 0] = 0
    return _grabcut(image, mask, domain)


def extract_initial_shape(image, n_nodes, polarity='bright', roi=None, supplied_mask=None):
    image = np.asarray(image)
    mask = _foreground(image, polarity) if supplied_mask is None else np.uint8(np.asarray(supplied_mask)>0)*255
    if mask.shape != image.shape[:2]:raise ValueError('分割掩膜尺寸不匹配')
    region = None
    if roi is not None:
        region = np.asarray(roi, dtype=int)
        if (region.shape != (4,) or np.any(region[:2] < 0) or
                region[2] > mask.shape[1] or region[3] > mask.shape[0] or
                np.any(region[2:]-region[:2] < 8)):
            raise ValueError('框选区域无效，请重新框选完整臂身')
        x0,y0,x1,y1 = region
        domain = np.zeros_like(mask);domain[y0:y1,x0:x1] = 255
        if supplied_mask is None:mask = _local_foreground(image, domain, polarity)
        else:mask[domain == 0] = 0
    # Remove thin wires without closing gaps across an occluder.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates = sorted(range(1, count), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
    minimum_area = max(30, mask.size*.0003)
    if not candidates or stats[candidates[0], cv2.CC_STAT_AREA] < minimum_area:
        raise ValueError('未提取到完整臂身，请检查明暗方向、背景或手动描画')
    largest = candidates[0]
    area = stats[largest, cv2.CC_STAT_AREA]
    if len(candidates) > 1 and stats[candidates[1], cv2.CC_STAT_AREA] > max(minimum_area, .15*area):
        raise ValueError('存在多个较大分离区域；请框选臂身排除背景，或直接手绘完整中心线')
    mask = np.uint8(labels == largest)*255
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        raise ValueError('分割区域连到图像边界，可能包含支架或背景；请框选臂身，或直接手绘。若臂身确实出画则需调整相机')
    curve, info = extract_centerline_2d(mask, n_points=n_nodes, endpoint_fix=True, return_info=True)
    if not info['success'] or info['arc_length_px'] < 30:
        raise ValueError('完整中心线提取失败，请检查图像或手动描画')
    warnings = []
    if not info.get('base_endpoint_fix_applied') or not info.get('tip_endpoint_fix_applied'):
        warnings.append('短边置信度不足，请拖动 BASE/TIP 到实际端面中心')
    if region is not None:
        if mask[y0:y1,x0:x0+3].any() or mask[y0:y1,x1-3:x1].any() or mask[y0:y0+3,x0:x1].any() or mask[y1-3:y1,x0:x1].any():
            warnings.append('分割触及框选边界，端点可能由框选截断；确认前必须检查完整臂身及两端')
    widths = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    xy = np.rint(curve[1:-1]).astype(int)
    radius = float(np.median(widths[xy[:, 1], xy[:, 0]]))
    if info['arc_length_px'] < 5*radius:
        raise ValueError('候选区域不像细长臂身，请检查背景或手动描画')
    return curve, mask, dict(info, radius_px=radius, warnings=warnings, roi=None if region is None else region.tolist(),
                            method='supplied_mask' if supplied_mask is not None else ('roi_otsu_grabcut' if roi is not None else 'otsu_opening'),
                            base_rule='upper endpoint; operator must review or reverse', area_px=int(area))


def refine_initial_shape(image, draft, polarity='bright', search_px=0., preserve_endpoints=False, supplied_mask=None):
    """Recenter a reviewed guide using a local color mask or supplied SAM mask.

    Long unsupported intervals are rejected. Short gaps retain the operator's
    curve and are reported. Manual endpoints need not be certified by a mask.
    The returned curve remains a draft, not a committed calibration.
    """
    from ..runtime.hereditary_deployment import resample_curve
    draft = np.asarray(draft, dtype=float)
    length = np.linalg.norm(np.diff(draft, axis=0), axis=1).sum()
    automatic_search = search_px == 0
    if automatic_search:search_px=float(np.clip(length*.15,8,240))
    if not np.isfinite(search_px) or not 4 <= search_px <= 240:
        raise ValueError('局部搜索半径必须为自动或 4..240 px')
    guide = resample_curve(draft, max(80, len(draft)))
    h, w = image.shape[:2]
    if np.any(guide < 0) or np.any(guide > [w-1, h-1]):
        raise ValueError('草稿超出图像')
    # Work only near the guide; a connected fixture at the image edge must not
    # invalidate manually identified endpoints elsewhere in the image.
    corridor = np.zeros((h,w),np.uint8)
    cv2.polylines(corridor, [np.rint(guide).astype(np.int32)], False, 255, max(1, int(2*search_px)))
    mask = _local_foreground(image, corridor, polarity) if supplied_mask is None else np.uint8(np.asarray(supplied_mask)>0)*255
    if mask.shape != (h,w):raise ValueError('分割掩膜尺寸不匹配')
    mask[corridor == 0] = 0
    if np.count_nonzero(mask) < max(20,length):raise ValueError('草稿附近没有足够臂身像素')
    widths = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    radii = widths[(mask > 0) & (corridor > 0)]
    radius = float(np.quantile(radii, .75))
    length = np.linalg.norm(np.diff(guide, axis=0), axis=1).sum()
    if length < max(30, 5*radius):
        raise ValueError('需要完整细长臂身草稿，不能用局部目标代替初始化')
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(guide, axis=0), axis=1))]
    interior = (arc > 1.5*radius) & (arc < arc[-1]-1.5*radius)
    offsets = np.arange(-search_px, search_px+.25, .5)
    current = guide.copy()
    missing_sections=set()
    for _ in range(3):
        tangent = np.gradient(current, axis=0)
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)
        updated = current.copy()
        for i in np.flatnonzero(interior):
            samples = current[i] + offsets[:, None]*normal[i]
            xy = np.rint(samples).astype(int)
            inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
            foreground = np.zeros(len(xy), dtype=bool)
            foreground[inside] = mask[xy[inside, 1], xy[inside, 0]] > 0
            edges = np.diff(np.r_[False, foreground, False].astype(int))
            runs = [(a, b-1) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
                    if a > 0 and b < len(offsets) and offsets[b-1]-offsets[a] >= max(3, radius)]
            if not runs:
                missing_sections.add(int(i));continue
            a, b = min(runs, key=lambda ab: abs((offsets[ab[0]]+offsets[ab[1]])/2))
            shift = (offsets[a]+offsets[b])/2
            updated[i] += .85*shift*normal[i]
            if np.linalg.norm(updated[i]-guide[i]) > search_px:
                raise ValueError('图像修正偏移过大，请重新检查草稿')
        current = updated
    if len(missing_sections) > max(2,int(np.count_nonzero(interior)*.08)):
        raise ValueError('草稿较长区段缺少双侧图像证据；请修正草稿、搜索半径或遮挡，原草稿仍可人工检查确认')
    if preserve_endpoints:
        current[0],current[-1]=draft[0],draft[-1]
        tip_reason=base_reason='operator_preserved'
    else:
        fixed, _, _, tip_reason, base_reason = _fix_path_endcaps(mask, current)
        if tip_reason == 'applied' and base_reason == 'applied':current=fixed
        else:
            current[0],current[-1]=draft[0],draft[-1]
            tip_reason=base_reason='operator_review_required'
    curve = resample_curve(current, len(draft))
    if np.max(np.linalg.norm(curve-draft, axis=1)) > 2*search_px:
        raise ValueError('短边修正偏移过大，请手动检查端点')
    return curve, mask, dict(radius_px=radius, search_px=float(search_px), automatic_search=automatic_search,
                            method='supplied_mask_guide_cross_sections' if supplied_mask is not None else 'guide_otsu_grabcut_cross_sections',manual_sections=sorted(missing_sections),
                            endpoints_preserved=preserve_endpoints or tip_reason=='operator_review_required',
                            max_shift_px=float(np.linalg.norm(curve-draft, axis=1).max()),
                            tip_endpoint_fix_reason=tip_reason, base_endpoint_fix_reason=base_reason,
                            arc_length_px=float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum()))
