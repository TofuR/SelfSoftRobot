"""Sparse side-edge measurements from an RGB frame and a predicted centerline.

The detector has no reference-skeleton input. Synthetic visibility masks may
only be supplied in the explicitly named oracle-visibility experiment.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def project_camera(shape_mm, matrix):
    import torch
    homogeneous = torch.cat([shape_mm, torch.ones_like(shape_mm[:, :1])], dim=-1)
    projected = homogeneous @ matrix.T
    return projected[:, :2] / projected[:, 2:3]


@dataclass(frozen=True)
class EdgeEvidence:
    pixels: np.ndarray
    segments: np.ndarray
    strengths: np.ndarray

    def residual(self, centerline, radius, sigma=2.0):
        import torch
        points = torch.as_tensor(self.pixels, dtype=centerline.dtype, device=centerline.device)
        indices = torch.as_tensor(self.segments, dtype=torch.long, device=centerline.device)
        start, end = centerline[indices], centerline[indices+1]
        vector = end-start
        ratio = ((points-start)*vector).sum(-1) / vector.square().sum(-1).clamp_min(1e-8)
        closest = start + ratio.clamp(0,1)[:,None]*vector
        return (torch.linalg.vector_norm(points-closest, dim=-1)-radius)/sigma


def extract_edges(image_bgr, predicted_px, *, radius=10.0, search=12,
                  oracle_hidden=None):
    """Find white-arm to darker-background transitions along predicted side normals.

    Two cross sections per segment, no joining components across missing regions.
    Search/rejection parameters are development settings for the current imagery.
    """
    gray = cv2.GaussianBlur(cv2.cvtColor(image_bgr,cv2.COLOR_BGR2GRAY),(3,3),0).astype(float)
    hsv = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2HSV)
    gx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)/8
    gy = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)/8
    height,width = gray.shape
    points,segments,strengths = [],[],[]
    for i in range(1,len(predicted_px)-2):  # exclude base attachment and terminal cap
        a,b = predicted_px[i:i+2]
        tangent = b-a
        length = np.linalg.norm(tangent)
        if length < 2:
            continue
        normal = np.array([-tangent[1],tangent[0]])/length
        for t in (0.25,0.75):
            center = a+t*tangent
            for side in (-1,1):
                outward = side*normal
                candidates=[]
                for offset in range(-search,search+1):
                    # A side must remain on its assigned half of the predicted tube.
                    if radius+offset < 2:
                        continue
                    q = center+(radius+offset)*outward
                    x,y = np.rint(q).astype(int)
                    xi,yi = np.rint(q-3*outward).astype(int)
                    xo,yo = np.rint(q+3*outward).astype(int)
                    if not (3 <= min(x,xi,xo) and max(x,xi,xo) < width-3 and
                            3 <= min(y,yi,yo) and max(y,yi,yo) < height-3):
                        continue
                    if oracle_hidden is not None and (oracle_hidden[y,x] or oracle_hidden[yi,xi] or oracle_hidden[yo,xo]):
                        continue
                    contrast = gray[yi,xi]-gray[yo,xo]
                    normal_gradient = -(gx[y,x]*outward[0]+gy[y,x]*outward[1])
                    grad_length = np.hypot(gx[y,x],gy[y,x])
                    if (contrast < 18 or normal_gradient < 6 or
                            normal_gradient < 0.65*grad_length or
                            hsv[yi,xi,1] > 135 or hsv[yi,xi,2] < 95):
                        continue
                    score = normal_gradient/(1+0.08*abs(offset))
                    candidates.append((score,offset,q))
                if not candidates:
                    continue
                candidates.sort(key=lambda item:item[0],reverse=True)
                best = candidates[0]
                if any(abs(c[1]-best[1])>4 and c[0]>0.9*best[0] for c in candidates[1:]):
                    continue
                if any(np.linalg.norm(best[2]-p)<3 for p in points):
                    continue
                points.append(best[2]); segments.append(i); strengths.append(best[0])
    return EdgeEvidence(np.asarray(points,dtype=np.float32).reshape(-1,2),
                        np.asarray(segments,dtype=np.int64),np.asarray(strengths,dtype=np.float32))


def extract_edges_vectorized(image_bgr, predicted_px, *, radius=10., search=12,
                             oracle_hidden=None):
    """Same detector decisions, batch all candidate pixel/gradient queries.

    Association order, stable score tie-breaking and greedy deduplication are
    preserved so this can be compared directly to the original nested loop.
    """
    gray = cv2.GaussianBlur(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(float)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)/8
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)/8
    height, width = gray.shape
    centers, normals, segment_ids = [], [], []
    for i in range(1, len(predicted_px)-2):
        a, b = predicted_px[i:i+2]
        tangent = b-a
        length = np.linalg.norm(tangent)
        if length < 2:
            continue
        normal = np.array([-tangent[1], tangent[0]])/length
        for t in (.25, .75):
            for side in (-1, 1):
                centers.append(a+t*tangent)
                normals.append(side*normal)
                segment_ids.append(i)
    if not centers:
        return EdgeEvidence(np.empty((0, 2), np.float32), np.empty(0, np.int64), np.empty(0, np.float32))
    centers, normals = np.asarray(centers), np.asarray(normals)
    offsets = np.arange(-search, search+1)
    # Preserve the scalar detector's float32 coordinate arithmetic. An int64
    # offset array would otherwise promote this product to float64 and change
    # rint decisions for points lying at half-pixel boundaries.
    distances = np.asarray(radius, dtype=centers.dtype)+offsets.astype(centers.dtype)
    q = centers[:, None]+distances[None, :, None]*normals[:, None]
    integer = np.rint(q).astype(int)
    inside = np.rint(q-3*normals[:, None]).astype(int)
    outside = np.rint(q+3*normals[:, None]).astype(int)
    valid = np.broadcast_to(radius+offsets >= 2, q.shape[:2]).copy()
    for array in (integer, inside, outside):
        valid &= (array[..., 0] >= 3) & (array[..., 0] < width-3)
        valid &= (array[..., 1] >= 3) & (array[..., 1] < height-3)
    integer = np.clip(integer, [0, 0], [width-1, height-1])
    inside = np.clip(inside, [0, 0], [width-1, height-1])
    outside = np.clip(outside, [0, 0], [width-1, height-1])
    x, y = integer[..., 0], integer[..., 1]
    xi, yi = inside[..., 0], inside[..., 1]
    xo, yo = outside[..., 0], outside[..., 1]
    contrast = gray[yi, xi]-gray[yo, xo]
    gradient = -(gx[y, x]*normals[:, 0, None]+gy[y, x]*normals[:, 1, None])
    valid &= (contrast >= 18) & (gradient >= 6) & (gradient >= .65*np.hypot(gx[y, x], gy[y, x]))
    valid &= (hsv[yi, xi, 1] <= 135) & (hsv[yi, xi, 2] >= 95)
    if oracle_hidden is not None:
        valid &= ~(oracle_hidden[y, x] | oracle_hidden[yi, xi] | oracle_hidden[yo, xo])
    scores = gradient/(1+.08*abs(offsets))
    scores[~valid] = -np.inf
    points, segments, strengths = [], [], []
    for i in range(len(centers)):
        order = np.argsort(-scores[i], kind="stable")
        best = order[0]
        if not np.isfinite(scores[i, best]):
            continue
        ambiguous = (abs(offsets-offsets[best]) > 4) & (scores[i] > .9*scores[i, best])
        if ambiguous.any() or any(np.linalg.norm(q[i, best]-p) < 3 for p in points):
            continue
        points.append(q[i, best])
        segments.append(segment_ids[i])
        strengths.append(scores[i, best])
    return EdgeEvidence(np.asarray(points, np.float32).reshape(-1, 2),
                        np.asarray(segments, np.int64), np.asarray(strengths, np.float32))
