import numpy as np
# from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.image
import os
from tqdm import trange
import torch
from torch import nn
from typing import Optional, Tuple, List, Union, Callable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
def rot_X(th: float) -> np.ndarray:
    """生成绕 X 轴旋转的 4x4 齐次矩阵。"""
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(th), -np.sin(th), 0],
        [0, np.sin(th), np.cos(th), 0],
        [0, 0, 0, 1]
    ])

def rot_Y(th: float) -> np.ndarray:
    """生成绕 Y 轴旋转的 4x4 齐次矩阵。"""
    return np.array([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1]
    ])

def rot_Z(th: float) -> np.ndarray:
    """生成绕 Z 轴旋转的 4x4 齐次矩阵。"""
    return np.array([
        [np.cos(th), -np.sin(th), 0, 0],
        [np.sin(th), np.cos(th), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def pts_trans_matrix_numpy(theta,phi,no_inverse=False):
    """使用 NumPy 构造世界/相机变换矩阵。

    Args:
        theta: 偏航角（角度制）。
        phi: 俯仰角（角度制）。
        no_inverse: 为 True 返回正向矩阵，否则返回逆矩阵。

    Returns:
        4x4 齐次变换矩阵。
    """
    # the coordinates in pybullet, camera is along X axis, but in the pts coordinates, the camera is along z axis

    w2c = transition_matrix("rot_z", -theta / 180. * np.pi)
    w2c = np.dot(transition_matrix("rot_y", -phi / 180. * np.pi), w2c)
    if no_inverse == False:
        w2c = np.linalg.inv(w2c)
    return w2c


def pts_trans_matrix(theta, phi, no_inverse=False):
    """使用 PyTorch 构造世界/相机变换矩阵。

    Args:
        theta: 偏航角（角度制）。
        phi: 俯仰角（角度制）。
        no_inverse: 为 True 返回正向矩阵，否则返回逆矩阵。

    Returns:
        4x4 的 torch 变换矩阵。
    """
    # the coordinates in pybullet, camera is along X axis,
    # but in the pts coordinates, the camera is along z axis

    w2c = transition_matrix_torch("rot_z", -theta / 180. * torch.pi)
    w2c = transition_matrix_torch("rot_y", -phi / 180. * torch.pi) @ w2c
    if not no_inverse:
        w2c = torch.inverse(w2c)
    return w2c


def rays_np(H, W, D, c_h=1.106):
    """numpy version my_ray"""
    rate = np.tan(21 * np.pi / 180)
    # co = torch.Tensor([0.8, 0, 0.606])
    #
    #               0.3          0.3        0.5
    #         far -----  object ----- near ----- camera
    #
    near = np.array([
        [[0.3, 0.5 * rate, c_h + 0.5 * rate], [0.3, -0.5 * rate, c_h + 0.5 * rate]],
        [[0.3, 0.5 * rate, c_h - 0.5 * rate], [0.3, -0.5 * rate, c_h - 0.5 * rate]]
    ])

    far = np.array([
        [[-0.3, 1.1 * rate, c_h + 1.1 * rate], [-0.3, -1.1 * rate, c_h + 1.1 * rate]],
        [[-0.3, 1.1 * rate, c_h - 1.1 * rate], [-0.3, -1.1 * rate, c_h - 1.1 * rate]]
    ])
    n_y_list = (np.linspace(near[0, 0, 1], near[0, 1, 1], W + 1) + 0.5 * (near[0, 1, 1] - near[0, 0, 1]) / W)[:-1]
    n_z_list = (np.linspace(near[0, 0, 2], near[1, 0, 2], H + 1) + 0.5 * (near[1, 0, 2] - near[0, 0, 2]) / H)[:-1]
    f_y_list = (np.linspace(far[0, 0, 1], far[0, 1, 1], W + 1) + 0.5 * (far[0, 1, 1] - far[0, 0, 1]) / W)[:-1]
    f_z_list = (np.linspace(far[0, 0, 2], far[1, 0, 2], H + 1) + 0.5 * (far[1, 0, 2] - far[0, 0, 2]) / H)[:-1]

    ny, nz = np.meshgrid(n_y_list, n_z_list)
    near_face = np.stack([0.3 * np.ones_like(ny.T), ny.T, nz.T], -1)

    fy, fz = np.meshgrid(f_y_list, f_z_list)
    far_face = np.stack([-0.3 * np.ones_like(fy.T), fy.T, fz.T], -1)
    D_list = np.linspace(0, 1, D + 1)[:-1] + .5 * (1 / D)
    box = []
    for d in D_list:
        one_face = (near_face - far_face) * d + far_face
        box.append(one_face)

    box = np.array(box)
    box = np.swapaxes(box, 0, 2)
    # box = torch.swapaxes(box, 1, 2)
    return near, far, near_face, far_face, box


def transfer_box(vbox, norm_angles, c_h=1.106, forward_flag=False):
    """在相机坐标系与机械臂坐标系之间旋转体素采样点。

    Args:
        vbox: 体素网格，形状 (H, W, D, 3)。
        norm_angles: 归一化角度。
        c_h: 相机高度偏移。
        forward_flag: True 表示静态机械臂/移动相机映射。

    Returns:
        (new_view_box, flatten_new_view_box)，分别为网格与展平点。
    """
    vb_shape = vbox.shape
    flatten_box = vbox.reshape(vb_shape[0] * vb_shape[1] * vb_shape[2], 3)
    flatten_box[:, 2] -= c_h
    full_matrix = np.dot(rot_Z(norm_angles[0] * 360 / 180 * np.pi), rot_Y(norm_angles[1] * 90 / 180 * np.pi))
    if forward_flag:
        # static arm, moving camera
        flatten_new_view_box = np.dot(
            full_matrix,
            np.hstack((flatten_box, np.ones((flatten_box.shape[0], 1)))).T
        )[:3]
    else:
        # static camera, moving arm
        flatten_new_view_box = np.dot(
            np.linalg.inv(full_matrix),
            np.hstack((flatten_box, np.ones((flatten_box.shape[0], 1)))).T
        )[:3]
    flatten_new_view_box[2] += c_h
    flatten_new_view_box = flatten_new_view_box.T
    new_view_box = flatten_new_view_box.reshape(vb_shape[0], vb_shape[1], vb_shape[2], 3)
    return new_view_box, flatten_new_view_box


def get_rays(
        height: int,
        width: int,
        focal_length: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """按项目坐标约定生成每个像素的相机射线。

    Args:
        height: 图像高度。
        width: 图像宽度。
        focal_length: 焦距张量。

    Returns:
        rays_o: 射线起点，形状 (height*width, 3)。
        rays_d: 射线方向，形状 (height*width, 3)。
    """
    # Find origin and direction of rays through every pixel and camera origin.

    # Apply pinhole camera model to gather directions at each pixel
    # i, j = torch.meshgrid(
    #     torch.arange(width, dtype=torch.float32).to(focal_length),
    #     torch.arange(height, dtype=torch.float32).to(focal_length),
    #     indexing='ij')

    # debug jiong @ Aug 26, to(focal_length) is to focal_length's device, transfered again after get_rays function
    
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32),
        torch.arange(height, dtype=torch.float32),
        indexing='ij')

    directions = torch.stack([(i - width * .5) / focal_length,
                              -(j - height * .5) / focal_length,
                              -torch.ones_like(i)
                              ], dim=-1)
    # directions: tan_i, tan_j, -1

    # Apply camera pose to directions
    rays_d = directions
    rays_o = torch.from_numpy(np.asarray([1,0,0],dtype=np.float32)).expand(directions.shape)

    rays_d_clone = rays_d.clone()
    rays_d[..., 0], rays_d[..., 2] = rays_d_clone[..., 2].clone(), rays_d_clone[..., 0].clone()

    # Origin is same for all directions (the optical center)
    rotation_matrix = torch.tensor([[1, 0, 0],
                                    [0, -1, 0],
                                    [0, 0, -1]])
    rotation_matrix = rotation_matrix[None, None].to(rays_d)

    # Rotate the points
    rays_d = torch.matmul(rays_d, rotation_matrix)
    rays_o = rays_o.reshape(-1,3)
    rays_d = rays_d.reshape(-1,3)
    return rays_o, rays_d

def sample_stratified(
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        arm_angle: torch.Tensor,
        near: float,
        far: float,
        n_samples: int,
        perturb: Optional[bool] = True,
        inverse_depth: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:

    """沿射线均匀（或逆深度）采样三维点。

    Args:
        rays_o: 射线起点 (N_rays, 3)。
        rays_d: 射线方向 (N_rays, 3)。
        arm_angle: 预留参数（当前未使用）。
        near/far: 采样近远边界。
        n_samples: 每条射线采样数量。
        perturb: 是否进行分层抖动采样。
        inverse_depth: 是否按逆深度线性采样。

    Returns:
        pts: 采样点，形状 (N_rays, n_samples, 3)。
        x_vals: 对应深度，形状 (N_rays, n_samples)。
    """

    # Grab samples for space integration along ray
    t_vals = torch.linspace(0., 1., n_samples, device=rays_o.device)
    if not inverse_depth:
        # Sample linearly between `near` and `far`
        x_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity)
        x_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    # Draw uniform samples from bins along ray
    if perturb:
        mids = .5 * (x_vals[1:] + x_vals[:-1])
        upper = torch.concat([mids, x_vals[-1:]], dim=-1)
        lower = torch.concat([x_vals[:1], mids], dim=-1)
        t_rand = torch.rand([n_samples], device=x_vals.device)
        x_vals = lower + (upper - lower) * t_rand
    x_vals = x_vals.expand(list(rays_o.shape[:-1]) + [n_samples])


    pts = rays_o[..., None, :] + rays_d[..., None, :] * x_vals[..., :, None]

    # 下面是针对特定机械臂的硬编码优化
    # pose_matrix = pts_trans_matrix(arm_angle[0], arm_angle[1])
    #
    # pose_matrix = pose_matrix.to(pts)
    # # Transpose your transformation matrix for correct matrix multiplication
    # transformation_matrix = pose_matrix[:3,:3]
    #
    # # Apply the transformation
    # pts = torch.matmul(pts,transformation_matrix)

    return pts, x_vals



"""
volume rendering
"""
def VR_rendering(
        raw: torch.Tensor,
        x_vals: torch.Tensor,
        rays_d: torch.Tensor,
        raw_noise_std: float = 0.0,
        white_bkgd: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """基于原始密度输出的简单体渲染累积。

    Args:
        raw: 网络原始输出，形状通常为 (N_rays, N_samples, C)。
        x_vals: 采样深度（为接口兼容保留）。
        rays_d: 射线方向（为接口兼容保留）。
        raw_noise_std: 预留噪声参数。
        white_bkgd: 预留背景参数。

    Returns:
        render_img: 累积结果，形状 (N_rays,)。
        dense: 每采样点不透明度，形状 (N_rays, N_samples)。
    """

    dense = 1.0 - torch.exp(-nn.functional.relu(raw[..., 0]))

    render_img = torch.sum(dense, dim=1)

    return render_img, dense

def VRAT_rendering(
        raw: torch.Tensor,
        x_vals: torch.Tensor,
        rays_d: torch.Tensor,
        raw_noise_std: float = 0.0,
        white_bkgd: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """带距离项的体渲染 alpha 计算。

    Args:
        raw: 网络原始输出。
        x_vals: 每条射线上的采样深度。
        rays_d: 用于距离缩放的射线方向。
        raw_noise_std: 预留噪声参数。
        white_bkgd: 预留背景参数。

    Returns:
        render_img: 累积 alpha，形状 (N_rays,)。
        alpha_dense: 每采样点 alpha，形状 (N_rays, N_samples)。
    """

    dists = x_vals[..., 1:] - x_vals[..., :-1]

    # add one elements for each ray to compensate the size to 64
    dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1).to(device)
    rays_d = rays_d.to(device)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    alpha_dense = 1.0 - torch.exp(-nn.functional.relu(raw[..., 0]) * dists)

    render_img = torch.sum(alpha_dense, dim=1)

    return render_img, alpha_dense

def OM_rendering(
        raw: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    """NeRF-PINN 使用的不透明度调制渲染。

    Args:
        raw: 解码器输出，形状 (N_rays, N_samples, 2)，通道为 [可见性, 密度]。

    Returns:
        render_img: 累积像素值，形状 (N_rays,)。
        alpha: 每采样点权重，形状 (N_rays, N_samples)。
    """

    alpha = 1.0 - torch.exp(-nn.functional.relu(raw[..., 1]))
    rgb_each_point = alpha * raw[..., 0]
    render_img = torch.sum(rgb_each_point, dim=1)

    return render_img, alpha

def OM_rendering_split_output(raw):
    """输出可见性分量的不透明度调制渲染。

    Args:
        raw: 解码器输出，形状 (N_rays, N_samples, 2)，
            通道为 [可见性, 密度]。

    Returns:
        render_img: 渲染结果。
        alpha: alpha 权重。
        visibility: 原始可见性通道。
    """
    alpha = 1.0 - torch.exp(-nn.functional.relu(raw[..., 1]))
    rgb_each_point = alpha*raw[..., 0]
    render_img = torch.sum(rgb_each_point, dim=1)
    visibility = raw[..., 0]
    return render_img, alpha, visibility

def sample_pdf(
        bins: torch.Tensor,
        weights: torch.Tensor,
        n_samples: int,
        perturb: bool = False
) -> torch.Tensor:
    r"""基于权重分布执行逆变换采样。

    Args:
        bins: 采样分箱中心或边界，形状 (N_rays, N_bins)。
        weights: 非负权重，形状 (N_rays, N_bins)。
        n_samples: 每条射线新增采样数。
        perturb: True 为随机采样，False 为确定性采样。

    Returns:
        重新采样深度，形状 (N_rays, n_samples)。
    """

    # Normalize weights to get PDF.
    pdf = (weights + 1e-5) / torch.sum(weights + 1e-5, -1, keepdims=True)  # [n_rays, weights.shape[-1]]

    # Convert PDF to CDF.
    cdf = torch.cumsum(pdf, dim=-1)  # [n_rays, weights.shape[-1]]
    cdf = torch.concat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # [n_rays, weights.shape[-1] + 1]

    # Take sample positions to grab from CDF. Linear when perturb == 0.
    if not perturb:
        u = torch.linspace(0., 1., n_samples, device=cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])  # [n_rays, n_samples]
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=cdf.device)  # [n_rays, n_samples]

    # Find indices along CDF where values in u would be placed.
    u = u.contiguous()  # Returns contiguous tensor with same values.
    inds = torch.searchsorted(cdf, u, right=True)  # [n_rays, n_samples]

    # Clamp indices that are out of bounds.
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], dim=-1)  # [n_rays, n_samples, 2]

    # Sample from cdf and the corresponding bin centers.
    matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1,
                         index=inds_g)
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), dim=-1,
                          index=inds_g)

    # Convert samples to ray length.
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples  # [n_rays, n_samples]


def sample_hierarchical(
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        z_vals: torch.Tensor,
        weights: torch.Tensor,
        n_samples: int,
        perturb: bool = False,
        angle: float = 1.,
        more_dof: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""执行分层重采样并重建查询点。

    Args:
        rays_o: 射线起点。
        rays_d: 射线方向。
        z_vals: 粗采样深度。
        weights: 粗采样权重。
        n_samples: 细采样数量。
        perturb: 是否随机采样。
        angle: `more_dof` 开启时附加的角度值。
        more_dof: 是否在点坐标后附加额外角度通道。

    Returns:
        pts: 合并后的采样点。
        z_vals_combined: 合并并排序后的深度。
        new_z_samples: 新增细采样深度。
    """

    # Draw samples from PDF using z_vals as bins and weights as probabilities.
    z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    new_z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], n_samples,
                               perturb=perturb)
    new_z_samples = new_z_samples.detach()

    # Resample points from ray based on PDF.
    z_vals_combined, _ = torch.sort(torch.cat([z_vals, new_z_samples], dim=-1), dim=-1)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :,
                                                        None]  # [N_rays, N_samples + n_samples, 3]
    if more_dof:
        # for 3dof arm
        add_angle = torch.ones(pts.shape[0], pts.shape[1], 1).to(device) * angle
        pts = torch.cat((pts, add_angle), 2)
        # print(pts.shape)
    return pts, z_vals_combined, new_z_samples


def prepare_chunks(
        points: torch.Tensor,
        chunksize: int = 2 ** 14
) -> List[torch.Tensor]:
    """将查询点拆分为小块，避免推理显存溢出。

    Args:
        points: 输入点张量，最后一维为特征维。
        chunksize: 每块最大点数。

    Returns:
        分块后的张量列表。
    """

    points = points.reshape((-1, points.shape[-1]))
    points = [points[i:i + chunksize] for i in range(0, points.shape[0], chunksize)]
    return points


def model_forward(
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: float,
        far: float,
        model: nn.Module,
        arm_angle: torch.Tensor,
        DOF: int,
        chunksize: int = 2 ** 15,
        n_samples: int = 64,
        output_flag: int = 0
):
    """执行从射线到渲染结果的完整 NeRF 风格前向。

    Args:
        rays_o: 射线起点。
        rays_d: 射线方向。
        near: 近裁剪面。
        far: 远裁剪面。
        model: 场函数网络。
        arm_angle: 执行器控制量。
        DOF: 使用的自由度数量。
        chunksize: 网络查询分块大小。
        n_samples: 每条射线采样数量。
        output_flag: 渲染模式标志位。

    Returns:
        默认返回包含 `rgb_map`、`rgb_each_point`、`query_points` 的字典。
        当 `output_flag == 3` 时返回元组
        `(rgb_map, query_points, rgb_each_point, visibility)`。
    """

    # Sample query points along each ray.
    query_points, z_vals = sample_stratified(
        rays_o, rays_d, arm_angle, near, far, n_samples=n_samples)
    # Prepare batches.
    # 归一化
    # arm_angle = arm_angle / 180 * np.pi

    # 原始写法 (只取下标2以后的参数)
    # if DOF > 2:
    #     model_input = torch.cat((query_points, arm_angle[2:DOF].repeat(list(query_points.shape[:2]) + [1])), dim=-1)
    #
    # # arm_angle[:DOF] -> use one angle
    # else:
    #     model_input = query_points  # orig version 3 input 2dof, Mar30

    # ===> 修改为 (使用所有参数):
    cmd_input = arm_angle.expand(list(query_points.shape[:2]) + [DOF])
    model_input = torch.cat((query_points, cmd_input), dim=-1)


    batches = prepare_chunks(model_input, chunksize=chunksize)
    predictions = []
    for batch in batches:
        batch = batch.to(device)
        predictions.append(model(batch))
    raw = torch.cat(predictions, dim=0)
    raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

    if output_flag ==0:
        rgb_map, rgb_each_point = OM_rendering(raw)
    elif output_flag ==1:
        rgb_map, rgb_each_point = VR_rendering(raw, z_vals, rays_d)
    elif output_flag ==2:
        rgb_map, rgb_each_point = VRAT_rendering(raw, z_vals, rays_d)
    elif output_flag ==3:
        rgb_map,rgb_each_point, visibility = OM_rendering_split_output(raw)
        return rgb_map, query_points, rgb_each_point, visibility


    outputs = {
        'rgb_map': rgb_map,
        'rgb_each_point': rgb_each_point,
        'query_points': query_points}

    # Store outputs.
    return outputs


# ---------------------------------------------------------
# Transformation Matrices for 3D Space
# ---------------------------------------------------------
def transition_matrix(label: str, value: float) -> np.ndarray:
    """Returns a 4x4 transformation matrix for rotation in 3D space."""
    if label == "rot_x":
        return rot_X(value)
    elif label == "rot_y":
        return rot_Y(value)
    elif label == "rot_z":
        return rot_Z(value)
    else:
        raise ValueError("Invalid label. Use 'rot_x', 'rot_y', or 'rot_z'.")


def transition_matrix_torch(label: str, value: torch.Tensor) -> torch.Tensor:
    """Returns a 4x4 transformation matrix for rotation in 3D space using PyTorch tensors."""
    matrix = torch.eye(4, dtype=torch.float32)

    if label == "rot_x":
        matrix[1, 1] = torch.cos(value)
        matrix[1, 2] = -torch.sin(value)
        matrix[2, 1] = torch.sin(value)
        matrix[2, 2] = torch.cos(value)
    elif label == "rot_y":
        matrix[0, 0] = torch.cos(value)
        matrix[0, 2] = -torch.sin(value)
        matrix[2, 0] = torch.sin(value)
        matrix[2, 2] = torch.cos(value)
    elif label == "rot_z":
        matrix[0, 0] = torch.cos(value)
        matrix[0, 1] = -torch.sin(value)
        matrix[1, 0] = torch.sin(value)
        matrix[1, 1] = torch.cos(value)
    else:
        raise ValueError("Invalid label. Use 'rot_x', 'rot_y', or 'rot_z'.")

    return matrix

def plot_3d_visual(x, y, z, if_transform=True):
    """快速三维散点可视化工具。

    Args:
        x, y, z: 坐标（张量或 ndarray）。
        if_transform: 是否先把张量转换为 NumPy。
    """
    if if_transform:
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        z = z.detach().cpu().numpy()

    ax = plt.axes(projection='3d')
    ax.scatter3D(x,
                 y,
                 z, s=1
                 )
    # ax.scatter3D(0,0,0)


if __name__ == "__main__":

    #              -0.4    0      0.4        0.6
    #         far -----  object ----- near ----- camera
    #

    DOF = 2  # the number of motors  # dof4 apr03
    num_data = 20**DOF
    pxs = 100  # collected data pixels

    HEIGHT = pxs
    WIDTH = pxs
    nf_size = 0.4
    cam_dist = 1
    camera_angle_x = 42 * np.pi / 180.
    focal = .5 * WIDTH / np.tan(.5 * camera_angle_x)
    rays_o, rays_d = get_rays(HEIGHT, WIDTH, focal)

    # Visualization
    ax = plt.figure().add_subplot(projection='3d')
    rays_o = rays_o.detach().cpu().numpy()
    rays_d = rays_d.detach().cpu().numpy()
    rays_d = rays_d[:10]
    idx = np.random.choice(list(np.arange(len(rays_d))),size = 1000)
    rays_d = rays_d[idx]
    rays_o = rays_o[idx]
    for plt_i in range(len(rays_d)):
        ax.plot3D([rays_d[plt_i, 0],0],
                  [rays_d[plt_i, 1],0],
                  [rays_d[plt_i, 2],0])
    print(rays_o)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.scatter(rays_o[0][0],rays_o[0][1],rays_o[0][2])
    plt.show()
    quit()

    data = np.load('data/sim_data/sim_data_robo0(arm).npz' )

    training_angles = torch.from_numpy(data['angles'].astype('float32'))
    training_pose_matrix = torch.from_numpy(data['poses'].astype('float32'))

    idxx = 265
    angle = training_angles[idxx]
    print(angle/90)
    pose_matrix = pts_trans_matrix(angle[0],angle[1])

    near, far = cam_dist - nf_size, cam_dist + nf_size
    kwargs_sample_stratified = {
        'n_samples': 64,
        'perturb': True,
        'inverse_depth': False
    }

    rays_o = rays_o.reshape([-1, 3])
    rays_d = rays_d.reshape([-1, 3])
    query_points, z_vals = sample_stratified(
        rays_o, rays_d, angle, near, far, **kwargs_sample_stratified)

