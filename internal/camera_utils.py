# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Camera pose and ray generation utility functions."""

import enum
import types
from typing import List, Mapping, Optional, Text, Tuple, Union

from internal import configs
from internal import math
from internal import stepfun
from internal import utils
import jax.numpy as jnp
import numpy as np
import scipy

_Array = Union[np.ndarray, jnp.ndarray]


def convert_to_ndc(origins: _Array,  # [num_patches, _patch_size, _patch_size, 3(xyz)]
                   directions: _Array,  # [num_patches, _patch_size, _patch_size, 3(dir)]
                   pixtocam: _Array,
                   near: float = 1.,
                   xnp: types.ModuleType = np) -> Tuple[_Array, _Array]:
    """Converts a set of rays to normalized device coordinates (NDC).

  Args:
    origins: ndarray(float32), [..., 3], world space ray origins.
    directions: ndarray(float32), [..., 3], world space ray directions. dx or dy
    pixtocam: ndarray(float32), [3, 3], inverse intrinsic matrix. 但投影结果是在[1,1,1]的标准坐标系下
    near: float, near plane along the negative z axis.
    xnp: either numpy or jax.numpy.

  Returns:
    origins_ndc: ndarray(float32), [..., 3].
    directions_ndc: ndarray(float32), [..., 3].

  This function assumes input rays should be mapped into the NDC space for a
  perspective projection pinhole camera, with identity extrinsic matrix (pose)
  and intrinsic parameters defined by inputs focal, width, and height.

  The near value specifies(指定) the near plane of the frustum, and the far plane is
  assumed to be infinity.

  The ray bundle for the identity pose camera will be remapped to parallel rays
  within the (-1, -1, -1) to (1, 1, 1) cube. Any other ray in the original
  world space can be remapped as long as it has dz < 0 (ray direction has a
  negative z-coord); this allows us to share a common NDC space for "forward
  facing" scenes.

  Note that
      projection(origins + t * directions) o+td
  will NOT be equal to
      origins_ndc + t * directions_ndc
  and that the directions_ndc are not unit length. Rather, directions_ndc is
  defined such that the valid near and far planes in NDC will be 0 and 1.

  See Appendix C in https://arxiv.org/abs/2003.08934 for additional details.
  """

    # Shift ray origins to near plane, such that oz = -near.
    # This makes the new near bound equal to 0.
    # [num_patches, _patch_size, _patch_size]
    t = -(near + origins[..., 2]) / directions[..., 2]  # 以dx or dy 的z轴作为世界单位z，消除直边斜边的差异
    origins = origins + t[..., None] * directions

    # 改变坐标轴的顺序 3(dir)移到最前面
    # [num_patches, _patch_size, _patch_size]
    dx, dy, dz = xnp.moveaxis(directions, -1, 0)
    ox, oy, oz = xnp.moveaxis(origins, -1, 0)

    xmult = 1. / pixtocam[0, 2]  # Equal to -2. * focal / cx
    ymult = 1. / pixtocam[1, 2]  # Equal to -2. * focal / cy

    # Perspective projection into NDC for the t = 0 near points
    #     origins + 0 * directions
    # 投影到NDC后的原点坐标
    # [num_patches, _patch_size, _patch_size, 3(xyz)]
    origins_ndc = xnp.stack([xmult * ox / oz, ymult * oy / oz, -xnp.ones_like(oz)], axis=-1)

    # Perspective projection into NDC for the t = infinity far points
    #     origins + infinity * directions
    # dz方向上任意远的点投影到NDC后的坐标
    # [num_patches, _patch_size, _patch_size, 3(xyz)]
    infinity_ndc = np.stack([xmult * dx / dz, ymult * dy / dz, xnp.ones_like(oz)], axis=-1)

    # directions_ndc points from origins_ndc to infinity_ndc
    # [num_patches, _patch_size, _patch_size, 3(dir)]
    directions_ndc = infinity_ndc - origins_ndc

    return origins_ndc, directions_ndc


def pad_poses(p: np.ndarray) -> np.ndarray:
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p: np.ndarray) -> np.ndarray:
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def recenter_poses(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Recenter poses around the origin."""
    cam2world = average_pose(poses)
    transform = np.linalg.inv(pad_poses(cam2world))
    poses = transform @ pad_poses(poses)
    return unpad_poses(poses), transform


def average_pose(poses: np.ndarray) -> np.ndarray:
    """New pose using average position, z-axis, and up vector of input poses."""
    position = poses[:, :3, 3].mean(0)
    z_axis = poses[:, :3, 2].mean(0)
    up = poses[:, :3, 1].mean(0)
    cam2world = viewmatrix(z_axis, up, position)
    return cam2world


def viewmatrix(lookdir: np.ndarray, up: np.ndarray,
               position: np.ndarray) -> np.ndarray:
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalization helper function."""
    return x / np.linalg.norm(x)


def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


# Constants for generate_spiral_path():
NEAR_STRETCH = .9  # Push forward near bound for forward facing render path.
FAR_STRETCH = 5.  # Push back far bound for forward facing render path.
FOCUS_DISTANCE = .75  # Relative weighting of near, far bounds for render path.


def generate_spiral_path(poses: np.ndarray,
                         bounds: np.ndarray,
                         n_frames: int = 120,
                         n_rots: int = 2,
                         zrate: float = .5) -> np.ndarray:
    """Calculates a forward facing spiral path for rendering."""
    # Find a reasonable 'focus depth' for this dataset as a weighted average
    # of conservative near and far bounds in disparity space.
    near_bound = bounds.min() * NEAR_STRETCH
    far_bound = bounds.max() * FAR_STRETCH
    # All cameras will point towards the world space point (0, 0, -focal).
    focal = 1 / (((1 - FOCUS_DISTANCE) / near_bound + FOCUS_DISTANCE / far_bound))

    # Get radii for spiral path using 90th percentile of camera positions.
    positions = poses[:, :3, 3]
    radii = np.percentile(np.abs(positions), 90, 0)
    radii = np.concatenate([radii, [1.]])

    # Generate poses for spiral path.
    render_poses = []
    cam2world = average_pose(poses)
    up = poses[:, :3, 1].mean(0)
    for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
        t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]
        position = cam2world @ t
        lookat = cam2world @ [0, 0, -focal, 1.]
        z_axis = position - lookat
        render_poses.append(viewmatrix(z_axis, up, position))
    render_poses = np.stack(render_poses, axis=0)
    return render_poses


def transform_poses_pca(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
  """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    return poses_recentered, transform


def generate_ellipse_path(poses: np.ndarray,
                          n_frames: int = 120,
                          const_speed: bool = True,
                          z_variation: float = 0.,
                          z_phase: float = 0.) -> np.ndarray:
    """Generate an elliptical render path based on the given poses."""
    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0], center[1], 0])

    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack([
            low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5),
            low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5),
            z_variation * (z_low[2] + (z_high - z_low)[2] *
                           (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
        ], -1)

    theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)

    if const_speed:
        # Resample theta angles so that the velocity is closer to constant.
        lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
        theta = stepfun.sample(None, theta, np.log(lengths), n_frames + 1)
        positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return np.stack([viewmatrix(p - center, up, p) for p in positions])


def generate_interpolated_path(poses: np.ndarray,
                               n_interp: int,
                               spline_degree: int = 5,
                               smoothness: float = .03,
                               rot_weight: float = .1):
    """Creates a smooth spline path between input keyframe camera poses.

  Spline is calculated with poses in format (position, lookat-point, up-point).

  Args:
    poses: (n, 3, 4) array of input pose keyframes.
    n_interp: returned path will have n_interp * (n - 1) total poses.
    spline_degree: polynomial degree of B-spline.
    smoothness: parameter for spline smoothing, 0 forces exact interpolation.
    rot_weight: relative weighting of rotation/translation in spline solve.

  Returns:
    Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
  """

    def poses_to_points(poses, dist):
        """Converts from pose matrices to (position, lookat, up) format."""
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(points):
        """Converts from (position, lookat, up) format to pose matrices."""
        return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])

    def interp(points, n, k, s):
        """Runs multidimensional B-spline interpolation on the input points."""
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
        return new_points

    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(points,
                        n_interp * (points.shape[0] - 1),
                        k=spline_degree,
                        s=smoothness)
    return points_to_poses(new_points)


def interpolate_1d(x: np.ndarray,
                   n_interp: int,
                   spline_degree: int,
                   smoothness: float) -> np.ndarray:
    """Interpolate 1d signal x (by a factor of n_interp times)."""
    t = np.linspace(0, 1, len(x), endpoint=True)
    tck = scipy.interpolate.splrep(t, x, s=smoothness, k=spline_degree)
    n = n_interp * (len(x) - 1)
    u = np.linspace(0, 1, n, endpoint=False)
    return scipy.interpolate.splev(u, tck)


def create_render_spline_path(
        config: configs.Config,
        image_names: Union[Text, List[Text]],
        poses: np.ndarray,
        exposures: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Creates spline interpolation render path from subset of dataset poses.

  Args:
    config: configs.Config object.
    image_names: either a directory of images or a text file of image names.
    poses: [N, 3, 4] array of extrinsic camera pose matrices.
    exposures: optional list of floating point exposure values.

  Returns:
    spline_indices: list of indices used to select spline keyframe poses.
    render_poses: array of interpolated extrinsic camera poses for the path.
    render_exposures: optional list of interpolated exposures for the path.
  """
    if utils.isdir(config.render_spline_keyframes):
        # If directory, use image filenames.
        keyframe_names = sorted(utils.listdir(config.render_spline_keyframes))
    else:
        # If text file, treat each line as an image filename.
        with utils.open_file(config.render_spline_keyframes, 'r') as fp:
            # Decode bytes into string and split into lines.
            keyframe_names = fp.read().decode('utf-8').splitlines()
    # Grab poses corresponding to the image filenames.
    spline_indices = np.array(
        [i for i, n in enumerate(image_names) if n in keyframe_names])
    keyframes = poses[spline_indices]
    render_poses = generate_interpolated_path(
        keyframes,
        n_interp=config.render_spline_n_interp,
        spline_degree=config.render_spline_degree,
        smoothness=config.render_spline_smoothness,
        rot_weight=.1)
    if config.render_spline_interpolate_exposure:
        if exposures is None:
            raise ValueError('config.render_spline_interpolate_exposure is True but '
                             'create_render_spline_path() was passed exposures=None.')
        # Interpolate per-frame exposure value.
        log_exposure = np.log(exposures[spline_indices])
        # Use aggressive smoothing for exposure interpolation to avoid flickering.
        log_exposure_interp = interpolate_1d(
            log_exposure,
            config.render_spline_n_interp,
            spline_degree=5,
            smoothness=20)
        render_exposures = np.exp(log_exposure_interp)
    else:
        render_exposures = None
    return spline_indices, render_poses, render_exposures


# 返回内参数矩阵
def intrinsic_matrix(fx: float,
                     fy: float,
                     cx: float,
                     cy: float,
                     xnp: types.ModuleType = np) -> _Array:
    """Intrinsic matrix for a pinhole camera in OpenCV coordinate system."""
    return xnp.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1.],
    ])


def get_pixtocam(focal: float,
                 width: float,
                 height: float,
                 xnp: types.ModuleType = np) -> _Array:
    """Inverse intrinsic matrix for a perfect pinhole camera."""
    camtopix = intrinsic_matrix(focal, focal, width * .5, height * .5, xnp)
    return xnp.linalg.inv(camtopix)  # 对内参数矩阵求逆 [3,3]


def pixel_coordinates(width: int,
                      height: int,
                      xnp: types.ModuleType = np) -> Tuple[_Array, _Array]:
    """Tuple of the x and y integer coordinates for a grid of pixels."""
    return xnp.meshgrid(xnp.arange(width), xnp.arange(height), indexing='xy')  # [h, w]


def _compute_residual_and_jacobian(  # 计算残差(观察值与拟合值的差)和雅可比矩阵
        x: _Array,
        y: _Array,
        xd: _Array,
        yd: _Array,
        k1: float = 0.0,
        k2: float = 0.0,
        k3: float = 0.0,
        k4: float = 0.0,
        p1: float = 0.0,
        p2: float = 0.0,
) -> Tuple[_Array, _Array, _Array, _Array, _Array, _Array]:
    """Auxiliary function of radial_and_tangential_undistort()."""
    # Adapted from https://github.com/google/nerfies/blob/main/nerfies/camera.py
    # 径向畸变 x_distorted: d(x,y), y_distorted: d(x,y)
    # 径向畸变矫正 x_corrected: xd(x,y), y_corrected: yd(x,y)
    # let r(x, y) = x^2 + y^2;
    # d(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3 + k4 * r(x, y)^4;
    r = x * x + y * y
    d = 1.0 + r * (k1 + r * (k2 + r * (k3 + r * k4)))

    # 切向畸变 x_distorted:2*p1*x*y+p2*(r(x,y)+2*x^2), y_distorted:2*p2*xy+p1*(r(x,y)+2*y^2)
    # 切向畸变矫正 x_corrected=x+x_distorted, y_corrected=y+y_distorted
    # The perfect projection is 对两种畸变同时进行矫正:
    # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
    # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);
    # 对应的畸变参数：k1,k2,k3,k4,p1,p2
    #
    # 循环迭代计算残差
    # Let's define
    # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
    # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;
    #
    # 希望残差为0
    # We are looking for a solution that satisfies
    # fx(x, y) = fy(x, y) = 0;
    fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
    fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

    # Compute derivative(导数) of d over [x, y] 计算d在x，y上的导数
    d_r = (k1 + r * (2.0 * k2 + r * (3.0 * k3 + r * 4.0 * k4)))
    d_x = 2.0 * x * d_r
    d_y = 2.0 * y * d_r

    # Compute derivative of fx over x and y.
    fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
    fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

    # Compute derivative of fy over x and y.
    fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
    fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

    return fx, fy, fx_x, fx_y, fy_x, fy_y  # 返回残差与残差对应的导数


# 去除径向和切向失真 返回去除畸变后的x,y
def _radial_and_tangential_undistort(  # radii: 3_cam(x_y, x+1_y, x_y+1), xd和yd分别包含3种位置情况下的x和y
        xd: _Array,  # cam_x: [3(radii), num_patches, _patch_size, _patch_size]
        yd: _Array,  # cam_y: [3(radii), num_patches, _patch_size, _patch_size]
        k1: float = 0,
        k2: float = 0,
        k3: float = 0,
        k4: float = 0,
        p1: float = 0,
        p2: float = 0,
        eps: float = 1e-9,  # 当畸变矫正的结果和原图像之间的残差很小时，不再迭代矫正
        max_iterations=10,
        xnp: types.ModuleType = np) -> Tuple[_Array, _Array]:
    """Computes undistorted (x, y) from (xd, yd)."""
    # From https://github.com/google/nerfies/blob/main/nerfies/camera.py
    # Initialize from the distorted point.
    x = xnp.copy(xd)
    y = xnp.copy(yd)

    for _ in range(max_iterations):  # 不断迭代，期望残差为0
        # 畸变在x,y上的残差与导数
        fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
            x=x, y=y, xd=xd, yd=yd, k1=k1, k2=k2, k3=k3, k4=k4, p1=p1, p2=p2)
        denominator = fy_x * fx_y - fx_x * fy_y  # 分母
        x_numerator = fx * fy_y - fy * fx_y
        y_numerator = fy * fx_x - fx * fy_x
        # np.where(condition,x,y) 条件成立时返回x，否则返回y
        step_x = xnp.where(
            xnp.abs(denominator) > eps, x_numerator / denominator,
            xnp.zeros_like(denominator))
        step_y = xnp.where(
            xnp.abs(denominator) > eps, y_numerator / denominator,
            xnp.zeros_like(denominator))

        x = x + step_x
        y = y + step_y

    return x, y


class ProjectionType(enum.Enum):
    """Camera projection type (standard perspective pinhole or fisheye model)."""
    PERSPECTIVE = 'perspective'
    FISHEYE = 'fisheye'


def pixels_to_rays(
        pix_x_int: _Array,  # [num_patches, _patch_size, _patch_size]
        pix_y_int: _Array,  # [num_patches, _patch_size, _patch_size]
        pixtocams: _Array,  # [num_patches, _patch_size, _patch_size, 3, 3]
        camtoworlds: _Array,  # [num_patches, _patch_size, _patch_size, 3, 4]
        distortion_params: Optional[Mapping[str, float]] = None,  # 用于对径向畸变的矫正
        pixtocam_ndc: Optional[_Array] = None,  # 将frustum映射到NDC（正交投影与透视投影）
        camtype: ProjectionType = ProjectionType.PERSPECTIVE,
        xnp: types.ModuleType = np,
) -> Tuple[_Array, _Array, _Array, _Array, _Array]:
    """Calculates rays given pixel coordinates, intrinisics, and extrinsics.

  Given 2D pixel coordinates pix_x_int, pix_y_int for cameras with
  inverse intrinsics pixtocams and extrinsics camtoworlds (and optional
  distortion coefficients distortion_params and NDC(normalized device coordinates) space projection matrix
  pixtocam_ndc), computes the corresponding 3D camera rays.

  Vectorized over the leading dimensions of the first four arguments.

  Args:
    pix_x_int: int array, shape SH, x coordinates of image pixels.
    pix_y_int: int array, shape SH, y coordinates of image pixels.
    pixtocams: float array, broadcastable to SH + [3, 3], inverse intrinsics.
    camtoworlds: float array, broadcastable to SH + [3, 4], camera extrinsics.
    distortion_params: dict of floats, optional camera distortion parameters.
    pixtocam_ndc: float array, [3, 3], optional inverse intrinsics for NDC.
    camtype: camera_utils.ProjectionType, fisheye or perspective camera.
    xnp: either numpy or jax.numpy.

  Returns:
    origins: float array, shape SH + [3], ray origin points.
    directions: float array, shape SH + [3], ray direction vectors.
    viewdirs: float array, shape SH + [3], normalized ray direction vectors.
    radii: float array, shape SH + [1], ray differential radii.
    imageplane: float array, shape SH + [2], xy coordinates on the image plane.
      If the image plane is at world space distance 1 from the pinhole, then
      imageplane will be the xy coordinates of a pixel in that space (so the
      camera ray direction at the origin would be (x, y, -1) in OpenGL coords).
  """

    # Must add half pixel offset to shoot rays through pixel centers.
    # 像素的半径为半个像素大小
    def pix_to_dir(x, y):
        return xnp.stack([x + .5, y + .5, xnp.ones_like(x)], axis=-1)

    # We need the dx and dy rays to calculate ray radii(半径) for mip-NeRF cones.
    # radii: 3(x_y, x+1_y, x_y+1)
    pixel_dirs_stacked = xnp.stack([  # [3(radii), num_patches, _patch_size, _patch_size, 3(pixel_xy1)]
        pix_to_dir(pix_x_int, pix_y_int),  # [num_patches, _patch_size, _patch_size, 3(pixel_xy1)]
        pix_to_dir(pix_x_int + 1, pix_y_int),
        pix_to_dir(pix_x_int, pix_y_int + 1)
    ], axis=0)

    # For jax, need to specify high-precision matmul.
    matmul = math.matmul if xnp == jnp else xnp.matmul
    mat_vec_mul = lambda A, b: matmul(A, b[..., None])[..., 0]

    # Apply inverse intrinsic matrices. 去除相机畸变 先将图像坐标系坐标转换到摄像机坐标系坐标
    # [3(radii), num_patches, _patch_size, _patch_size, 3(cam_xy1)]
    camera_dirs_stacked = mat_vec_mul(pixtocams, pixel_dirs_stacked)

    if distortion_params is not None:  # 对横纵坐标去畸变
        # Correct for distortion.
        x, y = _radial_and_tangential_undistort(
            camera_dirs_stacked[..., 0],  # cam_x: [3(radii), num_patches, _patch_size, _patch_size]
            camera_dirs_stacked[..., 1],  # cam_y: [3(radii), num_patches, _patch_size, _patch_size]
            **distortion_params,  # 用于对畸变矫正的参数
            xnp=xnp)
        # [3(radii), num_patches, _patch_size, _patch_size, 3(xy1)]
        camera_dirs_stacked = xnp.stack([x, y, xnp.ones_like(x)], -1)

    if camtype == ProjectionType.FISHEYE:  # 去鱼眼效果
        theta = xnp.sqrt(xnp.sum(xnp.square(camera_dirs_stacked[..., :2]), axis=-1))
        theta = xnp.minimum(xnp.pi, theta)

        sin_theta_over_theta = xnp.sin(theta) / theta
        camera_dirs_stacked = xnp.stack([
            camera_dirs_stacked[..., 0] * sin_theta_over_theta,
            camera_dirs_stacked[..., 1] * sin_theta_over_theta,
            xnp.cos(theta),
        ], axis=-1)

    # Flip from OpenCV to OpenGL coordinate system.
    # 变换坐标系，将倒立的像转换为正立：-y，像平面相对于光圈的位置为-f+z轴归一化: -1
    # [3(radii), num_patches, _patch_size, _patch_size, 3(xy1)]
    camera_dirs_stacked = matmul(camera_dirs_stacked,
                                 xnp.diag(xnp.array([1., -1., -1.])))  # 将一维数组转换为对角方阵

    # Extract 2D image plane (x, y) coordinates. 取出像素中心位置的坐标
    imageplane = camera_dirs_stacked[0, ..., :2]  # [num_patches, _patch_size, _patch_size, 2(xy)]

    # Apply camera rotation matrices.相机坐标系到世界坐标系
    # 猜测：z轴进行了归一化，即内参数矩阵也进行了相应的改变，此时z坐标值-f -> -1
    # [3(radii), num_patches, _patch_size, _patch_size, 3(xyz)]
    directions_stacked = mat_vec_mul(camtoworlds[..., :3, :3], camera_dirs_stacked)
    # Extract the offset rays. 将像素中心坐标的方向作为ray方向
    # [num_patches, _patch_size, _patch_size, 3(xyz)]
    directions, dx, dy = directions_stacked

    # 提取平移部分，即当前ray的原点在world中的坐标
    # [num_patches, _patch_size, _patch_size, 3(xyz)]
    origins = xnp.broadcast_to(camtoworlds[..., :3, -1], directions.shape)
    # 将dir单位化
    viewdirs = directions / xnp.linalg.norm(directions, axis=-1, keepdims=True)

    if pixtocam_ndc is None:
        # Distance from each unit-norm direction vector to its neighbors.
        # [num_patches, _patch_size, _patch_size]
        dx_norm = xnp.linalg.norm(dx - directions, axis=-1)
        dy_norm = xnp.linalg.norm(dy - directions, axis=-1)

    else:  # 将空间中的点投影到NDC中
        # Convert ray origins and directions into projective NDC space.
        # [num_patches, _patch_size, _patch_size, 3(xyz)]
        origins_dx, _ = convert_to_ndc(origins, dx, pixtocam_ndc)
        origins_dy, _ = convert_to_ndc(origins, dy, pixtocam_ndc)
        # [num_patches, _patch_size, _patch_size, 3(xyz)]
        # [num_patches, _patch_size, _patch_size, 3(dir)]
        origins, directions = convert_to_ndc(origins, directions, pixtocam_ndc)

        # In NDC space, we use the offset between origins instead of directions.
        dx_norm = xnp.linalg.norm(origins_dx - origins, axis=-1)
        dy_norm = xnp.linalg.norm(origins_dy - origins, axis=-1)

    # Cut the distance in half, multiply it to match the variance of a uniform
    # distribution the size of a pixel (1/12, see the original mipnerf paper).
    # 根据paper，将半径扩大为(2/√12)倍的像素大小
    # [num_patches, _patch_size, _patch_size, 1]
    radii = (0.5 * (dx_norm + dy_norm))[..., None] * 2 / xnp.sqrt(12)

    return origins, directions, viewdirs, radii, imageplane


def cast_ray_batch(
        cameras: Tuple[_Array, ...],
        pixels: utils.Pixels,
        camtype: ProjectionType = ProjectionType.PERSPECTIVE,
        xnp: types.ModuleType = np) -> utils.Rays:
    """Maps from input cameras and Pixel batch to output Ray batch.

  `cameras` is a Tuple of four sets of camera parameters.
    pixtocams: 1 or N stacked [3, 3]       inverse intrinsic matrices.
    camtoworlds: 1 or N stacked [3, 4]     extrinsic pose matrices.
    distortion_params: optional, dict[str, float] containing pinhole model distortion parameters.
    pixtocam_ndc: optional, [3, 3] inverse intrinsic matrix for mapping to NDC.

  Args:
    cameras: described above.
    pixels: integer pixel coordinates and camera indices, plus ray metadata.
            pix_x_int, pix_y_int, lossmult, nea, far, cam_idx, exposure_idx, exposure_values
    camtype: camera_utils.ProjectionType, fisheye or perspective camera.
    xnp: either numpy or jax.numpy.

  Returns:
    rays: Rays dataclass with computed 3D world space ray data.
  """

    '''
    pixtocams: [n, 3, 3]
    camtoworlds: [n, 3, 4]
    distortion_params: dict[str, float]
    pixtocam_ndc: [3, 3]
    '''
    pixtocams, camtoworlds, distortion_params, pixtocam_ndc = cameras

    # pixels.cam_idx has shape [..., 1], remove this hanging dimension.
    cam_idx = pixels.cam_idx[..., 0]  # [num_patches, _patch_size, _patch_size]
    # 使用cam_idx对arr的第0维进行截取或重复操作
    batch_index = lambda arr: arr if arr.ndim == 2 else arr[cam_idx]

    # Compute rays from pixel coordinates.
    origins, directions, viewdirs, radii, imageplane = pixels_to_rays(
        pixels.pix_x_int,  # [num_patches, _patch_size, _patch_size]
        pixels.pix_y_int,  # [num_patches, _patch_size, _patch_size]
        batch_index(pixtocams),  # [num_patches, _patch_size, _patch_size, 3, 3]
        batch_index(camtoworlds),  # [num_patches, _patch_size, _patch_size, 3, 4]
        distortion_params=distortion_params,
        pixtocam_ndc=pixtocam_ndc,
        camtype=camtype,
        xnp=xnp)

    # Create Rays data structure.
    return utils.Rays(
        origins=origins,  # ray的原点 [num_patches, _patch_size, _patch_size, 3(xyz)]
        directions=directions,  # 沿着像素中心的方向 [num_patches, _patch_size, _patch_size, 3(dir)]
        viewdirs=viewdirs,  # directions的单位向量[num_patches, _patch_size, _patch_size, 3(dir)]
        radii=radii,  # 半径大小 [num_patches, _patch_size, _patch_size, 1]
        imageplane=imageplane,  # [num_patches, _patch_size, _patch_size, 2(xy)]
        lossmult=pixels.lossmult,
        near=pixels.near,
        far=pixels.far,
        cam_idx=pixels.cam_idx,
        exposure_idx=pixels.exposure_idx,
        exposure_values=pixels.exposure_values,
    )


def cast_pinhole_rays(camtoworld: _Array,
                      height: int,
                      width: int,
                      focal: float,
                      near: float,
                      far: float,
                      xnp: types.ModuleType) -> utils.Rays:
    """Wrapper for generating a pinhole camera ray batch (w/o distortion)."""

    pix_x_int, pix_y_int = pixel_coordinates(width, height, xnp=xnp)
    pixtocam = get_pixtocam(focal, width, height, xnp=xnp)

    ray_args = pixels_to_rays(pix_x_int, pix_y_int, pixtocam, camtoworld, xnp=xnp)

    broadcast_scalar = lambda x: xnp.broadcast_to(x, pix_x_int.shape)[..., None]
    ray_kwargs = {
        'lossmult': broadcast_scalar(1.),
        'near': broadcast_scalar(near),
        'far': broadcast_scalar(far),
        'cam_idx': broadcast_scalar(0),
    }

    return utils.Rays(*ray_args, **ray_kwargs)


def cast_spherical_rays(camtoworld: _Array,  # [4, 4]
                        height: int,
                        width: int,
                        near: float,
                        far: float,
                        xnp: types.ModuleType) -> utils.Rays:
    """Generates a spherical camera ray batch."""

    # 使用w+1和h+1的大小来对2Π，Π进行分割，使得最终算出的每一个方向向量都对应了一个像素
    theta_vals = xnp.linspace(0, 2 * xnp.pi, width + 1)  # [0, 2Π]
    phi_vals = xnp.linspace(0, xnp.pi, height + 1)  # [0, Π]
    theta, phi = xnp.meshgrid(theta_vals, phi_vals, indexing='xy')  # [h+1, w+1]

    # Spherical coordinates in camera reference frame (y is up).
    # 球面坐标的半径为单位像素长度
    directions = xnp.stack([  # [h+1, w+1, 3(yzx)]
        -xnp.sin(phi) * xnp.sin(theta),  # y
        xnp.cos(phi),  # z
        xnp.sin(phi) * xnp.cos(theta),  # x
    ],
        axis=-1)

    # For jax, need to specify high-precision matmul.
    matmul = math.matmul if xnp == jnp else xnp.matmul
    # 投影到世界坐标系 [h+1, w+1, 3(yzx)]
    directions = matmul(camtoworld[:3, :3], directions[..., None])[..., 0]

    # np.diff: 后一个元素-前一个元素，导致元素数量-1
    # 每一个方向向量都对应了一个像素，向量相减则表示相邻像素间的距离
    dy = xnp.diff(directions[:, :-1], axis=0)  # [h, w, 3(yzx)]
    dx = xnp.diff(directions[:-1, :], axis=1)  # [h, w, 3(yzx)]
    directions = directions[:-1, :-1]  # [h, w, 3(yzx)]
    viewdirs = directions  # [h, w, 3(yzx)]

    # 原点在世界坐标系中的坐标为c_to_w的T部分
    origins = xnp.broadcast_to(camtoworld[:3, -1], directions.shape)  # [h, w, 3(T)]

    # 求范数
    dx_norm = xnp.linalg.norm(dx, axis=-1)  # [h, w]
    dy_norm = xnp.linalg.norm(dy, axis=-1)  # [h, w]
    # 求像素半径
    radii = (0.5 * (dx_norm + dy_norm))[..., None] * 2 / xnp.sqrt(12)  # [h, w, 1]

    imageplane = xnp.zeros_like(directions[..., :2])  # [h, w, 2]

    ray_args = (origins, directions, viewdirs, radii, imageplane)

    broadcast_scalar = lambda x: xnp.broadcast_to(x, radii.shape[:-1])[..., None]  # [h, w, 1]
    ray_kwargs = {
        'lossmult': broadcast_scalar(1.),
        'near': broadcast_scalar(near),
        'far': broadcast_scalar(far),
        'cam_idx': broadcast_scalar(0),
    }

    return utils.Rays(*ray_args, **ray_kwargs)
