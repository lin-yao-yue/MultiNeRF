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

"""Utility functions."""

import enum
import os
from typing import Any, Dict, Optional, Union

import flax
import jax
import jax.numpy as jnp
import numpy as np
from PIL import ExifTags
from PIL import Image

_Array = Union[np.ndarray, jnp.ndarray]


@flax.struct.dataclass
class Pixels:  # 类属性，所有实例共享的属性。非__init__初始化的实例属性
    """
    All tensors must have the same num_dims and first n-1 dims must match.
    """
    pix_x_int: _Array  # [num_patches, _patch_size, _patch_size]
    pix_y_int: _Array  # [num_patches, _patch_size, _patch_size]
    # weight to apply to each ray when computing loss fn
    lossmult: _Array  # [num_patches, _patch_size, _patch_size, 1] or [num_patches, _patch_size, _patch_size, 3]
    near: _Array  # [num_patches, _patch_size, _patch_size, 1]
    far: _Array  # [num_patches, _patch_size, _patch_size, 1]
    cam_idx: _Array  # [num_patches, _patch_size, _patch_size, 1]

    exposure_idx: Optional[_Array] = None
    exposure_values: Optional[_Array] = None


@flax.struct.dataclass
class Rays:
    """All tensors must have the same num_dims and first n-1 dims must match."""
    # ray的原点 [num_patches, _patch_size, _patch_size, 3(xyz)] for _next_train
    # [h, w, 3(xyz)] for _next_test
    origins: _Array
    # 沿着像素中心的方向 [num_patches, _patch_size, _patch_size, 3(dir)] for _next_train
    # [h, w, 3(dir)] for _next_test
    directions: _Array
    # directions的单位向量[num_patches, _patch_size, _patch_size, 3(dir)] for _next_train
    # [h, w, 3(dir)] = directions for _next_test
    viewdirs: _Array
    # 半径大小 [num_patches, _patch_size, _patch_size, 1] for _next_train
    # [h, w, 1] for _next_test
    radii: _Array
    # [num_patches, _patch_size, _patch_size, 2(xy)] for _next_train
    # [h, w, 2] for _next_test
    imageplane: _Array

    # weight to apply to each ray when computing loss fn
    # [num_patches, _patch_size, _patch_size, 1 or 3] for _next_train
    # [h, w, 1] for _next_test
    lossmult: _Array
    # [num_patches, _patch_size, _patch_size, 1] for _next_train
    # [h, w, 1] for _next_test
    near: _Array
    # [num_patches, _patch_size, _patch_size, 1] for _next_train
    # [h, w, 1] for _next_test
    far: _Array
    # [num_patches, _patch_size, _patch_size, 1] for _next_train
    # [h, w, 1] for _next_test
    cam_idx: _Array
    exposure_idx: Optional[_Array] = None
    exposure_values: Optional[_Array] = None


# Dummy Rays object that can be used to initialize NeRF model.
def dummy_rays(include_exposure_idx: bool = False,
               include_exposure_values: bool = False) -> Rays:
    data_fn = lambda n: jnp.zeros((1, n))
    exposure_kwargs = {}
    if include_exposure_idx:
        exposure_kwargs['exposure_idx'] = data_fn(1).astype(jnp.int32)
    if include_exposure_values:
        exposure_kwargs['exposure_values'] = data_fn(1)
    return Rays(
        origins=data_fn(3),
        directions=data_fn(3),
        viewdirs=data_fn(3),
        radii=data_fn(1),
        imageplane=data_fn(2),
        lossmult=data_fn(1),
        near=data_fn(1),
        far=data_fn(1),
        cam_idx=data_fn(1).astype(jnp.int32),
        **exposure_kwargs)


@flax.struct.dataclass
class Batch:
    """Data batch for NeRF training or testing."""
    rays: Union[Pixels, Rays]
    # [num_patches, _patch_size, _patch_size, 3] -> [device_count, batches_per_device, _patch_size, _patch_size, 3]
    rgb: Optional[_Array] = None
    # [num_patches, _patch_size, _patch_size, c] -> [device_count, batches_per_device, _patch_size, _patch_size, c]
    disps: Optional[_Array] = None
    # [num_patches, _patch_size, _patch_size, c] -> [device_count, batches_per_device, _patch_size, _patch_size, c]
    normals: Optional[_Array] = None
    # [num_patches, _patch_size, _patch_size, 1] -> [device_count, batches_per_device, _patch_size, _patch_size, 1]
    alphas: Optional[_Array] = None


class DataSplit(enum.Enum):
    """Dataset split."""
    TRAIN = 'train'
    TEST = 'test'


class BatchingMethod(enum.Enum):
    """Draw rays randomly from a single image or all images, in each batch."""
    ALL_IMAGES = 'all_images'
    SINGLE_IMAGE = 'single_image'


def open_file(pth, mode='r'):
    return open(pth, mode=mode)


def file_exists(pth):
    return os.path.exists(pth)


def listdir(pth):
    return os.listdir(pth)


def isdir(pth):
    return os.path.isdir(pth)


def makedirs(pth):
    if not file_exists(pth):
        os.makedirs(pth)


def shard(xs):
    """Split data into shards for multiple devices along the first dimension."""
    # tree_map(f, tree): f匿名函数作用于tree中的叶子，将Batch类对象xs中的leaf(属性成员)全部进行reshape
    # num_patches -> device_count, batches_per_device
    return jax.tree_util.tree_map(
        # +在list与tuple中是拼接，在numpy中是运算
        lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]), xs)


def unshard(x, padding=0):
    """Collect the sharded tensor to the shape before sharding."""
    y = x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))
    if padding > 0:
        y = y[:-padding]
    return y


def load_img(pth: str) -> np.ndarray:
    """Load an image and cast to float32."""
    with open_file(pth, 'rb') as f:  # 以二进制流读取图片
        # PIL：Python Imaging Library
        # 加载png图片得[h,w,4(RGBA)]
        # 加载diff图片得[h,w]
        image = np.array(Image.open(f), dtype=np.float32)
    return image


def load_exif(pth: str) -> Dict[str, Any]:
    """Load EXIF data for an image."""
    with open_file(pth, 'rb') as f:
        image_pil = Image.open(f)
        exif_pil = image_pil._getexif()  # pylint: disable=protected-access
        if exif_pil is not None:
            exif = {
                ExifTags.TAGS[k]: v for k, v in exif_pil.items() if k in ExifTags.TAGS
            }
        else:
            exif = {}
    return exif


def save_img_u8(img, pth):
    """Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
    with open_file(pth, 'wb') as f:
        Image.fromarray(
            (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)).save(
            f, 'PNG')


def save_img_f32(depthmap, pth):
    """Save an image (probably a depthmap) to disk as a float32 TIFF."""
    with open_file(pth, 'wb') as f:
        Image.fromarray(np.nan_to_num(depthmap).astype(np.float32)).save(f, 'TIFF')
