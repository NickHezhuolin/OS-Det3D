
# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.parallel import DataContainer as DC

from mmdet3d.core.bbox import BaseInstance3DBoxes
from mmdet3d.core.points import BasePoints
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.datasets.pipelines import DefaultFormatBundle3D
from mmdet3d.datasets.pipelines import Collect3D

@PIPELINES.register_module()
class CustomDefaultFormatBundle3D(DefaultFormatBundle3D):
    """Default formatting bundle.
    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.
    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        results = super(CustomDefaultFormatBundle3D, self).__call__(results)
        results['gt_map_masks'] = DC(
            to_tensor(results['gt_map_masks']), stack=True)

        return results
    
@PIPELINES.register_module()
class OWCollect3D(Collect3D):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - 'img_shape': shape of the image input to the network as a tuple
            (h, w, c).  Note that images may be zero padded on the
            bottom/right if the batch tensor is larger than this shape.
        - 'scale_factor': a float indicating the preprocessing scale
        - 'flip': a boolean indicating if image flip transform was used
        - 'filename': path to the image file
        - 'ori_shape': original shape of the image as a tuple (h, w, c)
        - 'pad_shape': image shape after padding
        - 'lidar2img': transform from lidar to image
        - 'depth2img': transform from depth to image
        - 'cam2img': transform from camera to image
        - 'pcd_horizontal_flip': a boolean indicating if point cloud is
            flipped horizontally
        - 'pcd_vertical_flip': a boolean indicating if point cloud is
            flipped vertically
        - 'box_mode_3d': 3D box mode
        - 'box_type_3d': 3D box type
        - 'img_norm_cfg': a dict of normalization information:
            - mean: per channel mean subtraction
            - std: per channel std divisor
            - to_rgb: bool indicating if bgr was converted to rgb
        - 'pcd_trans': point cloud transformations
        - 'sample_idx': sample index
        - 'pcd_scale_factor': point cloud scale factor
        - 'pcd_rotation': rotation applied to point cloud
        - 'pts_filename': path to point cloud file.

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ('filename', 'ori_shape', 'img_shape', 'lidar2img',
            'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
            'box_type_3d', 'img_norm_cfg', 'pcd_trans',
            'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename')
    """

    def __init__(
        self,
        keys,
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'cam_intrinsic', 'lidar2cam', 'depth_map_tensor', # img-metas add-items
                   'sensor2ego_translation', 'sensor2ego_rotation', 'cam_ego2global_translation', 'cam_ego2global_rotation','lidar_ego2global_translation', 'lidar_ego2global_rotation', 'lidar2ego_translation', 'lidar2ego_rotation',# img-metas add-items
                   'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                   'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                   'pcd_scale_factor', 'pcd_rotation', 'pcd_rotation_angle',
                   'pts_filename', 'transformation_3d_flow', 'trans_mat',
                   'affine_aug')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``img_metas``
        """
        data = {}
        img_metas = {}
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        data['img_metas'] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'