import torch
import copy
from mmcv.parallel import DataContainer as DC
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

import mmcv
import cv2
import tempfile
import numpy as np
from os import path as osp
from mmdet.datasets.builder import DATASETS
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.pipelines import Compose
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
import random
from datetime import date

@DATASETS.register_module()
class OWCustomNuScenesDatasetUnkGT_Task1(NuScenesDataset):
    r"""NuScenes Dataset.
        This datset only add camera intrinsics and extrinsics to the results.
        """
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
        'unk_obj': '',
    }
    TRAIN_CLASSES = (  'unk_obj', 'car', 'construction_vehicle', 'barrier', 'bicycle', 'pedestrian', 'truck', 'trailer', 'bus', 'motorcycle', 'traffic_cone' )
    EVAL_CLASSES = ( 'unk_obj', 'car', 'construction_vehicle', 'barrier','bicycle', 'pedestrian')

    def __init__(self, queue_length=4, bev_size=(200, 200), overlap_test=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        
    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []
        index_list = list(range(index-self.queue_length, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[1:])
        index_list.append(index)
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            if self.filter_empty_gt and \
                    (example is None or ~(example['gt_labels_3d']._data != -1).any()):
                return None
            queue.append(example)
        return self.union2one(queue)
    
    def union2one(self, queue):
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if metas_map[i]['scene_token'] != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
        queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        return input_dict
    
    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info['valid_flag']
            gt_names = set(info['gt_names'][mask])
        else:
            gt_names = set(info['gt_names'])

        cat_ids = []

        if self.test_mode:
            for name in gt_names:
                if name in self.EVAL_CLASSES:
                    cat_ids.append(self.cat2id[name])
                else:
                    cat_ids.append(self.cat2id['unk_obj'])
        else:
            for name in gt_names:
                if name in self.TRAIN_CLASSES:
                    cat_ids.append(self.cat2id[name])

        return cat_ids
    
    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]

        gt_labels_3d = []
        if self.test_mode:
            for cat in gt_names_3d:
                if cat in self.EVAL_CLASSES:
                    gt_labels_3d.append(self.EVAL_CLASSES.index(cat))
                else:
                    gt_labels_3d.append(self.EVAL_CLASSES.index('unk_obj'))
        else:
            for cat in gt_names_3d:
                if cat in self.TRAIN_CLASSES:
                    gt_labels_3d.append(self.TRAIN_CLASSES.index(cat))
                else:
                    gt_labels_3d.append(-1)

        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results
    

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}

        if self.test_mode:
            mapped_class_names = self.EVAL_CLASSES
        else:
            mapped_class_names = self.TRAIN_CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(det, self.with_velocity)
            sample_token = self.data_infos[sample_id]['token']
            boxes = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes,
                                             mapped_class_names,
                                             self.eval_detection_configs,
                                             self.eval_version)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = OWCustomNuScenesDatasetUnkGT_Task1.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = OWCustomNuScenesDatasetUnkGT_Task1.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path
    
    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            metric (str, optional): Metric name used for evaluation.
                Default: 'bbox'.
            result_name (str, optional): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        # from nuscenes.eval.detection.evaluate import NuScenesEval
        from projects.mmdet3d_plugin.core.evaluation import CustomNusceneEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = CustomNusceneEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=True)
        nusc_eval.main(render_curves=True)

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.EVAL_CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val

        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail
    
    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))
        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: 'bbox'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str, optional): The prefix of json files including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(result_files[name])
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files)

        # if tmp_dir is not None:
        #     tmp_dir.cleanup()

        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return results_dict
    
    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        if self.test_mode:
            pipeline = [
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadPointsFromMultiSweeps',
                    sweeps_num=10,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=self.EVAL_CLASSES,
                    with_label=False),
                dict(type='OWCollect3D', keys=['points'])
            ]
        else:
            pipeline = [
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadPointsFromMultiSweeps',
                    sweeps_num=10,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=self.TRAIN_CLASSES,
                    with_label=False),
                dict(type='OWCollect3D', keys=['points'])
            ]
        return Compose(pipeline)

@DATASETS.register_module()
class OWCustomNuScenesDatasetUnkGT_Task2_ft(NuScenesDataset):
    r"""NuScenes Dataset.
        This datset only add camera intrinsics and extrinsics to the results.
        """
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
        'unk_obj': '',
    }
    TRAIN_CLASSES = ( 'unk_obj', 'car', 'construction_vehicle', 'barrier', 'bicycle', 'pedestrian', 'truck', 'bus', 'motorcycle', 'traffic_cone', 'trailer')
    EVAL_CLASSES = ( 'unk_obj', 'car', 'construction_vehicle', 'barrier', 'bicycle', 'pedestrian',  'truck', 'bus', 'motorcycle')

    def __init__(self, queue_length=4, bev_size=(200, 200), overlap_test=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        
    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []
        index_list = list(range(index-self.queue_length, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[1:])
        index_list.append(index)
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            if self.filter_empty_gt and \
                    (example is None or ~(example['gt_labels_3d']._data != -1).any()):
                return None
            queue.append(example)
        return self.union2one(queue)
    
    def union2one(self, queue):
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if metas_map[i]['scene_token'] != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
        queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        return input_dict
    
    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info['valid_flag']
            gt_names = set(info['gt_names'][mask])
        else:
            gt_names = set(info['gt_names'])

        cat_ids = []

        if self.test_mode:
            for name in gt_names:
                if name in self.EVAL_CLASSES:
                    cat_ids.append(self.cat2id[name])
                else:
                    cat_ids.append(self.cat2id['unk_obj'])
        else:
            for name in gt_names:
                if name in self.TRAIN_CLASSES:
                    cat_ids.append(self.cat2id[name])

        return cat_ids
    
    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]

        gt_labels_3d = []
        if self.test_mode:
            for cat in gt_names_3d:
                if cat in self.EVAL_CLASSES:
                    gt_labels_3d.append(self.EVAL_CLASSES.index(cat))
                else:
                    gt_labels_3d.append(self.EVAL_CLASSES.index('unk_obj'))
        else:
            for cat in gt_names_3d:
                if cat in self.TRAIN_CLASSES:
                    gt_labels_3d.append(self.TRAIN_CLASSES.index(cat))
                else:
                    gt_labels_3d.append(-1)

        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results
    

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}

        if self.test_mode:
            mapped_class_names = self.EVAL_CLASSES
        else:
            mapped_class_names = self.TRAIN_CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(det, self.with_velocity)
            sample_token = self.data_infos[sample_id]['token']
            boxes = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes,
                                             mapped_class_names,
                                             self.eval_detection_configs,
                                             self.eval_version)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = OWCustomNuScenesDatasetUnkGT_Task2_ft.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = OWCustomNuScenesDatasetUnkGT_Task2_ft.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path
    
    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            metric (str, optional): Metric name used for evaluation.
                Default: 'bbox'.
            result_name (str, optional): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        # from nuscenes.eval.detection.evaluate import NuScenesEval
        from projects.mmdet3d_plugin.core.evaluation import CustomNusceneEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = CustomNusceneEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=True)
        nusc_eval.main(render_curves=True)

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.EVAL_CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val

        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail
    
    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))
        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: 'bbox'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str, optional): The prefix of json files including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(result_files[name])
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files)

        # if tmp_dir is not None:
        #     tmp_dir.cleanup()

        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return results_dict
    
    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        if self.test_mode:
            pipeline = [
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadPointsFromMultiSweeps',
                    sweeps_num=10,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=self.EVAL_CLASSES,
                    with_label=False),
                dict(type='OWCollect3D', keys=['points'])
            ]
        else:
            pipeline = [
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadPointsFromMultiSweeps',
                    sweeps_num=10,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=self.TRAIN_CLASSES,
                    with_label=False),
                dict(type='OWCollect3D', keys=['points'])
            ]
        return Compose(pipeline)
       
    
def output_to_nusc_box(detection, with_velocity=True):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()

    # our LiDAR coordinate system -> nuScenes box coordinate system
    nus_box_dims = box_dims[:, [1, 0, 2]]

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        if with_velocity:
            velocity = (*box3d.tensor[i, 7:9], 0.0)
        else:
            velocity = (0, 0, 0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            nus_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(info,
                             boxes,
                             classes,
                             eval_configs,
                             eval_version='detection_cvpr_2019'):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str, optional): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        cls_range_map['unk_obj'] = 30
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list