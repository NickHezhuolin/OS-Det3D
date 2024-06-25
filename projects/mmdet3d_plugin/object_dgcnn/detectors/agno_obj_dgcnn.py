import torch
import numpy as np
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector


@DETECTORS.register_module()
class AgnoObjDGCNN(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(AgnoObjDGCNN,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None

        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        bev_feature = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(bev_feature)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return (x, bev_feature)

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        (x, bev_feature) = pts_feats
        outs = self.pts_bbox_head(x)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def simple_test_pts(self, pts_feats, img_metas, rescale=False):
        """Test function of point cloud branch."""
        (x, bev_feature) = pts_feats
        outs = self.pts_bbox_head(x)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        
        # img_metas - only lidar info
        # bbox_results[0]['boxes_3d'] : dtype - LiDARInstance3DBoxes
        # bbox_list[0][0][0] : xyz, lwh, raw, vx, vy
        # bbox_results[0]['scores_3d'] : shape [300], dtype - torch.Tensor , box-sore is very low
        # bbox_results[0]['labels_3d'] : all = 0, shape [300], dtype - torch.Tensor

        # import os
        # import pickle
        # pts_path = img_metas[0]['pts_filename'] # 
        # file_name_with_extension = pts_path.split("/")[-1]  # 提取文件名和扩展名 data/nuscenes-mini/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin
        # file_name = file_name_with_extension.split(".")[0]  # 去掉扩展名
        
        # if not os.path.exists(os.path.join("./data/lidar_rpn/", 'voxel_agno_10cls_lidar_rpn_train_cs05')):
        #     os.makedirs(os.path.join("./data/lidar_rpn/", 'voxel_agno_10cls_lidar_rpn_train_cs05'))
        # save_query_path = os.path.join("./data/lidar_rpn/voxel_agno_10cls_lidar_rpn_train_cs05/", file_name + ".pkl")

        # bbox_results_tensor_list = []

        # for i in range(len(bbox_results[0]['boxes_3d'])):
        #     box = bbox_results[0]['boxes_3d'][i].tensor.squeeze(0) # torch.Size([1, 9]) 
        #     score = bbox_results[0]['scores_3d'][i] # torch.Size([]) : example - bbox_results[0]['scores_3d'][0] = tensor(0.1028)
        #     bbox_results_tensor_list.append(torch.cat((box, score.unsqueeze(0))))

        # with open(save_query_path, 'wb') as f:
        #     pickle.dump(bbox_results_tensor_list, f)
        
        # # for test
        # # with open(save_query_path, 'rb') as f:
        # #     loaded_data = pickle.load(f)
        
        #############################################
        # # for vis result : bev_embed
        # import numpy
        # import torch.nn as nn
        # import seaborn as sns
        # import matplotlib.pyplot as plt
        # from matplotlib.patches import Rectangle
        # import sys
        # import os
        
        # sample_idx = img_metas[0]['sample_idx']
        
        # visual_dir = f'visualization/{sample_idx}/'
        # if not os.path.isdir(visual_dir):
        #     os.makedirs(visual_dir)
            
            
        # # # 生成 gt 数据
        # from nuscenes.nuscenes import NuScenes
        # nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes', verbose=True)
        # my_sample = nusc.get('sample', sample_idx)
        
        # sensor = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT', 'LIDAR_TOP']
        # # sensor = ['CAM_FRONT', 'LIDAR_TOP']
        # for ss in sensor:
        #     cam_data = nusc.get('sample_data', my_sample['data'][ss])
        #     file_name = f'{visual_dir}gt_{ss}.png' 
        #     nusc.render_sample_data(cam_data['token'], out_path=file_name)
        #     print(f"{file_name} Save successfully!")
        
        #生成 bev_feat 可视化
        # upsaple = nn.Upsample(size=(int(1024),int(1024)), mode='bilinear', align_corners=True)
        # dense_heatmap_bev_queries_image = torch.mean(bev_feature.detach(), dim=1)
        # dense_image_bev_queries = dense_heatmap_bev_queries_image.cpu().clone()  # clone the tensor
        # dense_image_bev_queries = upsaple(dense_image_bev_queries.unsqueeze(0))
        # dense_image_bev_queries = dense_image_bev_queries.squeeze(0).squeeze(0)  # remove the fake batch dimension
        
        # 转换角点，忽略 z 轴
        # bbox_results_tensor_list = torch.stack(bbox_results_tensor_list)
        
        # 降序排列， 置信度高的在前面
        # values = bbox_results_tensor_list[:, -1]
        # sorted_indices = values.argsort(descending=True)
        # bbox_results_show = bbox_results_tensor_list[sorted_indices[:30]]
        
        # box_3d_points = get_corners_gt(bbox_results_show)
        
        # pdb.set_trace()
        
        # x_min, _ = torch.min(box_3d_points[..., 0], dim=1)
        # y_min, _ = torch.min(box_3d_points[..., 1], dim=1)
        # x_max, _ = torch.max(box_3d_points[..., 0], dim=1)
        # y_max, _ = torch.max(box_3d_points[..., 1], dim=1)
        # boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)
        
        # boxes += 51.2
        # boxes *= 10
        
        # plt.figure()
        # fig_path = visual_dir + f'L_BEV_after_transformer.png'
        # ax = sns.heatmap(dense_image_bev_queries.detach().numpy())  # Added ax
        
        # for box in boxes:
        #             xmin, ymin, xmax, ymax = box
        #             rect = Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='r', facecolor='none')
        #             ax.add_patch(rect)
        
        # plt.gca().invert_yaxis()
        # plt.title('C_BEV_after_transformer')
        # hm = ax.get_figure()  # Modified this line as well
        # hm.savefig(fig_path, dpi=36*36)
        # plt.close()
        # print(f"{fig_path} Save successfully!")
        # pdb.set_trace()
        #############################################

        return bbox_results

    def aug_test_pts(self, pts_feats, img_metas, rescale=False):
        """Test function of point cloud branch with augmentaiton.
        The function implementation process is as follows:
            - step 1: map features back for double-flip augmentation.
            - step 2: merge all features and generate boxes.
            - step 3: map boxes back for scale augmentation.
            - step 4: merge results.
        Args:
            feats (list[torch.Tensor]): Feature of point cloud.
            img_metas (list[dict]): Meta information of samples.
            rescale (bool): Whether to rescale bboxes. Default: False.
        Returns:
            dict: Returned bboxes consists of the following keys:
                - boxes_3d (:obj:`LiDARInstance3DBoxes`): Predicted bboxes.
                - scores_3d (torch.Tensor): Scores of predicted boxes.
                - labels_3d (torch.Tensor): Labels of predicted boxes.
        """
        (feats, bev_feature) = pts_feats
        
        # only support aug_test for one sample
        outs_list = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.pts_bbox_head(x[0])
            # merge augmented outputs before decoding bboxes
            for task_id, out in enumerate(outs):
                for key in out[0].keys():
                    if img_meta[0]['pcd_horizontal_flip']:
                        outs[task_id][0][key] = torch.flip(
                            outs[task_id][0][key], dims=[2])
                        if key == 'reg':
                            outs[task_id][0][key][:, 1, ...] = 1 - outs[
                                task_id][0][key][:, 1, ...]
                        elif key == 'rot':
                            outs[task_id][0][
                                key][:, 1,
                                     ...] = -outs[task_id][0][key][:, 1, ...]
                        elif key == 'vel':
                            outs[task_id][0][
                                key][:, 1,
                                     ...] = -outs[task_id][0][key][:, 1, ...]
                    if img_meta[0]['pcd_vertical_flip']:
                        outs[task_id][0][key] = torch.flip(
                            outs[task_id][0][key], dims=[3])
                        if key == 'reg':
                            outs[task_id][0][key][:, 0, ...] = 1 - outs[
                                task_id][0][key][:, 0, ...]
                        elif key == 'rot':
                            outs[task_id][0][
                                key][:, 0,
                                     ...] = -outs[task_id][0][key][:, 0, ...]
                        elif key == 'vel':
                            outs[task_id][0][
                                key][:, 0,
                                     ...] = -outs[task_id][0][key][:, 0, ...]

            outs_list.append(outs)

        preds_dicts = dict()
        scale_img_metas = []

        # concat outputs sharing the same pcd_scale_factor
        for i, (img_meta, outs) in enumerate(zip(img_metas, outs_list)):
            pcd_scale_factor = img_meta[0]['pcd_scale_factor']
            if pcd_scale_factor not in preds_dicts.keys():
                preds_dicts[pcd_scale_factor] = outs
                scale_img_metas.append(img_meta)
            else:
                for task_id, out in enumerate(outs):
                    for key in out[0].keys():
                        preds_dicts[pcd_scale_factor][task_id][0][key] += out[
                            0][key]

        aug_bboxes = []

        for pcd_scale_factor, preds_dict in preds_dicts.items():
            for task_id, pred_dict in enumerate(preds_dict):
                # merge outputs with different flips before decoding bboxes
                for key in pred_dict[0].keys():
                    preds_dict[task_id][0][key] /= len(outs_list) / len(
                        preds_dicts.keys())
            bbox_list = self.pts_bbox_head.get_bboxes(
                preds_dict, img_metas[0], rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        if len(preds_dicts.keys()) > 1:
            # merge outputs with different scales after decoding bboxes
            merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, scale_img_metas,
                                                self.pts_bbox_head.test_cfg)
            return merged_bboxes
        else:
            for key in bbox_list[0].keys():
                bbox_list[0][key] = bbox_list[0][key].to('cpu')
            return bbox_list[0]

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)
        bbox_list = dict()
        if pts_feats and self.with_pts_bbox:
            pts_bbox = self.aug_test_pts(pts_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=pts_bbox)
        return [bbox_list]
    
def get_corners_gt(bbox):

    # 分解边界框
    # center in the bev
    cx = bbox[:, 0:1]
    cy = bbox[:, 1:2]
    cz = bbox[:, 2:3]

    # size
    w = bbox[:, 3:4]
    l = bbox[:, 4:5]
    h = bbox[:, 5:6]

    # rot
    rot = bbox[:,6:7]
    cos, sin = torch.cos(rot), torch.sin(rot)

    # 创建一个3D边界框的8个角点在局部坐标系中的坐标

    lwh = torch.cat([l, w, h], dim=-1)
    x, y, z = lwh[:, 0] / 2, lwh[:, 1] / 2, lwh[:, 2] / 2
    local_corners = torch.stack([-x, -y, -z, x, -y, -z, x, y, -z, -x, y, -z,
                                  -x, -y, z, x, -y, z, x, y, z, -x, y, z], dim=-1)

    local_corners = local_corners.view(*lwh.shape[:-1], 8, 3)
    local_corners = local_corners.permute(0, 2, 1)  # 重塑为[900, 3, 8]

    # 首先，将旋转的弧度转换为角度
    rot_degrees = rot * (180 / np.pi)

    for i in range(len(rot_degrees)):
        if rot_degrees[i] >= 0:
            rot_degrees[i] -= 90
        else:
            rot_degrees[i] += 90

    rot = rot_degrees / (180 / np.pi)

    # 构建旋转矩阵
    cos, sin = rot.cos(), rot.sin()

    zeros, ones = torch.zeros_like(rot), torch.ones_like(rot)
    rotation_matrix = torch.stack([cos, -sin, zeros, sin, cos, zeros, zeros, zeros, ones], dim=-1)
    rotation_matrix = rotation_matrix.view(*rot.shape[:-1], 3, 3)

    # 旋转角点
    rotated_corners = torch.einsum('bij,bjk->bik', rotation_matrix, local_corners)

    # 平移角点
    xyz = torch.cat([cx, cy, cz], dim=-1)
    global_corners = rotated_corners.permute(0, 2, 1) + xyz.unsqueeze(1)

    return global_corners