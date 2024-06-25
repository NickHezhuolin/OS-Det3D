import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import force_fp32, auto_fp16
                        
from mmdet.core import multi_apply, reduce_mean, build_assigner, build_sampler
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
from mmdet.models import build_loss

from nuscenes.eval.common.loaders import filter_eval_boxes
import math
from mmdet3d.core.bbox.iou_calculators import bbox_overlaps_nearest_3d, bbox_overlaps_3d
import pdb

@HEADS.register_module()
class AgnoDGCNN3DHeadV2(DETRHead):
    """Head of DeformDETR3D. 
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """
    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 sampling=False,
                 objectness_type='Centerness',
                 loss_obj=dict(type='L1Loss', loss_weight=1.0),
                 objectness_assigner=dict(
                    type='AgnoHungarianAssigner3D',
                    reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                    iou_cost=dict(type='IoUCost', weight=0.0),
                    pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
                 **kwargs):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.voxel_size = self.bbox_coder.voxel_size

        self.bev_shape = (int((self.pc_range[3] - self.pc_range[0]) / self.voxel_size[0]), 
                          int((self.pc_range[4] - self.pc_range[1]) / self.voxel_size[1]))

        self.num_cls_fcs = num_cls_fcs - 1
        self.objectness_type = objectness_type
        self.sampling = sampling
        
        super(AgnoDGCNN3DHeadV2, self).__init__(
            *args, transformer=transformer, **kwargs)
        # # Define objectness assigner and sampler
        self.objectness_assigner = build_assigner(objectness_assigner)
        if loss_obj is not None:
            self.loss_obj = build_loss(loss_obj)
    

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        
        # classification branch - Foreground-Background Classification Head
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, 1))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            # reg_branch.append(nn.LayerNorm(self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)
        
        # objectness branch
        obj_branch = []
        for _ in range(self.num_reg_fcs):
            obj_branch.append(Linear(self.embed_dims, self.embed_dims))
            # reg_branch.append(nn.LayerNorm(self.embed_dims))
            obj_branch.append(nn.ReLU())
        obj_branch.append(Linear(self.embed_dims, 1))
        obj_branch = nn.Sequential(*obj_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
            self.obj_branches = _get_clones(obj_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
            self.obj_branches = nn.ModuleList(
                [obj_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)
            for m in self.obj_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def forward(self, mlvl_feats):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
            enc_outputs_class (Tensor): The score of each point on encode \
                feature map, has shape (N, h*w, num_class). Only when \
                as_two_stage is Ture it would be returned, otherwise \
                `None` would be returned.
            enc_outputs_coord (Tensor): The proposal generate from the \
                encode feature map, has shape (N, h*w, 4). Only when \
                as_two_stage is Ture it would be returned, otherwise \
                `None` would be returned.
        """

        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = self.bev_shape
        img_masks = mlvl_feats[0].new_zeros(
            (batch_size, input_img_h, input_img_w))

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight
        hs, init_reference, inter_references, \
            enc_outputs_class, enc_outputs_coord = self.transformer(
                    mlvl_feats,
                    mlvl_masks,
                    query_embeds,
                    mlvl_positional_encodings,
                    reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                    cls_branches=self.cls_branches if self.as_two_stage else None  # noqa:E501
            )
            
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        outputs_objs = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            outputs_obj = self.obj_branches[lvl](hs[lvl])

            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., 0:2] += reference
                tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
                tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
                tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])

            if tmp.size(-1) > 8:
                outputs_coord = torch.cat((tmp[..., :6], tmp[..., 6:8], tmp[..., 8:]), -1)
            else:
                outputs_coord = torch.cat((tmp[..., :6], tmp[..., 6:8]), -1)
                
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_objs.append(outputs_obj)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_objs = torch.stack(outputs_objs)
        
        if self.as_two_stage:
            outs = {
                'all_cls_scores': outputs_classes,
                'all_bbox_preds': outputs_coords,
                'all_obj_scores': outputs_objs,
                'enc_cls_scores': enc_outputs_class,
                'enc_bbox_preds': enc_outputs_coord.sigmoid(), 
            }
        else:
            outs = {
                'all_cls_scores': outputs_classes,
                'all_bbox_preds': outputs_coords,
                'all_obj_scores': outputs_objs,
                'enc_cls_scores': None,
                'enc_bbox_preds': None, 
            }
        return outs

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_bboxes,
                           gt_labels,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :self.code_size-1]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        
        ############################################################################

        obj_assign_result = self.objectness_assigner.assign(bbox_pred, gt_bboxes, gt_labels, gt_bboxes_ignore)
        objectness_sampling_result = self.sampler.sample(obj_assign_result, bbox_pred, gt_bboxes)
        
        num_valid_anchors = bbox_pred.shape[0]
        
        objectness_targets = bbox_pred.new_zeros(
            num_valid_anchors, dtype=torch.float)
        objectness_weights = bbox_pred.new_zeros(
            num_valid_anchors, dtype=torch.float)
        objectness_pos_inds = objectness_sampling_result.pos_inds # 1
        objectness_neg_inds = objectness_sampling_result.neg_inds # 0
        
        # 对 gt 对应对象操作，对两次二分匹配的结果 设置 conf 为 1
        # 对 非gt 对应对象操作，对两次二分匹配的结果 设置 conf 为 计算的 distance-base score
        # 对 无lidar点 对应对象，对两次二分匹配的结果 设置 conf 为 0
        
        if len(objectness_neg_inds) > 0:
            # Centerness as tartet -- Default
            denormalize_pred_bbox = denormalize_bbox(objectness_sampling_result.pos_bboxes, self.pc_range)
            if self.objectness_type == 'Centerness':
                # 提取xyz坐标
                pos_bboxes_xyz = denormalize_pred_bbox[:, :3]
                pos_gt_bboxes_xyz = objectness_sampling_result.pos_gt_bboxes[:, :3]
                # 计算欧几里得距离
                distance = torch.norm(pos_bboxes_xyz - pos_gt_bboxes_xyz, dim=1)
                # 计算匹配框中心点距离分数
                match_bbox_score = torch.exp(-0.1 * distance) # 0.1 是超参，控制分数下降速度， 当 d=4 时，得分为 0.67
                pos_objectness_targets = match_bbox_score
            elif self.objectness_type == 'Scaleness':
                # 提取xyz坐标
                pos_bboxes_xyz = denormalize_pred_bbox[:, :3]
                pos_gt_bboxes_xyz = sampling_result.pos_gt_bboxes[:, :3]
                # 计算欧几里得距离
                distance = torch.norm(pos_bboxes_xyz - pos_gt_bboxes_xyz, dim=1)
                # 计算匹配框计算bboxL1分数
                pos_bboxes_hwl = denormalize_pred_bbox[:,3:6]
                pos_gt_bboxes_hwl = sampling_result.pos_gt_bboxes[:, 3:6]
                pos_bboxes_hwl_scale_prod = pos_bboxes_hwl.prod(dim=1)
                pos_gt_bboxes_hwl_scale_prod = pos_gt_bboxes_hwl.prod(dim=1)
                # 计算体积比
                match_bbox_score = torch.minimum(pos_bboxes_hwl_scale_prod, pos_gt_bboxes_hwl_scale_prod) / torch.maximum(pos_bboxes_hwl_scale_prod, pos_gt_bboxes_hwl_scale_prod)
                match_bbox_score = match_bbox_score ** (1/3)
                distance_mask = distance < 4
                pos_objectness_targets = match_bbox_score * distance_mask
            elif self.objectness_type == 'Centerness_Scaleness_01':
                # 提取xyz坐标
                pos_bboxes_xyz = denormalize_pred_bbox[:, :3]
                pos_gt_bboxes_xyz = sampling_result.pos_gt_bboxes[:, :3]
                # 计算欧几里得距离
                distance = torch.norm(pos_bboxes_xyz - pos_gt_bboxes_xyz, dim=1)
                # 计算匹配框计算bboxL1分数
                pos_bboxes_hwl = denormalize_pred_bbox[:,3:6]
                pos_gt_bboxes_hwl = sampling_result.pos_gt_bboxes[:, 3:6]
                pos_bboxes_hwl_scale_prod = pos_bboxes_hwl.prod(dim=1)
                pos_gt_bboxes_hwl_scale_prod = pos_gt_bboxes_hwl.prod(dim=1)
                # 计算体积比
                match_bbox_score = torch.minimum(pos_bboxes_hwl_scale_prod, pos_gt_bboxes_hwl_scale_prod) / torch.maximum(pos_bboxes_hwl_scale_prod, pos_gt_bboxes_hwl_scale_prod)
                match_bbox_score = match_bbox_score ** (1/3)
                distance_mask = torch.exp(-0.1 * distance) # 0.1 是超参，控制分数下降速度， 当 d=4 时，得分为 0.67
                pos_objectness_targets = match_bbox_score * distance_mask
            elif self.objectness_type == 'Centerness_Scaleness_05':
                # 提取xyz坐标
                pos_bboxes_xyz = denormalize_pred_bbox[:, :3]
                pos_gt_bboxes_xyz = sampling_result.pos_gt_bboxes[:, :3]
                # 计算欧几里得距离
                distance = torch.norm(pos_bboxes_xyz - pos_gt_bboxes_xyz, dim=1)
                # 计算匹配框计算bboxL1分数
                pos_bboxes_hwl = denormalize_pred_bbox[:,3:6]
                pos_gt_bboxes_hwl = sampling_result.pos_gt_bboxes[:, 3:6]
                pos_bboxes_hwl_scale_prod = pos_bboxes_hwl.prod(dim=1)
                pos_gt_bboxes_hwl_scale_prod = pos_gt_bboxes_hwl.prod(dim=1)
                # 计算体积比
                match_bbox_score = torch.minimum(pos_bboxes_hwl_scale_prod, pos_gt_bboxes_hwl_scale_prod) / torch.maximum(pos_bboxes_hwl_scale_prod, pos_gt_bboxes_hwl_scale_prod)
                match_bbox_score = match_bbox_score ** (1/3)
                distance_mask = torch.exp(-0.5 * distance) # 0.5 是超参，控制分数下降速度， 当 d=4 时，得分为 0.67
                pos_objectness_targets = match_bbox_score * distance_mask
            elif self.objectness_type == 'Centerness_Scaleness_005':
                # 提取xyz坐标
                pos_bboxes_xyz = denormalize_pred_bbox[:, :3]
                pos_gt_bboxes_xyz = sampling_result.pos_gt_bboxes[:, :3]
                # 计算欧几里得距离
                distance = torch.norm(pos_bboxes_xyz - pos_gt_bboxes_xyz, dim=1)
                # 计算匹配框计算bboxL1分数
                pos_bboxes_hwl = denormalize_pred_bbox[:,3:6]
                pos_gt_bboxes_hwl = sampling_result.pos_gt_bboxes[:, 3:6]
                pos_bboxes_hwl_scale_prod = pos_bboxes_hwl.prod(dim=1)
                pos_gt_bboxes_hwl_scale_prod = pos_gt_bboxes_hwl.prod(dim=1)
                # 计算体积比
                match_bbox_score = torch.minimum(pos_bboxes_hwl_scale_prod, pos_gt_bboxes_hwl_scale_prod) / torch.maximum(pos_bboxes_hwl_scale_prod, pos_gt_bboxes_hwl_scale_prod)
                match_bbox_score = match_bbox_score ** (1/3)
                distance_mask = torch.exp(-0.05 * distance) # 0.5 是超参，控制分数下降速度， 当 d=4 时，得分为 0.67
                pos_objectness_targets = match_bbox_score * distance_mask
            elif self.objectness_type == 'DIOU3D':
                diou3d_scores = calculate_diou3d(denormalize_pred_bbox, sampling_result.pos_gt_bboxes)
                pos_objectness_targets = diou3d_scores
            elif self.objectness_type == 'RDIOU3D':
                rdiou3d_scores = calculate_rdiou3d(denormalize_pred_bbox, sampling_result.pos_gt_bboxes)
                pos_objectness_targets = rdiou3d_scores
            elif self.objectness_type == 'center_IOU3D':
                # 提取用于计算的 7 维数据：[x, y, z, h, w, l, ry]
                pred_bboxes_7d = denormalize_pred_bbox[:, :7]
                gt_bboxes_7d = sampling_result.pos_gt_bboxes[:, :7]

                # 计算边界框的中心点距离
                pred_centers, pred_dimensions = pred_bboxes_7d[:, :3], pred_bboxes_7d[:, 3:6]
                gt_centers, gt_dimensions = gt_bboxes_7d[:, :3], gt_bboxes_7d[:, 3:6]
                center_distance = torch.norm(pred_centers - gt_centers, dim=1)

                # 计算 IOU3D
                iou3d_matrix = bbox_overlaps_3d(pred_bboxes_7d, gt_bboxes_7d)
                iou3d = torch.diag(iou3d_matrix)
                distance_mask = torch.exp(-0.5 * center_distance) # 0.5 是超参，控制分数下降速度， 当 d=4 时，得分为 0.67
                pos_objectness_targets = iou3d * distance_mask
                pdb.set_trace()
            elif self.objectness_type == 'Center_Vector_Minkowski_1':
                # 提取 xyz 坐标
                pos_bboxes_xyz = denormalize_pred_bbox[:, :3]
                pos_gt_bboxes_xyz = sampling_result.pos_gt_bboxes[:, :3]
                # 计算欧几里得距离
                distance = torch.norm(pos_bboxes_xyz - pos_gt_bboxes_xyz, dim=1)
                distance_mask = torch.exp(-0.1 * distance)
                
                pos_bboxes = denormalize_pred_bbox[:,3:7]
                pos_gt_bboxes = sampling_result.pos_gt_bboxes[:, 3:7]
                bbox_vectors  = create_bbox_vectors(pos_bboxes)
                gt_bbox_vectors  = create_bbox_vectors(pos_gt_bboxes)
                
                # 假设 create_bbox_vectors 函数返回的是每个向量的元组
                def stack_vectors(vectors_list):
                    return torch.stack([torch.cat(vectors) for vectors in vectors_list])

                # 将向量列表转换为张量
                bbox_vectors_tensor = stack_vectors(bbox_vectors)
                gt_bbox_vectors_tensor = stack_vectors(gt_bbox_vectors)
                
                match_bbox_score = torch.sqrt(torch.sum((bbox_vectors_tensor - gt_bbox_vectors_tensor) ** 2, dim=1))
                match_bbox_score = torch.exp(-0.01 * match_bbox_score)
                match_bbox_score = match_bbox_score.to(distance_mask.device)
                pos_objectness_targets = match_bbox_score * distance_mask
            elif self.objectness_type == 'Center_Vector_Minkowski_2':
                pos_bboxes = denormalize_pred_bbox[:,:7]
                pos_gt_bboxes = sampling_result.pos_gt_bboxes[:, :7]
                match_bbox_score = torch.sum((pos_bboxes - pos_gt_bboxes) ** 2, dim=1)
            elif self.objectness_type == 'Center_Vector_gas':
                # 提取 xyz 坐标
                pos_bboxes_xyz = denormalize_pred_bbox[:, :3]
                pos_gt_bboxes_xyz = sampling_result.pos_gt_bboxes[:, :3]
                # 计算欧几里得距离
                distance = torch.norm(pos_bboxes_xyz - pos_gt_bboxes_xyz, dim=1)
                # 提取 lwh 尺寸和朝向角 raw
                pos_bboxes_lwh = denormalize_pred_bbox[:, 3:6]
                pos_gt_bboxes_lwh = sampling_result.pos_gt_bboxes[:, 3:6]
                # 计算 lwh 尺寸比
                size_ratio = torch.prod(torch.minimum(pos_bboxes_lwh[:, :3], pos_gt_bboxes_lwh[:, :3]) / torch.maximum(pos_bboxes_lwh[:, :3], pos_gt_bboxes_lwh[:, :3]), dim=1)
                # 朝向角归一化
                pos_bboxes_raw = denormalize_pred_bbox[:, 7]
                pos_bboxes_degrees = radians_to_degrees(pos_bboxes_raw)
                pos_gt_bboxes_raw = sampling_result.pos_gt_bboxes[:, 7]
                pos_gt_bboxes_degrees = radians_to_degrees(pos_gt_bboxes_raw)
                orientation_ratio = torch.minimum(pos_bboxes_degrees, pos_gt_bboxes_degrees) / torch.maximum(pos_bboxes_degrees, pos_gt_bboxes_degrees) 
                 
                # 计算得分
                score = size_ratio * orientation_ratio
                distance_mask = torch.exp(-0.1 * distance)
                match_bbox_score = score ** (1/4) + distance_mask
                pos_objectness_targets = match_bbox_score
            else:
                raise ValueError(
                    'objectness_type must be either "Centerness" (Default) or '
                    '"BoxIoU".')
        
            objectness_targets[objectness_pos_inds] = pos_objectness_targets
            objectness_weights[objectness_pos_inds] = 1.0   

        if len(objectness_neg_inds) > 0: 
            objectness_targets[objectness_neg_inds] = 0.0
            objectness_weights[objectness_neg_inds] = 1.0
            
        ############################################################################
        
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds,
                objectness_targets, objectness_weights, objectness_pos_inds, objectness_neg_inds)


    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, pos_inds_list, neg_inds_list, 
         objectness_targets_list, objectness_weights_list, objectness_pos_inds_list, 
         objectness_neg_inds_list) = multi_apply( self._get_target_single, cls_scores_list, bbox_preds_list,
                                                    gt_bboxes_list, gt_labels_list, gt_bboxes_ignore_list)
        
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        
        obj_num_total_pos = sum((inds.numel() for inds in objectness_pos_inds_list))
        obj_num_total_neg = sum((inds.numel() for inds in objectness_neg_inds_list))
        
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg,
                objectness_targets_list, objectness_weights_list, obj_num_total_pos, obj_num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    obj_scores,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg,
                objectness_targets_list, objectness_weights_list, 
                obj_num_total_pos, obj_num_total_neg) = cls_reg_targets
        
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        objectness_targets = torch.cat(objectness_targets_list, 0)
        objectness_weights = torch.cat(objectness_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)

        loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :8], normalized_bbox_targets[isnotnan, :8], bbox_weights[isnotnan, :8], avg_factor=num_total_pos)
        if self.code_size > 8:
            loss_bbox_vel = self.loss_bbox(
                    bbox_preds[isnotnan, 8:], normalized_bbox_targets[isnotnan, 8:], bbox_weights[isnotnan, 8:], avg_factor=num_total_pos)
            loss_bbox = loss_bbox + loss_bbox_vel * 0.2
            
        # objective loss
        obj_scores = obj_scores.reshape(-1, self.cls_out_channels)
        obj_scores = obj_scores.squeeze()
        # construct weighted avg_factor to match with the official DETR repo
        loss_obj = self.loss_obj(
            obj_scores[isnotnan], objectness_targets[isnotnan], objectness_weights[isnotnan], avg_factor=num_total_pos)

        return loss_cls, loss_bbox, loss_obj
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:
            
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        all_obj_scores = preds_dicts['all_obj_scores']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        
        # 全部映射为 0
        for tensor in gt_labels_list:
            # 将张量移动到CPU并转换为NumPy数组
            numpy_array = tensor.cpu().numpy()
            # 使用if语句来检查是否存在大于或等于5的值
            assert not (numpy_array >= 10).any(), "Value greater than or equal to 10 found"
        gt_labels_list = [torch.zeros_like(ts) for ts in gt_labels_list]

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox, losses_obj = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds, all_obj_scores,
            all_gt_bboxes_list, all_gt_labels_list, 
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_obj'] = losses_obj[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_obj_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1],
                                           losses_obj[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_obj'] = loss_obj_i
            num_dec_layer += 1
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            if bboxes.size(-1) == 9:
                bboxes = img_metas[i]['box_type_3d'](bboxes, 9)
            else:
                bboxes = img_metas[i]['box_type_3d'](bboxes, 7)
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list
    
def calculate_diou3d(pred_bboxes, gt_bboxes):
    # 提取用于计算的 7 维数据：[x, y, z, h, w, l, ry]
    pred_bboxes_7d = pred_bboxes[:, :7]
    gt_bboxes_7d = gt_bboxes[:, :7]

    # 计算边界框的中心点距离
    pred_centers, pred_dimensions = pred_bboxes[:, :3], pred_bboxes[:, 3:6]
    gt_centers, gt_dimensions = gt_bboxes[:, :3], gt_bboxes[:, 3:6]
    center_distance = torch.norm(pred_centers - gt_centers, dim=1)

    # 计算 IOU3D
    iou3d_matrix = bbox_overlaps_3d(pred_bboxes_7d, gt_bboxes_7d)
    iou3d = torch.diag(iou3d_matrix)
    # 计算对角线长度的最大值
    c2 = calculate_diagonal(pred_bboxes_7d, gt_bboxes_7d).to(iou3d.device)
    
    # 计算 DIOU3D
    diou3d = iou3d - (center_distance ** 2 / c2)

    return diou3d

def calculate_rdiou3d(bboxes1, bboxes2):
    x1u, y1u, z1u = bboxes1[:,0], bboxes1[:,1], bboxes1[:,2]
    l1, w1, h1 =  torch.exp(bboxes1[:,3]), torch.exp(bboxes1[:,4]), torch.exp(bboxes1[:,5])
    t1 = torch.sin(bboxes1[:,6]) * torch.cos(bboxes2[:,6])
    x2u, y2u, z2u = bboxes2[:,0], bboxes2[:,1], bboxes2[:,2]
    l2, w2, h2 =  torch.exp(bboxes2[:,3]), torch.exp(bboxes2[:,4]), torch.exp(bboxes2[:,5])
    t2 = torch.cos(bboxes1[:,6]) * torch.sin(bboxes2[:,6])

    # we emperically scale the y/z to make their predictions more sensitive.
    x1 = x1u
    y1 = y1u * 2
    z1 = z1u * 2
    x2 = x2u
    y2 = y2u * 2
    z2 = z2u * 2

    # clamp is necessray to aviod inf.
    l1, w1, h1 = torch.clamp(l1, max=10), torch.clamp(w1, max=10), torch.clamp(h1, max=10)
    j1, j2 = torch.ones_like(h2), torch.ones_like(h2)

    volume_1 = l1 * w1 * h1 * j1
    volume_2 = l2 * w2 * h2 * j2

    inter_l = torch.max(x1 - l1 / 2, x2 - l2 / 2)
    inter_r = torch.min(x1 + l1 / 2, x2 + l2 / 2)
    inter_t = torch.max(y1 - w1 / 2, y2 - w2 / 2)
    inter_b = torch.min(y1 + w1 / 2, y2 + w2 / 2)
    inter_u = torch.max(z1 - h1 / 2, z2 - h2 / 2)
    inter_d = torch.min(z1 + h1 / 2, z2 + h2 / 2)
    inter_m = torch.max(t1 - j1 / 2, t2 - j2 / 2)
    inter_n = torch.min(t1 + j1 / 2, t2 + j2 / 2)

    inter_volume = torch.clamp((inter_r - inter_l),min=0) * torch.clamp((inter_b - inter_t),min=0) \
        * torch.clamp((inter_d - inter_u),min=0) * torch.clamp((inter_n - inter_m),min=0)
    
    c_l = torch.min(x1 - l1 / 2,x2 - l2 / 2)
    c_r = torch.max(x1 + l1 / 2,x2 + l2 / 2)
    c_t = torch.min(y1 - w1 / 2,y2 - w2 / 2)
    c_b = torch.max(y1 + w1 / 2,y2 + w2 / 2)
    c_u = torch.min(z1 - h1 / 2,z2 - h2 / 2)
    c_d = torch.max(z1 + h1 / 2,z2 + h2 / 2)
    c_m = torch.min(t1 - j1 / 2,t2 - j2 / 2)
    c_n = torch.max(t1 + j1 / 2,t2 + j2 / 2)

    inter_diag = (x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2 + (t2 - t1)**2
    c_diag = torch.clamp((c_r - c_l),min=0)**2 + torch.clamp((c_b - c_t),min=0)**2 + torch.clamp((c_d - c_u),min=0)**2  + torch.clamp((c_n - c_m),min=0)**2

    union = volume_1 + volume_2 - inter_volume
    u = (inter_diag) / c_diag
    rdiou = inter_volume / union
    rdiou_score = rdiou - u
    return rdiou_score

def rotation_matrix(angle):
    """ 创建绕 Z 轴的旋转矩阵 """
    cos_val = torch.cos(angle)
    sin_val = torch.sin(angle)
    return torch.stack([cos_val, -sin_val, torch.zeros_like(cos_val),
                        sin_val, cos_val, torch.zeros_like(cos_val),
                        torch.zeros_like(cos_val), torch.zeros_like(cos_val), torch.ones_like(cos_val)], dim=1).view(-1, 3, 3)

def calculate_corners_3d(bbox):
    x, y, z, l, w, h, r = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3], bbox[:, 4], bbox[:, 5], bbox[:, 6]
    corners = []
    for dx in [-0.5, 0.5]:
        for dy in [-0.5, 0.5]:
            for dz in [-0.5, 0.5]:
                corner = torch.stack([dx * l, dy * w, dz * h], dim=1)
                # 应用旋转
                rot_mat = rotation_matrix(r)
                rotated_corner = torch.bmm(rot_mat, corner.unsqueeze(-1)).squeeze(-1)
                corners.append(rotated_corner + torch.stack([x, y, z], dim=1))
    return torch.stack(corners, dim=1)  # shape [n, 8, 3]

def calculate_diagonal(pred_bboxes, gt_bboxes):
    # 计算预测框和真实框的所有角点
    pred_corners = calculate_corners_3d(pred_bboxes)  # shape [n, 8, 3]
    gt_corners = calculate_corners_3d(gt_bboxes)  # shape [n, 8, 3]

    # 计算角点间的距离
    max_diagonals = []
    for i in range(pred_corners.size(0)):
        distances = torch.norm(pred_corners[i].unsqueeze(0) - gt_corners[i].unsqueeze(1), dim=2)
        max_distance = torch.max(distances)
        max_diagonals.append(max_distance)

    return torch.tensor(max_diagonals).pow(2)

def calculate_size_similarity(pred_dimensions,gt_dimensions):
    ratio_width = torch.atan(gt_dimensions[:,0]/gt_dimensions[:,1]) - torch.atan(pred_dimensions[:,0]/pred_dimensions[:,1])
    ratio_height = torch.atan(gt_dimensions[:,1]/gt_dimensions[:,2]) - torch.atan(pred_dimensions[:,1]/pred_dimensions[:,2])
    ratio_depth = torch.atan(gt_dimensions[:,2]/gt_dimensions[:,0]) - torch.atan(pred_dimensions[:,2]/pred_dimensions[:,0])
    
    # 定义v为这些比例差异的平方和
    v = (4 / math.pi**2) * (ratio_width**2 + ratio_height**2 + ratio_depth**2)
    return v

def radians_to_degrees(radians):
    """
    Convert radians to degrees and ensure the angle is within 0 to 360 degrees.
    """
    degrees = radians * (180 / math.pi)
    degrees = degrees % 360  # Ensure the angle is within 0 to 360 degrees
    degrees = degrees / 360 + 1e-12
    return degrees

def create_bbox_vectors(bbox_data):
    bbox_vectors = []

    for bbox in bbox_data:
        l, w, h, r = bbox
        length_vector = torch.tensor([l, 0, 0])
        width_vector = torch.tensor([0, w, 0])
        height_vector = torch.tensor([0, 0, h])

        rotation_matrix = torch.tensor([
            [torch.cos(r), -torch.sin(r), 0],
            [torch.sin(r), torch.cos(r), 0],
            [0, 0, 1]
        ])

        rotated_length_vector = torch.matmul(rotation_matrix, length_vector)
        rotated_width_vector = torch.matmul(rotation_matrix, width_vector)

        bbox_vectors.append((rotated_length_vector, rotated_width_vector, height_vector))

    return bbox_vectors
