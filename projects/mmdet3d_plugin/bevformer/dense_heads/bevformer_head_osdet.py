import copy
import torch
import torch.nn as nn
import numpy as np
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import build_loss
import pickle
import pdb
import time
from nuscenes.nuscenes import NuScenes

import os

@HEADS.register_module()
class BEVFormerHead_OSDET(DETRHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 topk=3, # owod parm
                 owod=False, # owod parm
                 owod_decoder_layer=6, # owod parm
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.topk = topk # owod parm
        self.owod = owod # owod parm
        self.owod_decoder_layer = owod_decoder_layer # owod parm
        self.fp16_enabled = False

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        super(BEVFormerHead_OSDET, self).__init__(
            *args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None,  only_bev=False):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev
        )
    
        bev_embed, hs, init_reference, inter_references = outputs
        
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                             self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                             self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                             self.pc_range[2]) + self.pc_range[2])

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        
        if self.owod:
              outs = {
                'bev_embed': bev_embed,
                'all_cls_scores': outputs_classes,
                'all_bbox_preds': outputs_coords,
                'enc_cls_scores': None,
                'enc_bbox_preds': None,
            }
        else:
            outs = {
                'bev_embed': bev_embed,
                'all_cls_scores': outputs_classes,
                'all_bbox_preds': outputs_coords,
                'enc_cls_scores': None,
                'enc_bbox_preds': None,
            }

        return outs

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
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
        gt_c = gt_bboxes.shape[-1]

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        
        targets_indices = (pos_inds, assign_result.gt_inds[assign_result.gt_inds > 0] - 1)

        return (labels, label_weights, bbox_targets, bbox_weights, targets_indices,
                pos_inds, neg_inds)

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

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, targets_indices, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, targets_indices, num_total_pos, num_total_neg, pos_inds_list)
        
    def loss_single(self,
                    cls_scores,
                    bbox_preds,
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
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, pos_inds_list) = cls_reg_targets
        
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

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
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan,
                                                               :10], bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox
        
    def get_owod_target(self,
                        cls_scores_list,
                        backbone_feature,
                        img_metas_list,
                        gt_bboxes_list,
                        gt_labels_list
                        ):
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
        owod_device = cls_scores_list[0].device
        
        # read from proposal_file
        proposal_file_name = img_metas_list[0]['proposal']
        with open(proposal_file_name, 'rb') as f:
                proposal = pickle.load(f)
        
        # Lidar RPN proposal
        all_proposal_bbox = torch.stack(proposal, dim=0).to(owod_device)
        
        values = all_proposal_bbox[:, -1]
        sorted_indices = values.argsort(descending=True)
        rpn_select_num = gt_bboxes_list[0].shape[0] + 30
        proposal_bbox = all_proposal_bbox[sorted_indices[:rpn_select_num]]
        proposal_bbox = proposal_bbox[:,:9]
        
        queries = torch.arange(proposal_bbox.shape[0])
        querie_box = queries.clone().to(owod_device)
        unmatched_indices = querie_box
        
        gt_boxes = gt_bboxes_list[0].clone()
        boxes = proposal_bbox

        box_3d_points = get_corners_gt(boxes)
        gt_box_3d_points = get_corners_gt(gt_boxes)

        x_min, _ = torch.min(box_3d_points[..., 0], dim=1)
        y_min, _ = torch.min(box_3d_points[..., 1], dim=1)
        x_max, _ = torch.max(box_3d_points[..., 0], dim=1)
        y_max, _ = torch.max(box_3d_points[..., 1], dim=1)
        gt_x_min, _ = torch.min(gt_box_3d_points[..., 0], dim=1)
        gt_y_min, _ = torch.min(gt_box_3d_points[..., 1], dim=1)
        gt_x_max, _ = torch.max(gt_box_3d_points[..., 0], dim=1)
        gt_y_max, _ = torch.max(gt_box_3d_points[..., 1], dim=1)

        unmatched_boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)
        gt_boxes_img = torch.stack([gt_x_min, gt_y_min, gt_x_max, gt_y_max], dim=1)
      
        # Add the offset to all tensors in one go
        unmatched_boxes += 51.2
        gt_boxes_img += 51.2
        
        bb = unmatched_boxes.clone()
        

        def bbox_iou_vectorized(bboxes1, bboxes2):
            bboxes1 = bboxes1[:, None, :]
            bboxes2 = bboxes2[None, :, :]
            
            x1 = torch.max(bboxes1[..., 0], bboxes2[..., 0])
            y1 = torch.max(bboxes1[..., 1], bboxes2[..., 1])
            x2 = torch.min(bboxes1[..., 2], bboxes2[..., 2])
            y2 = torch.min(bboxes1[..., 3], bboxes2[..., 3])
            
            intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
            area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
            area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
            union = area1 + area2 - intersection
            
            return intersection / union

        if len(unmatched_indices) > 0 and len(gt_boxes_img) > 0:
            ious = bbox_iou_vectorized(bb[unmatched_indices], gt_boxes_img)
            
            max_ious, _ = torch.max(ious, dim=1)
            keep_indices = torch.nonzero(max_ious < 0.01).squeeze()
            
            unmatched_indices = unmatched_indices[keep_indices]
        
        xmin = (bb[unmatched_indices, 0] * 10).long()
        ymin = (bb[unmatched_indices, 1] * 10).long()
        xmax = (bb[unmatched_indices, 2] * 10).long()
        ymax = (bb[unmatched_indices, 3] * 10).long()
        
        if  xmin.dim() != 0 and xmin.shape[0] > self.topk:
            upsaple = nn.Upsample(size=(int(self.real_h*10),int(self.real_w*10)), mode='bilinear', align_corners=True) 
            bev_feat_up = upsaple(backbone_feature.unsqueeze(0)) # 1x1x1024x1024
            bev_feat = bev_feat_up.squeeze(0).squeeze(0) # 1024x1024
            
            scaled_boxes = (bb[unmatched_indices] * 10).long()
            xmin, ymin, xmax, ymax = scaled_boxes.t()

            bev_feat_slices = [bev_feat[ymin[i]:ymax[i], xmin[i]:xmax[i]] for i in range(len(xmin))]

            means_bb_slices = torch.tensor([torch.mean(slice) for slice in bev_feat_slices], device=unmatched_boxes.device)        
            
            mask = torch.isnan(means_bb_slices)
            means_bb_slices[mask] = 10e10

            _, topk_inds = torch.topk(means_bb_slices, self.topk)
            topk_inds = unmatched_indices[topk_inds]
            topk_inds = topk_inds.to(owod_device)
            
            # gt_label_list + owod_targets
            owod_targets = proposal_bbox[topk_inds]

            owod_num = owod_targets.shape[0]
            original_labels_tensor = gt_labels_list[0].clone()
            original_labels_tensor_device = original_labels_tensor.device
            ow_label = torch.full((owod_num,), cls_scores_list[0].shape[1] - 1).to(original_labels_tensor_device)
            ow_label_new_tensor = torch.cat((original_labels_tensor, ow_label))
            ow_gt_labels_list = [ow_label_new_tensor]
            
            original_bboxes_tensor = gt_bboxes_list[0].clone()
            original_bboxes_tensor_device = original_bboxes_tensor.device
            ow_bbox_new_tensor = torch.cat((original_bboxes_tensor, owod_targets.to(original_bboxes_tensor_device)), dim=0)
            ow_gt_bboxes_list = [ow_bbox_new_tensor]
            
            ow_weight = all_proposal_bbox[topk_inds,9:].t()
            
            return ow_gt_labels_list, ow_gt_bboxes_list, ow_weight
        else:
            return gt_labels_list, gt_bboxes_list, 0
    
    def loss_single_owod(self,
                        cls_scores,
                        bbox_preds,
                        gt_bboxes_list,
                        gt_labels_list,
                        bev_heatmap_list,
                        img_metas_list,
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
        
        ow_gt_labels_list, ow_gt_bboxes_list , ow_weight =  self.get_owod_target(cls_scores_list,
                                    bev_heatmap_list, img_metas_list, gt_bboxes_list, gt_labels_list)

        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           ow_gt_bboxes_list, ow_gt_labels_list, gt_bboxes_ignore_list)
        
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, targets_indices,
         num_total_pos, num_total_neg, pos_inds_list) = cls_reg_targets

        ow_match_labels = labels_list[0][pos_inds_list[0]].clone()
        ow_match_labels_idx = pos_inds_list[0][ow_match_labels == self.num_classes - 1]

        # class init - list to tensor
        labels = torch.cat(labels_list, 0) 
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        
        # with soft weight
        label_weights[ow_match_labels_idx] = ow_weight
        bbox_weights[ow_match_labels_idx] = 0

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
        bbox_pos_num = num_total_pos - self.topk
        
        # normalization purposes
        bbox_pos_num = loss_cls.new_tensor([bbox_pos_num])
        bbox_pos_num = torch.clamp(reduce_mean(bbox_pos_num), min=1).item()
        
        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=bbox_pos_num)

        # all loss
        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
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

        all_cls_scores = preds_dicts['all_cls_scores']  # 6 x b x num_query x 10
        all_bbox_preds = preds_dicts['all_bbox_preds']  # 6 x b x num_query x 10
        enc_cls_scores = preds_dicts['enc_cls_scores']  # none
        enc_bbox_preds = preds_dicts['enc_bbox_preds']  # none
        
        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]
        
        if self.owod:
            bev_heatmap = preds_dicts['bev_embed'] # (bev_h x bev_w) x b x channel
            dense_heatmap_bev = torch.mean(bev_heatmap.detach().permute(1, 2, 0).view(all_bbox_preds.shape[1], bev_heatmap.shape[2], self.bev_h, self.bev_w), dim=1) # b x bev_h x bev_w  均值bev
            bev_heatmap_list = [ dense_heatmap_bev for _ in range(num_dec_layers) ] # 6 x b x channel x bev_h x bev_w
            img_metas_list = [ img_metas for _ in range(num_dec_layers)] # 6

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)] # 6 x b x gt_num x 9
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)] # 6 x b x 7
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        if self.owod:        
            losses_cls, losses_bbox = multi_apply(
                self.loss_single_owod, all_cls_scores, all_bbox_preds,
                all_gt_bboxes_list, all_gt_labels_list,
                bev_heatmap_list, img_metas_list,
                all_gt_bboxes_ignore_list)
        else:
            losses_cls, losses_bbox = multi_apply(
                self.loss_single, all_cls_scores, all_bbox_preds,
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

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
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

            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']

            ret_list.append([bboxes, scores, labels])

        return ret_list
    
def get_corners_pred_bb(bbox):

    # center in the bev
    cx = bbox[..., 0:1]
    cy = bbox[..., 1:2]
    cz = bbox[..., 4:5]

    # size
    w = bbox[..., 2:3]
    l = bbox[..., 3:4]
    h = bbox[..., 5:6]

    # exp
    w = w.exp()
    l = l.exp()
    h = h.exp()

    sin, cos = bbox[:,6], bbox[:,7]
    rot = torch.atan2(sin, cos)

    rot_degrees = rot * (180 / np.pi)

    for i in range(len(rot_degrees)):
        if rot_degrees[i] >= 0:
            rot_degrees[i] -= 90
        else:
            rot_degrees[i] += 90

    rot = rot_degrees / (180 / np.pi)

    local_corners = torch.stack([-l/2, -w/2, -h/2, l/2, -w/2, -h/2, l/2, w/2, -h/2, -l/2, w/2, -h/2,
                                 -l/2, -w/2, h/2, l/2, -w/2, h/2, l/2, w/2, h/2, -l/2, w/2, h/2], dim=-1)

    local_corners = local_corners.view(*l.shape[:-1], 8, 3)
    local_corners = local_corners.permute(0, 2, 1)  

    zeros, ones = torch.zeros_like(rot), torch.ones_like(rot)
    rotation_matrix = torch.stack([cos, -sin, zeros, sin, cos, zeros, zeros, zeros, ones], dim=-1)
    rotation_matrix = rotation_matrix.view(-1, 3, 3) 

    rotated_corners = torch.einsum('bij,bjk->bik', rotation_matrix, local_corners)

    xyz = torch.cat([cx, cy, cz], dim=-1)
    global_corners = rotated_corners.permute(0, 2, 1) + xyz.unsqueeze(1)

    return global_corners

def get_corners_gt(bbox):

    # center in the bev
    cx = bbox[..., 0:1]
    cy = bbox[..., 1:2]
    cz = bbox[..., 2:3]

    # size
    w = bbox[..., 3:4]
    l = bbox[..., 4:5]
    h = bbox[..., 5:6]

    # rot
    rot = bbox[:,6:7]
    cos, sin = torch.cos(rot), torch.sin(rot)

    lwh = torch.cat([l, w, h], dim=-1)
    x, y, z = lwh[:, 0] / 2, lwh[:, 1] / 2, lwh[:, 2] / 2
    local_corners = torch.stack([-x, -y, -z, x, -y, -z, x, y, -z, -x, y, -z,
                                  -x, -y, z, x, -y, z, x, y, z, -x, y, z], dim=-1)

    local_corners = local_corners.view(*lwh.shape[:-1], 8, 3)
    local_corners = local_corners.permute(0, 2, 1)  

    rot_degrees = rot * (180 / np.pi)

    for i in range(len(rot_degrees)):
        if rot_degrees[i] >= 0:
            rot_degrees[i] -= 90
        else:
            rot_degrees[i] += 90

    rot = rot_degrees / (180 / np.pi)

    cos, sin = rot.cos(), rot.sin()

    zeros, ones = torch.zeros_like(rot), torch.ones_like(rot)
    rotation_matrix = torch.stack([cos, -sin, zeros, sin, cos, zeros, zeros, zeros, ones], dim=-1)
    rotation_matrix = rotation_matrix.view(*rot.shape[:-1], 3, 3)

    rotated_corners = torch.einsum('bij,bjk->bik', rotation_matrix, local_corners)

    xyz = torch.cat([cx, cy, cz], dim=-1)
    global_corners = rotated_corners.permute(0, 2, 1) + xyz.unsqueeze(1)

    return global_corners

