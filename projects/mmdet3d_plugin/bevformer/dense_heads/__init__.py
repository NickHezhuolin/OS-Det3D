from .bev_head import BEVHead
from .bevformer_head import BEVFormerHead
from .owbevformer_head_unk_gt import OWBEVFormerHead_UnkGT

# based on BEVFormer & OWDETR
from .owbevformer_headV1 import OWBEVFormerHeadV1
from .owbevformer_headV1_rpn_w_bbox import OWBEVFormerHeadV1RPN # with bbox refine
from .owbevformer_headV1_rpn_wo_bbox import OWBEVFormerHeadV1RPNV1  # with out bbox refine
from .owbevformer_headV1_rpn_wo_ow import OWBEVFormerHeadV1RPNV1_Without_OWDETR_Select # with out owdetr select
from .owbevformer_headV1_rpn_wo_bbox_w_sw import OWBEVFormerHeadV1RPNV1_with_soft_weight # with soft weight
from .owbevformer_headV1_rpn_wo_bbox_wo_nc import OWBEVFormerHeadV1RPNV1_without_nc_branch # without nc branch
from .owbevformer_headV1_rpn_wo_bbox_w_sw_rescale import OWBEVFormerHeadV1RPNV1_with_soft_weight_rescale # with soft weight rescale
from .owbevformer_headV1_rpn_w_bbox_w_sw import OWBEVFormerHeadV1RPN_w_bb_with_soft_weight # with bbox refine with soft weight rescale

# based on owbevformer_headV1_rpn_wo_bbox + new nc branch
from .owbevformer_headV2 import OWBEVFormerHeadV2