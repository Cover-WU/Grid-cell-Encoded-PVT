_base_ = './mask_rcnn_pcpvt_s_fpn_1x_coco_pvt_setting.py'

model = dict(
    pretrained='pretrained/alt_gvt_base.pth',
    backbone=dict(
        type='alt_gvt_base',
        style='pytorch'),
    neck=dict(
        in_channels=[96, 192, 384, 768],
        out_channels=256,))
