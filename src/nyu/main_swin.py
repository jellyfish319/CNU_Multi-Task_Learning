import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import os
from utils import SegMetric, DepthMetric, NormalMetric, SegLoss, DepthLoss, NormalLoss
from aspp import DeepLabHead
from create_dataset import NYUv2

from LibMTL.model.swin import SwinTransformerV2

from LibMTL import Trainer
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args

def parse_args(parser):
    parser.add_argument('--aug', action='store_true', default=False, help='data augmentation')
    parser.add_argument('--train_bs', default=2, type=int, help='batch size for training (adjusted for Swin-B)')
    parser.add_argument('--test_bs', default=2, type=int, help='batch size for test (adjusted for Swin-B)')
    parser.add_argument('--epochs', default=100, type=int, help='training epochs')
    parser.add_argument('--dataset_path', default='/', type=str, help='dataset path')
    return parser.parse_args()

class Swin_V2_B_Encoder(SwinTransformerV2):
    def __init__(self, pretrained_path="/path/to/swin_pretrained_model.pth", **kwargs):
        if pretrained_path:
            expanded_pretrained_path = os.path.expanduser(pretrained_path)
            print(f"Loading pretrained weights from: {expanded_pretrained_path}")
        else:
            expanded_pretrained_path = None
            print("No pretrained weights will be loaded.")
        expanded_abs_path = None
        if pretrained_path:
            expanded_abs_path = os.path.expanduser(pretrained_path)
            print(f"DEBUG main_swin.py: Expanded path to: {expanded_abs_path}")

        super(Swin_V2_B_Encoder, self).__init__(
            pretrain_img_size=224,
            patch_size=4,
            in_chans=3,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            ape=False,
            patch_norm=True,
            out_indices=(3,),
            qk_scale=None,
            use_checkpoint=False,
            frozen_stages=-1,
            pretrain_window_size=[12, 12, 12, 6],
            **kwargs
        )
        
        if expanded_abs_path and os.path.exists(expanded_abs_path):
            print(f"DEBUG main_swin.py: Calling self.init_weights with existing file: {expanded_abs_path}")
            self.init_weights(pretrained=expanded_abs_path)
        elif expanded_abs_path:
            print(f"Warning: Pretrained path {expanded_abs_path} specified but file not found.")
            self.init_weights(pretrained=None)
        else:
            self.init_weights(pretrained=None)

    def forward(self, x):
        features_list = super().forward(x)
        return features_list[-1]

def main(params):
    kwargs, optim_param, scheduler_param = prepare_args(params)

    nyuv2_train_set = NYUv2(root=params.dataset_path, mode='train', augmentation=params.aug)
    nyuv2_test_set = NYUv2(root=params.dataset_path, mode='test', augmentation=False)

    nyuv2_train_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_train_set,
        batch_size=params.train_bs,
        shuffle=True,
        num_workers=params.num_workers if hasattr(params, 'num_workers') else 2,
        pin_memory=True,
        drop_last=True)

    nyuv2_test_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_test_set,
        batch_size=params.test_bs,
        shuffle=False,
        num_workers=params.num_workers if hasattr(params, 'num_workers') else 2,
        pin_memory=True)

    task_dict = {
        'segmentation': {'metrics': ['mIoU', 'pixAcc'],
                         'metrics_fn': SegMetric(num_classes=13),
                         'loss_fn': SegLoss(),
                         'weight': [1, 1]},
        'depth': {'metrics': ['abs_err', 'rel_err'],
                  'metrics_fn': DepthMetric(),
                  'loss_fn': DepthLoss(),
                  'weight': [0, 0]},
        'normal': {'metrics': ['mean', 'median', '<11.25', '<22.5', '<30'],
                   'metrics_fn': NormalMetric(),
                   'loss_fn': NormalLoss(),
                   'weight': [0, 0, 1, 1, 1]}
    }

    def encoder_class_fn():
        return Swin_V2_B_Encoder()

    encoder_final_out_channels = 1024
    num_out_channels = {'segmentation': 13, 'depth': 1, 'normal': 3}
    decoders = nn.ModuleDict({task: DeepLabHead(encoder_final_out_channels,
                                                num_out_channels[task])
                              for task in list(task_dict.keys())})

    class MTLTrainer(Trainer):
        def __init__(self, task_dict, weighting, architecture, encoder_class,
                     decoders, rep_grad, multi_input, optim_param,
                     scheduler_param, **kwargs_trainer):
            super(MTLTrainer, self).__init__(task_dict=task_dict,
                                             weighting=weighting,
                                             architecture=architecture,
                                             encoder_class=encoder_class,
                                             decoders=decoders,
                                             rep_grad=rep_grad,
                                             multi_input=multi_input,
                                             optim_param=optim_param,
                                             scheduler_param=scheduler_param,
                                             **kwargs_trainer)

        def process_preds(self, preds):
            img_size = (288, 384)
            for task in self.task_name:
                preds[task] = F.interpolate(preds[task], img_size, mode='bilinear', align_corners=True)
            return preds

    model = MTLTrainer(task_dict=task_dict,
                       weighting=params.weighting,
                       architecture=params.arch,
                       encoder_class=encoder_class_fn,
                       decoders=decoders,
                       rep_grad=params.rep_grad,
                       multi_input=params.multi_input,
                       optim_param=optim_param,
                       scheduler_param=scheduler_param,
                       save_path=params.save_path,
                       load_path=params.load_path,
                       **kwargs) # [

    if params.mode == 'train':
        model.train(nyuv2_train_loader, nyuv2_test_loader, params.epochs)
    elif params.mode == 'test':
        if params.load_path is None:
            raise ValueError("Test mode requires a load_path to a trained model.")
        model.test(nyuv2_test_loader)
    else:
        raise ValueError(f"Unknown mode: {params.mode}. Choose 'train' or 'test'.")

if __name__ == "__main__":
    params = parse_args(LibMTL_args)

    set_device(params.gpu_id)
    set_random_seed(params.seed)

    main(params)