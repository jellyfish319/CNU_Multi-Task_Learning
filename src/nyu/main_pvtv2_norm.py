import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from utils import SegMetric, DepthMetric, NormalMetric, SegLoss, DepthLoss, NormalLoss
from aspp import DeepLabHead
from create_dataset import NYUv2
from LibMTL.model.pvtv2 import PyramidVisionTransformerV2



from LibMTL import Trainer
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args


def parse_args(parser):
    parser.add_argument('--aug', action='store_true', default=False, help='data augmentation')
    parser.add_argument('--train_bs', default=4, type=int, help='batch size for training')
    parser.add_argument('--test_bs', default=4, type=int, help='batch size for test')
    parser.add_argument('--epochs', default=100, type=int, help='training epochs')
    parser.add_argument('--dataset_path', default='/', type=str, help='dataset path')
    return parser.parse_args()

class PVT_V2_B2_Encoder(PyramidVisionTransformerV2):
    def __init__(self, pretrained_path="~/pretrained/pvt_v2_b2.pth", **kwargs):
        super(PVT_V2_B2_Encoder, self).__init__(
            img_size=224,
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
            num_stages=4,
            pretrained=pretrained_path,
            **kwargs
        )

    def forward(self, x):
        features_list = self.forward_features(x)
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
        'normal': {'metrics': ['mean', 'median', '<11.25', '<22.5', '<30'], 
                   'metrics_fn': NormalMetric(),
                   'loss_fn': NormalLoss(),
                   'weight': [0, 0, 1, 1, 1]} 
    }

    def encoder_class():
        return PVT_V2_B2_Encoder()

    pvt_final_out_channels = 512
    num_out_channels = {'normal': 3}
    decoders = nn.ModuleDict({task: DeepLabHead(pvt_final_out_channels,
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
                       encoder_class=encoder_class,
                       decoders=decoders,
                       rep_grad=params.rep_grad,
                       multi_input=params.multi_input,
                       optim_param=optim_param,
                       scheduler_param=scheduler_param,
                       save_path=params.save_path,
                       load_path=params.load_path,
                       **kwargs)

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