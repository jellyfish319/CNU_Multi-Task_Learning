import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from functools import partial

from create_dataset import NYUv2
from LibMTL.model.pvtv2 import PyramidVisionTransformerV2

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class PVT_V2_B2_Encoder(PyramidVisionTransformerV2):
    def __init__(self, pretrained_path="~/pretrained/pvt_v2_b2.pth", **kwargs):
        super(PVT_V2_B2_Encoder, self).__init__(
            img_size=224, patch_size=4, embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1,
            num_stages=4, pretrained=pretrained_path, **kwargs
        )

    def forward(self, x):
        features_list = self.forward_features(x)
        return features_list[-1]

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates=[12, 24, 36]):
        super(ASPP, self).__init__()
        out_channels = 256
        
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)))

        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res[-1] = F.interpolate(res[-1], size=x.shape[2:], mode='bilinear', align_corners=False)
        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__()
        self.aspp = ASPP(in_channels)
        
        # 원래 구조에 맞춤: decoders.X.1, 2, 4
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),  # .1
            nn.BatchNorm2d(256),                            # .2
            nn.ReLU(inplace=True),                          # .3 (파라미터 없음)
            nn.Conv2d(256, num_classes, 1)                  # .4
        )
        
    def forward(self, x):
        x = self.aspp(x)
        x = self.classifier(x)
        return x

class SimpleMultiTaskModel(nn.Module):
    def __init__(self):
        super(SimpleMultiTaskModel, self).__init__()
        self.encoder = PVT_V2_B2_Encoder()
        self.seg_decoder = DeepLabHead(512, 13)
        self.depth_decoder = DeepLabHead(512, 1)
        self.normal_decoder = DeepLabHead(512, 3)
        
    def forward(self, x):
        features = self.encoder(x)
        seg_out = self.seg_decoder(features)
        depth_out = self.depth_decoder(features)
        normal_out = self.normal_decoder(features)
        
        seg_out = F.interpolate(seg_out, size=x.shape[2:], mode='bilinear', align_corners=False)
        depth_out = F.interpolate(depth_out, size=x.shape[2:], mode='bilinear', align_corners=False)
        normal_out = F.interpolate(normal_out, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return {'segmentation': seg_out, 'depth': depth_out, 'normal': normal_out}

class SimpleNYUVisualizer:
    def __init__(self, model_path, dataset_path, device='cuda'):
        self.device = device
        self.model_path = model_path
        self.dataset_path = dataset_path
        
        set_seed(42)
        
        self.seg_colors = np.array([
            [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128]
        ], dtype=np.uint8)
        
        self.load_model()
    
    def load_model(self):
        print(f"Loading model from {self.model_path}")
        
        self.model = SimpleMultiTaskModel()
        
        # 키 매핑
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            state_dict = checkpoint
            
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = None
                
                if key.startswith('encoder.'):
                    new_key = key
                elif key.startswith('decoders.segmentation.'):
                    if key.startswith('decoders.segmentation.0.'):
                        new_key = key.replace('decoders.segmentation.0.', 'seg_decoder.aspp.')
                    elif key.startswith('decoders.segmentation.1.'):
                        new_key = key.replace('decoders.segmentation.1.', 'seg_decoder.classifier.0.')
                    elif key.startswith('decoders.segmentation.2.'):
                        new_key = key.replace('decoders.segmentation.2.', 'seg_decoder.classifier.1.')
                    elif key.startswith('decoders.segmentation.4.'):
                        new_key = key.replace('decoders.segmentation.4.', 'seg_decoder.classifier.3.')
                elif key.startswith('decoders.depth.'):
                    if key.startswith('decoders.depth.0.'):
                        new_key = key.replace('decoders.depth.0.', 'depth_decoder.aspp.')
                    elif key.startswith('decoders.depth.1.'):
                        new_key = key.replace('decoders.depth.1.', 'depth_decoder.classifier.0.')
                    elif key.startswith('decoders.depth.2.'):
                        new_key = key.replace('decoders.depth.2.', 'depth_decoder.classifier.1.')
                    elif key.startswith('decoders.depth.4.'):
                        new_key = key.replace('decoders.depth.4.', 'depth_decoder.classifier.3.')
                elif key.startswith('decoders.normal.'):
                    if key.startswith('decoders.normal.0.'):
                        new_key = key.replace('decoders.normal.0.', 'normal_decoder.aspp.')
                    elif key.startswith('decoders.normal.1.'):
                        new_key = key.replace('decoders.normal.1.', 'normal_decoder.classifier.0.')
                    elif key.startswith('decoders.normal.2.'):
                        new_key = key.replace('decoders.normal.2.', 'normal_decoder.classifier.1.')
                    elif key.startswith('decoders.normal.4.'):
                        new_key = key.replace('decoders.normal.4.', 'normal_decoder.classifier.3.')
                
                if new_key is not None:
                    new_state_dict[new_key] = value
            
            missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
            
            print(f"Successfully loaded model!")
            print(f"Total mapped keys: {len(new_state_dict)}")
            print(f"Missing keys: {len(missing_keys)}")
            print(f"Unexpected keys: {len(unexpected_keys)}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using random weights for demo")
        
        self.model.to(self.device)
        self.model.eval()
    
    # 나머지 메서드들은 동일...
    def load_dataset(self, mode='val'):
        return NYUv2(root=self.dataset_path, mode=mode, augmentation=False)
    
    def predict(self, image):
        with torch.no_grad():
            image = image.unsqueeze(0).to(self.device)
            predictions = self.model(image)
            for task in predictions:
                predictions[task] = predictions[task].squeeze(0)
            return predictions
    
    def visualize_segmentation(self, seg_pred):
        if seg_pred.dim() == 3:
            seg_pred = torch.argmax(seg_pred, dim=0)
        seg_pred = seg_pred.cpu().numpy()
        seg_pred[seg_pred == -1] = 0
        seg_pred = np.clip(seg_pred, 0, len(self.seg_colors) - 1).astype(np.uint8)
        h, w = seg_pred.shape
        colored_seg = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(len(self.seg_colors)):
            mask = seg_pred == i
            colored_seg[mask] = self.seg_colors[i]
        return colored_seg
    
    def visualize_depth(self, depth_pred):
        if depth_pred.dim() == 3:
            depth_pred = depth_pred.squeeze(0)
        depth_pred = depth_pred.cpu().numpy()
        depth_min, depth_max = depth_pred.min(), depth_pred.max()
        if depth_max > depth_min:
            depth_normalized = (depth_pred - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth_pred)
        depth_colored = plt.cm.viridis(depth_normalized)[:, :, :3]
        return (depth_colored * 255).astype(np.uint8)
    
    def visualize_normal(self, normal_pred):
        if normal_pred.dim() == 3:
            normal_pred = normal_pred.permute(1, 2, 0)
        normal_pred = normal_pred.cpu().numpy()
        normal_normalized = (normal_pred + 1) / 2
        normal_normalized = np.clip(normal_normalized, 0, 1)
        return (normal_normalized * 255).astype(np.uint8)
    
    def denormalize_image(self, image):
        if image.dim() == 3:
            image = image.permute(1, 2, 0)
        image = image.cpu().numpy()
        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)
    
    def check_dataset_info(self):
        dataset = self.load_dataset('val')
        print(f"Dataset size: {len(dataset)}")
        if len(dataset) > 0:
            image, targets = dataset[0]
            print(f"Image shape: {image.shape}, dtype: {image.dtype}")
            print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
            for task, target in targets.items():
                print(f"{task} shape: {target.shape}, dtype: {target.dtype}")
                if task == 'segmentation':
                    unique_vals = torch.unique(target)
                    print(f"{task} unique values: {unique_vals}")
                else:
                    print(f"{task} range: [{target.min():.3f}, {target.max():.3f}]")
    
    def visualize_sample(self, sample_idx, save_path=None):
        dataset = self.load_dataset('val')
        if sample_idx >= len(dataset):
            raise ValueError(f"Sample index {sample_idx} out of range. Dataset has {len(dataset)} samples.")
        
        print(f"Visualizing sample {sample_idx}")
        image, targets = dataset[sample_idx]
        predictions = self.predict(image)
        
        image_np = self.denormalize_image(image)
        seg_pred_vis = self.visualize_segmentation(predictions['segmentation'])
        depth_pred_vis = self.visualize_depth(predictions['depth'])
        normal_pred_vis = self.visualize_normal(predictions['normal'])
        
        seg_gt_vis = self.visualize_segmentation(targets['segmentation'])
        depth_gt_vis = self.visualize_depth(targets['depth'])
        normal_gt_vis = self.visualize_normal(targets['normal'])
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        axes[0, 0].imshow(image_np)
        axes[0, 0].set_title('Input Image', fontsize=14)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(seg_pred_vis)
        axes[0, 1].set_title('Segmentation Prediction', fontsize=14)
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(depth_pred_vis)
        axes[0, 2].set_title('Depth Prediction', fontsize=14)
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(normal_pred_vis)
        axes[0, 3].set_title('Normal Prediction', fontsize=14)
        axes[0, 3].axis('off')
        
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(seg_gt_vis)
        axes[1, 1].set_title('Segmentation GT', fontsize=14)
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(depth_gt_vis)
        axes[1, 2].set_title('Depth GT', fontsize=14)
        axes[1, 2].axis('off')
        
        axes[1, 3].imshow(normal_gt_vis)
        axes[1, 3].set_title('Normal GT', fontsize=14)
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Simple NYU v2 Visualization')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--sample_idx', type=int, default=0, help='Sample index')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--mode', type=str, default='info', choices=['info', 'single'], help='Mode')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    visualizer = SimpleNYUVisualizer(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        device=device
    )
    
    if args.mode == 'info':
        visualizer.check_dataset_info()
    elif args.mode == 'single':
        visualizer.visualize_sample(args.sample_idx)

if __name__ == "__main__":
    main()