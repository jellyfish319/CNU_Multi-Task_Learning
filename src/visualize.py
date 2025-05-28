import unittest
from unittest.mock import patch, MagicMock
import torch
import numpy as np
import os
from examples.nyu.create_dataset import NYUv2, RandomScaleCrop

class TestNYUv2Dataset(unittest.TestCase):
    
    def setUp(self):
        self.test_root = '/fake/dataset/path'
        self.mock_image = np.random.rand(480, 640, 3).astype(np.float32)
        self.mock_semantic = np.random.randint(0, 13, (480, 640)).astype(np.float32)
        self.mock_depth = np.random.rand(480, 640, 1).astype(np.float32)
        self.mock_normal = np.random.rand(480, 640, 3).astype(np.float32)
    
    @patch('os.listdir')
    @patch('os.path.expanduser')
    def test_dataset_init_train_mode(self, mock_expanduser, mock_listdir):
        mock_expanduser.return_value = self.test_root
        mock_listdir.return_value = ['0.npy', '1.npy', '2.npy']
        
        dataset = NYUv2(root=self.test_root, mode='train', augmentation=False)
        
        self.assertEqual(dataset.mode, 'train')
        self.assertEqual(dataset.root, self.test_root)
        self.assertEqual(dataset.augmentation, False)
        self.assertEqual(len(dataset.index_list), 3)
        self.assertEqual(dataset.data_path, self.test_root + '/train')
    
    @patch('os.listdir')
    @patch('os.path.expanduser')
    def test_dataset_init_val_mode(self, mock_expanduser, mock_listdir):
        mock_expanduser.return_value = self.test_root
        mock_listdir.return_value = ['0.npy', '1.npy']
        
        dataset = NYUv2(root=self.test_root, mode='val', augmentation=True)
        
        self.assertEqual(dataset.mode, 'val')
        self.assertEqual(dataset.augmentation, True)
        self.assertEqual(len(dataset.index_list), 2)
        self.assertEqual(dataset.data_path, self.test_root + '/val')
    
    @patch('numpy.load')
    @patch('os.listdir')
    @patch('os.path.expanduser')
    def test_getitem_without_augmentation(self, mock_expanduser, mock_listdir, mock_load):
        mock_expanduser.return_value = self.test_root
        mock_listdir.return_value = ['0.npy']
        
        def load_side_effect(path):
            if 'image' in path:
                return self.mock_image
            elif 'label' in path:
                return self.mock_semantic
            elif 'depth' in path:
                return self.mock_depth
            elif 'normal' in path:
                return self.mock_normal
        
        mock_load.side_effect = load_side_effect
        
        dataset = NYUv2(root=self.test_root, mode='train', augmentation=False)
        image, targets = dataset[0]
        
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.dtype, torch.float32)
        self.assertEqual(image.shape, (3, 480, 640))  # channels first
        
        self.assertIn('segmentation', targets)
        self.assertIn('depth', targets)
        self.assertIn('normal', targets)
        
        self.assertEqual(targets['segmentation'].shape, (480, 640))
        self.assertEqual(targets['depth'].shape, (1, 480, 640))
        self.assertEqual(targets['normal'].shape, (3, 480, 640))
    
    @patch('torch.rand')
    @patch('numpy.load')
    @patch('os.listdir')
    @patch('os.path.expanduser')
    def test_getitem_with_augmentation_no_flip(self, mock_expanduser, mock_listdir, mock_load, mock_rand):
        mock_expanduser.return_value = self.test_root
        mock_listdir.return_value = ['0.npy']
        mock_rand.return_value = torch.tensor([0.6])  # No flip
        
        def load_side_effect(path):
            if 'image' in path:
                return self.mock_image
            elif 'label' in path:
                return self.mock_semantic
            elif 'depth' in path:
                return self.mock_depth
            elif 'normal' in path:
                return self.mock_normal
        
        mock_load.side_effect = load_side_effect
        
        dataset = NYUv2(root=self.test_root, mode='train', augmentation=True)
        image, targets = dataset[0]
        
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.dtype, torch.float32)
    
    @patch('torch.rand')
    @patch('numpy.load')
    @patch('os.listdir')
    @patch('os.path.expanduser')
    def test_getitem_with_augmentation_with_flip(self, mock_expanduser, mock_listdir, mock_load, mock_rand):
        mock_expanduser.return_value = self.test_root
        mock_listdir.return_value = ['0.npy']
        mock_rand.return_value = torch.tensor([0.3])  # Flip
        
        def load_side_effect(path):
            if 'image' in path:
                return self.mock_image
            elif 'label' in path:
                return self.mock_semantic
            elif 'depth' in path:
                return self.mock_depth
            elif 'normal' in path:
                return self.mock_normal
        
        mock_load.side_effect = load_side_effect
        
        dataset = NYUv2(root=self.test_root, mode='train', augmentation=True)
        image, targets = dataset[0]
        
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.dtype, torch.float32)
    
    @patch('os.listdir')
    @patch('os.path.expanduser')
    def test_dataset_length(self, mock_expanduser, mock_listdir):
        mock_expanduser.return_value = self.test_root
        mock_listdir.return_value = ['0.npy', '1.npy', '2.npy', '3.npy', '4.npy']
        
        dataset = NYUv2(root=self.test_root, mode='train')
        
        self.assertEqual(len(dataset), 5)


class TestRandomScaleCrop(unittest.TestCase):
    
    def setUp(self):
        self.transform = RandomScaleCrop(scale=[1.0, 1.2])
        self.img = torch.rand(3, 480, 640)
        self.label = torch.randint(0, 13, (480, 640))
        self.depth = torch.rand(1, 480, 640)
        self.normal = torch.rand(3, 480, 640)
    
    def test_random_scale_crop_output_shapes(self):
        img_aug, label_aug, depth_aug, normal_aug = self.transform(
            self.img, self.label, self.depth, self.normal
        )
        
        # Output shapes should match input shapes
        self.assertEqual(img_aug.shape, self.img.shape)
        self.assertEqual(label_aug.shape, self.label.shape)
        self.assertEqual(depth_aug.shape, self.depth.shape)
        self.assertEqual(normal_aug.shape, self.normal.shape)
    
    def test_random_scale_crop_depth_scaling(self):
        # Test that depth is properly scaled
        original_depth_mean = self.depth.mean()
        
        # Mock random to always select scale 1.2
        with patch('random.randint', return_value=1):
            img_aug, label_aug, depth_aug, normal_aug = self.transform(
                self.img, self.label, self.depth, self.normal
            )
            
            # Depth should be scaled by 1/1.2
            expected_scale = 1.0 / 1.2
            self.assertLess(depth_aug.mean(), original_depth_mean)


if __name__ == '__main__':
    unittest.main()