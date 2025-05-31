import torch
import os

# 경로 확장
model_path = os.path.expanduser("~/save/best_pvtv2b2_pretrained.pt")
print(f"Looking for model at: {model_path}")
print(f"File exists: {os.path.exists(model_path)}")

if not os.path.exists(model_path):
    print("File not found. Let's check the save directory:")
    save_dir = os.path.expanduser("~/save/")
    if os.path.exists(save_dir):
        files = os.listdir(save_dir)
        print(f"Files in {save_dir}:")
        for f in files:
            print(f"  {f}")
    else:
        print(f"Save directory {save_dir} does not exist")
    exit()

checkpoint = torch.load(model_path, map_location='cpu')

print("=== Checkpoint Structure Analysis ===")
print(f"Checkpoint type: {type(checkpoint)}")

if isinstance(checkpoint, dict):
    print(f"Top-level keys: {list(checkpoint.keys())}")
    
    # 실제 state_dict 찾기
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

print(f"\nTotal parameters: {len(state_dict)}")

# 키들을 그룹별로 분석
encoder_keys = [k for k in state_dict.keys() if k.startswith('encoder.')]
seg_keys = [k for k in state_dict.keys() if k.startswith('decoders.segmentation.')]
depth_keys = [k for k in state_dict.keys() if k.startswith('decoders.depth.')]
normal_keys = [k for k in state_dict.keys() if k.startswith('decoders.normal.')]

print(f"\nEncoder keys: {len(encoder_keys)}")
print(f"Segmentation keys: {len(seg_keys)}")
print(f"Depth keys: {len(depth_keys)}")
print(f"Normal keys: {len(normal_keys)}")

print("\n=== Segmentation Decoder Structure ===")
seg_structure = {}
for key in seg_keys:  # 모든 키 확인
    parts = key.split('.')
    level = '.'.join(parts[:3])  # decoders.segmentation.X
    if level not in seg_structure:
        seg_structure[level] = []
    seg_structure[level].append(key)

for level, keys in seg_structure.items():
    print(f"{level}: {len(keys)} keys")
    print(f"  Sample: {keys[0] if keys else 'None'}")

print("\n=== All Segmentation Keys ===")
for key in seg_keys[:15]:  # 처음 15개
    print(f"  {key}")