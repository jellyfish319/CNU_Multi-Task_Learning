# Run

## 1. Download Dataset

NYU: you can download dataset in [here](https://www.dropbox.com/scl/fo/p7n54hqfpfyc6fe6n62qk/AKVb28ZmgDiGdRMNkX5WJvo?rlkey=hcf31bdrezqjih36oi8usjait&e=1&dl=0)

## 2. Train

You should download pretrained model for training at [here(PVTv2)](https://drive.google.com/drive/folders/1hCaKNrlMF6ut0b36SedPRNC_434R8VVa) and [here(Swin Transformer)](https://drive.google.com/drive/folders/1MCiyAnMI14gyfCrKC4yme0UANNOdtvuH)

Run main_pvtv2.py in nyu

```
python main_pvtv2.py --weighting EW --arch HPS --dataset_path /path/to/data --gpu_id 0 --scheduler step --seed 2025 --save_path path/to/save
```

## 3. Visualize

```
python visualize.py --model_path path/to/model.pt --dataset_path /path/to/data --mode single --sample_idx 0
```