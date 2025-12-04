# Point Transformer V3 on ShapeNetPart

Training PTv3 for 3D part segmentation on ShapeNetPart dataset based on [Pointcept](https://github.com/Pointcept/Pointcept).

## Changes Made

- `configs/shapenetpart/partseg-pt-v3m1-0-base.py` - Config without normals
- `configs/shapenetpart/partseg-pt-v3m1-1-normal.py` - Config with normals (recommended)
- `pointcept/datasets/shapenet_part.py` - Bug fixes
- `pointcept/datasets/preprocessing/shapenetpart/preprocess_shapenetpart.py` - Preprocessing script

## Quick Start

### 1. Install Environment

```bash
conda create -n pointcept python=3.10 -y
conda activate pointcept
conda install pytorch==2.1.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch-geometric spconv-cu118 open3d

cd libs/pointops && python setup.py install && cd ../..
```

### 2. Download Dataset

Download ShapeNetPart from [Kaggle](https://www.kaggle.com/datasets/majdouline20/shapenetpart-dataset) and place in `Pointcept/PartAnnotation/`.

### 3. Preprocess Data

```bash
# With normals (recommended)
python pointcept/datasets/preprocessing/shapenetpart/preprocess_shapenetpart.py \
    --raw_dir PartAnnotation \
    --output_dir data/shapenetcore_partanno_segmentation_benchmark_v0_normal \
    --estimate_normals

# Without normals (faster)
python pointcept/datasets/preprocessing/shapenetpart/preprocess_shapenetpart.py \
    --raw_dir PartAnnotation \
    --output_dir data/shapenetcore_partanno_segmentation_benchmark_v0_normal
```

### 4. Train

```bash
# With normals
sh scripts/train.sh -g 4 -d shapenetpart -c partseg-pt-v3m1-1-normal -n shapenetpart-ptv3

# Without normals
sh scripts/train.sh -g 4 -d shapenetpart -c partseg-pt-v3m1-0-base -n shapenetpart-ptv3-base
```

### 5. Test

```bash
sh scripts/test.sh -g 4 -d shapenetpart -n shapenetpart-ptv3 -w model_best
```

## Dataset

ShapeNetPart: 16 object categories, 50 part classes.

## Reference

- [Point Transformer V3](https://github.com/Pointcept/PointTransformerV3)
- [Pointcept](https://github.com/Pointcept/Pointcept)
