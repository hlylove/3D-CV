"""
ShapeNetPart Dataset Preprocessing Script

Convert raw ShapeNet PartAnnotation format to Pointcept expected format.

Raw format:
- points/*.pts: x y z (coordinates only)
- expert_verified/points_label/*.seg: label per line

Target format (shapenetcore_partanno_segmentation_benchmark_v0_normal):
- {category}/{id}.txt: x y z nx ny nz label (7 columns)
- synsetoffset2category.txt
- train_test_split/shuffled_{train,val,test}_file_list.json

Author: Auto-generated for Pointcept
"""

import os
import json
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import random


# Category mapping: synset ID -> category name
SYNSET_TO_CATEGORY = {
    "02691156": "Airplane",
    "02773838": "Bag",
    "02954340": "Cap",
    "02958343": "Car",
    "03001627": "Chair",
    "03261776": "Earphone",
    "03467517": "Guitar",
    "03624134": "Knife",
    "03636649": "Lamp",
    "03642806": "Laptop",
    "03790512": "Motorbike",
    "03797390": "Mug",
    "03948459": "Pistol",
    "04099429": "Rocket",
    "04225987": "Skateboard",
    "04379243": "Table",
}

# Part label offset for each category (to make labels globally unique)
CATEGORY_PART_OFFSET = {
    "Airplane": 0,      # parts 0-3
    "Bag": 4,           # parts 4-5
    "Cap": 6,           # parts 6-7
    "Car": 8,           # parts 8-11
    "Chair": 12,        # parts 12-15
    "Earphone": 16,     # parts 16-18
    "Guitar": 19,       # parts 19-21
    "Knife": 22,        # parts 22-23
    "Lamp": 24,         # parts 24-27
    "Laptop": 28,       # parts 28-29
    "Motorbike": 30,    # parts 30-35
    "Mug": 36,          # parts 36-37
    "Pistol": 38,       # parts 38-40
    "Rocket": 41,       # parts 41-43
    "Skateboard": 44,   # parts 44-46
    "Table": 47,        # parts 47-49
}


def estimate_normals_open3d(points, k=30):
    """Estimate normals using Open3D (recommended, better quality)."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    pcd.orient_normals_consistent_tangent_plane(k=15)
    return np.asarray(pcd.normals).astype(np.float32)


def process_category(raw_dir, output_dir, synset_id, estimate_normals=False):
    """Process all samples in a category."""
    category_name = SYNSET_TO_CATEGORY[synset_id]
    part_offset = CATEGORY_PART_OFFSET[category_name]
    
    raw_category_dir = Path(raw_dir) / synset_id
    output_category_dir = Path(output_dir) / synset_id
    output_category_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all point files
    points_dir = raw_category_dir / "points"
    labels_dir = raw_category_dir / "expert_verified" / "points_label"
    
    if not points_dir.exists():
        print(f"Warning: {points_dir} does not exist, skipping")
        return []
    
    processed_files = []
    pts_files = list(points_dir.glob("*.pts"))
    
    for pts_file in tqdm(pts_files, desc=f"Processing {category_name}"):
        sample_id = pts_file.stem
        seg_file = labels_dir / f"{sample_id}.seg"
        
        if not seg_file.exists():
            continue
        
        # Load points (x, y, z)
        try:
            points = np.loadtxt(pts_file).astype(np.float32)
            if points.ndim == 1:
                points = points.reshape(1, -1)
        except Exception as e:
            print(f"Error loading {pts_file}: {e}")
            continue
        
        # Load labels
        try:
            labels = np.loadtxt(seg_file).astype(np.int32)
            if labels.ndim == 0:
                labels = np.array([labels])
        except Exception as e:
            print(f"Error loading {seg_file}: {e}")
            continue
        
        # Verify dimensions match
        if len(points) != len(labels):
            print(f"Warning: Points ({len(points)}) and labels ({len(labels)}) mismatch for {sample_id}")
            continue
        
        # Estimate or use zero normals
        if estimate_normals:
            normals = estimate_normals_open3d(points)
        else:
            normals = np.zeros_like(points)
        
        # Apply part offset to make labels globally unique
        # Raw labels are 1-indexed, convert to 0-indexed
        global_labels = labels - 1 + part_offset
        
        # Combine: x y z nx ny nz label
        data = np.column_stack([points, normals, global_labels])
        
        # Save
        output_file = output_category_dir / f"{sample_id}.txt"
        np.savetxt(output_file, data, fmt="%.6f %.6f %.6f %.6f %.6f %.6f %d")
        
        processed_files.append(f"shape_data/{synset_id}/{sample_id}")
    
    return processed_files


def create_splits(all_files, output_dir, train_ratio=0.8, val_ratio=0.1):
    """Create train/val/test splits."""
    random.seed(42)
    random.shuffle(all_files)
    
    n_total = len(all_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train + n_val]
    test_files = all_files[n_train + n_val:]
    
    split_dir = Path(output_dir) / "train_test_split"
    split_dir.mkdir(parents=True, exist_ok=True)
    
    with open(split_dir / "shuffled_train_file_list.json", "w") as f:
        json.dump(train_files, f, indent=2)
    
    with open(split_dir / "shuffled_val_file_list.json", "w") as f:
        json.dump(val_files, f, indent=2)
    
    with open(split_dir / "shuffled_test_file_list.json", "w") as f:
        json.dump(test_files, f, indent=2)
    
    print(f"Split: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")


def create_category_file(output_dir):
    """Create synsetoffset2category.txt file."""
    output_file = Path(output_dir) / "synsetoffset2category.txt"
    with open(output_file, "w") as f:
        for synset_id, category_name in SYNSET_TO_CATEGORY.items():
            f.write(f"{category_name}\t{synset_id}\n")
    print(f"Created {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess ShapeNetPart dataset")
    parser.add_argument("--raw_dir", type=str, required=True,
                        help="Path to raw PartAnnotation directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to output directory")
    parser.add_argument("--estimate_normals", action="store_true",
                        help="Estimate normals using Open3D (slower but may improve results)")
    args = parser.parse_args()
    
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.estimate_normals:
        try:
            import open3d
            print("✓ Open3D found, will estimate normals")
        except ImportError:
            print("Error: --estimate_normals requires Open3D. Install with: pip install open3d")
            return
    
    # Process each category
    all_files = []
    for synset_id in SYNSET_TO_CATEGORY.keys():
        if (raw_dir / synset_id).exists():
            files = process_category(raw_dir, output_dir, synset_id, args.estimate_normals)
            all_files.extend(files)
    
    # Create splits
    create_splits(all_files, output_dir)
    
    # Create category file
    create_category_file(output_dir)
    
    print(f"\nDone! Processed {len(all_files)} samples")
    print(f"Output directory: {output_dir}")
    
    if args.estimate_normals:
        print("\nTo train with normals:")
        print("  sh scripts/train.sh -g 4 -d shapenetpart -c partseg-pt-v3m1-1-normal -n shapenetpart-ptv3")
    else:
        print("\nTo train without normals:")
        print("  sh scripts/train.sh -g 4 -d shapenetpart -c partseg-pt-v3m1-0-base -n shapenetpart-ptv3")


if __name__ == "__main__":
    main()
