import torch
import numpy as np
import argparse
import os
import json
import scipy.spatial
from tqdm import tqdm
from pointcept.models import build_model
from pointcept.utils.config import Config
from pointcept.datasets.transform import Compose
import open3d as o3d

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to your config.py")
    parser.add_argument("--checkpoint", required=True, help="Path to model_best.pth")
    parser.add_argument("--out_dir", required=True, help="Folder to save the .txt results")
    parser.add_argument("--device", default="cuda", help="Device to use")
    args = parser.parse_args()

    # 1. Setup
    cfg = Config.fromfile(args.config)
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 2. Build Model
    print(f"Loading model from {args.checkpoint}...")
    model = build_model(cfg.model).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict)
    model.eval()

    # 3. Define Transform Pipeline
    test_transform_cfg = [
        dict(type="CenterShift", apply_z=True),
        dict(type="NormalizeCoord"),
        dict(
            type="GridSample",
            grid_size=0.01,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
        ),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "segment"),
            feat_keys=["coord", "normal"], 
        ),
    ]
    transform_pipeline = Compose(test_transform_cfg)

    # 4. Load Test File List
    split_file = os.path.join(cfg.data_root, "train_test_split", "shuffled_test_file_list.json")
    
    if not os.path.exists(split_file):
        print(f"Error: Could not find split file at {split_file}")
        return

    with open(split_file, "r") as f:
        test_files = json.load(f)

    print(f"Found {len(test_files)} files in the test set.")
    print(f"Saving results to: {args.out_dir}")

    # 5. Inference Loop
    for file_rel_path in tqdm(test_files):
        parts = file_rel_path.split("/")
        synset_id = parts[-2]
        sample_id = parts[-1]
        
        file_path = os.path.join(cfg.data_root, synset_id, f"{sample_id}.txt")
        
        if not os.path.exists(file_path):
            file_path = os.path.join(cfg.data_root, synset_id, sample_id)
            if not os.path.exists(file_path):
                print(f"Skipping missing file: {file_path}")
                continue

        # Load Data
        try:
            data = np.loadtxt(file_path).astype(np.float32)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        raw_coord = data[:, :3]
        raw_normals = data[:, 3:6]
        
        input_dict = {
            "coord": raw_coord,
            "normal": raw_normals,
            "segment": np.full((len(raw_coord),), -1).astype(np.int32)
        }
        
        input_dict = transform_pipeline(input_dict)
        
        # Add offset dim (Required for PTv3)
        # Offset is a 1D tensor [N] indicating the size of the point cloud
        input_dict["offset"] = torch.tensor([input_dict["coord"].shape[0]], dtype=torch.int)
        
        # --- FIXED GPU MOVE LOGIC ---
        # We simply move tensors to GPU. We DO NOT add batch dimensions (unsqueeze).
        # Sparse models expect [N, C] inputs, not [1, N, C].
        for key in input_dict:
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].to(args.device)
        # ----------------------------

        with torch.no_grad():
            retval = model(input_dict)
            seg_logits = retval["seg_logits"]
            pred_labels_voxel = torch.argmax(seg_logits, dim=1).cpu().numpy()

        # Interpolation
        voxel_coords = input_dict["coord"].cpu().numpy()
        tree = scipy.spatial.cKDTree(voxel_coords)
        _, indices = tree.query(raw_coord, k=1)
        final_preds = pred_labels_voxel[indices]

        # Save
        save_folder = os.path.join(args.out_dir, synset_id)
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, f"{sample_id}.txt")
        
        np.savetxt(save_path, final_preds, fmt="%d")

    print("Done.")

if __name__ == "__main__":
    main()