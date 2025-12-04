_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 128  # bs: total bs in all gpus, def 32
num_worker = 8
batch_size_val = 8
empty_cache = False
enable_amp = True
clip_grad = 2.0

# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=50,  # ShapeNetPart has 50 part classes
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=6,  # coord(3) + normal(3)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(256, 256, 256, 256, 256),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(256, 256, 256, 256),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings
epoch = 300
# ref rl 0.001
optimizer = dict(type="AdamW", lr=0.0014, weight_decay=0.01)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.001, 0.0014],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0014)]

# dataset settings
dataset_type = "ShapeNetPartDataset"
data_root = "data/shapenetcore_partanno_segmentation_benchmark_v0_normal"

data = dict(
    num_classes=50,
    ignore_index=-1,
    names=[
        # Airplane (4 parts)
        "Airplane_wing",
        "Airplane_body",
        "Airplane_tail",
        "Airplane_engine",
        
        # Bag (2 parts)
        "Bag_handle",
        "Bag_body",
        
        # Cap (2 parts)
        "Cap_panels",
        "Cap_peak",
        
        # Car (4 parts)
        "Car_body",
        "Car_wheel",
        "Car_hood",
        "Car_roof",
        
        # Chair (4 parts)
        "Chair_leg",
        "Chair_arm",
        "Chair_back",
        "Chair_seat",
        
        # Earphone (3 parts)
        "Earphone_earphone",
        "Earphone_headband",
        "Earphone_cord",
        
        # Guitar (3 parts)
        "Guitar_head",
        "Guitar_body",
        "Guitar_neck",
        
        # Knife (2 parts)
        "Knife_handle",
        "Knife_blade",
        
        # Lamp (4 parts)
        "Lamp_canopy",
        "Lamp_lampshade",
        "Lamp_base",
        "Lamp_bulb", 
        
        # Laptop (2 parts) 
        "Laptop_keyboard",
        "Laptop_screen",
        
        # Motorbike (6 parts)
        "Motorbike_wheel",
        "Motorbike_handle",
        "Motorbike_gas_tank",
        "Motorbike_light",
        "Motorbike_seat",
        "Motorbike_frame",
        
        # Mug (2 parts) 
        "Mug_handle",
        "Mug_body",
        
        # Pistol (3 parts)
        "Pistol_trigger_and_guard",
        "Pistol_handle",
        "Pistol_barrel",
        
        # Rocket (3 parts)
        "Rocket_nose",
        "Rocket_body",
        "Rocket_fin",
        
        # Skateboard (3 parts)
        "Skateboard_wheel",
        "Skateboard_deck",
        "Skateboard_axle",
        
        # Table (3 parts)
        "Table_leg",
        "Table_top",
        "Table_connector"
    ],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1/24, 1/24], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1/24, 1/24], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.8, 1.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="NormalizeCoord"),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=["coord", "norm"],  # use normals
            ),
        ],
        test_mode=False,
        loop=2,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
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
                feat_keys=["coord", "norm"],  # use normals
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeCoord"),
        ],
        test_mode=True,
        test_cfg=dict(
            post_transform=[
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
                    keys=("coord", "grid_coord"),
                    feat_keys=["coord", "norm"],  # use normals
                ),
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)],
            ],
        ),
    ),
)

# hooks
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
]

