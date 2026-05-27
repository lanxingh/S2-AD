import torch
import os

from Datasets import (
    mvtec, visa, btad, mpdd, 
    tn3k, clinicdb, colondb, isic
)

# (Dataset, Split, root_path) 
# 支持通过环境变量 DATASET_ROOT 设置数据集根目录
# 如果没有设置，优先使用项目根目录下的 Data 文件夹
root_dir = os.environ.get("DATASET_ROOT", None)

if root_dir is None:
    # 获取当前文件所在目录（Datasets文件夹）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录（Datasets的父目录）
    project_root = os.path.dirname(current_dir)
    # 优先使用项目根目录下的 Data 文件夹
    project_data_dir = os.path.join(project_root, "Data")
    
    # 尝试路径列表（优先级从高到低）
    possible_paths = [
        project_data_dir,  # 项目根目录下的Data文件夹（最高优先级）
        "./Data",          # 当前目录下的Data
        "./data",          # 当前目录下的data
        "D:/Data",         # Windows常见路径
        "E:/Data", 
        "C:/Data",
        "/Data",           # Linux路径
    ]
    
    # 查找第一个存在的路径
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            root_dir = abs_path
            print(f"[INFO] 自动检测到数据集路径: {root_dir}")
            break
    else:
        # 如果都没找到，使用项目根目录下的Data（即使不存在也使用，让用户知道应该放在哪里）
        root_dir = project_data_dir
        print(f"[WARNING] 数据集根目录不存在，使用默认路径: {root_dir}")
        print(f"   请确保数据集放在: {root_dir}")
        print(f"   或设置环境变量 DATASET_ROOT 指定数据集路径")

DATASET_REGISTRY = {
    # 注意：实际文件夹名称是 Industrial_Datasets (复数)
    "mvtec":       (mvtec.Dataset, mvtec.DatasetSplit, os.path.join(root_dir, "Industrial_Datasets", "MVTecAD")),
    "visa":        (visa.Dataset, visa.DatasetSplit, os.path.join(root_dir, "Industrial_Datasets", "VisA_20220922")),
    "btad":        (btad.Dataset, btad.DatasetSplit, os.path.join(root_dir, "Industrial_Datasets", "BTAD", "BTech_Dataset_transformed")),
    "mpdd":        (mpdd.Dataset, mpdd.DatasetSplit, os.path.join(root_dir, "Industrial_Datasets", "MPDD")),
    "tn3k":        (tn3k.Dataset, tn3k.DatasetSplit, os.path.join(root_dir, "Medical_Datasets", "TN3K")),
    "clinicdb":    (clinicdb.Dataset, clinicdb.DatasetSplit, os.path.join(root_dir, "Medical_Datasets", "CVC-ClinicDB")),
    "colondb":     (colondb.Dataset, colondb.DatasetSplit, os.path.join(root_dir, "Medical_Datasets", "CVC-ColonDB")),
    "isic":        (isic.Dataset, isic.DatasetSplit, os.path.join(root_dir, "Medical_Datasets", "ISIC")),
}

DATASET_CLASSES = {
    "mvtec":    {'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'},
    "visa":     {"candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pcb1","pcb2", "pcb3", "pcb4", "pipe_fryum"},
    "btad":     {'01', '02', '03'},
    "mpdd":     {'bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes'},
    "tn3k":     {'01'},
    "clinicdb": {'01'},
    "colondb":  {'01'},
    "isic":     {'01'},
}