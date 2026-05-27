import torch.nn.functional as F
import torch
import cv2
import numpy as np
import random
from tqdm import tqdm
import os
from CLIP.clip import create_model
from CLIP.adapter import CLIP_Inplanted as model_adapter
import Datasets.visa as visa
import Datasets.mvtec as mvtec
import time
import argparse
from tools.loss import FocalLoss, BinaryDiceLoss
from tools.utils import get_anomaly_map
from torch.optim.lr_scheduler import LambdaLR
import math
from Datasets import DATASET_REGISTRY, DATASET_CLASSES
from torch.utils.data.dataloader import default_collate

def safe_collate(batch):
    new_batch = []
    for item in batch:
        new_item = {}
        for k, v in item.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.bool:
                new_item[k] = v.float()
            else: new_item[k] = v
        new_batch.append(new_item)
    return default_collate(new_batch)

def prepare_data(dataset_name, category, args, **kwargs):
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_REGISTRY: raise ValueError(f"❌ Unsupported dataset: {dataset_name}")
    dataset_cls, split_cls, root_path = DATASET_REGISTRY[dataset_name]
    test_dataset = dataset_cls(source=root_path, split=split_cls.TEST, classname=category, resize=512, imagesize=512)
    print(f"✅ Loaded [{dataset_name}] ({category}) set, size: {len(test_dataset)}")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=safe_collate, **kwargs)
    return test_loader

def train_epoch(optimizer, loss_focal, loss_dice, epoch, anomaly_awareness_loss_list, seg_loss_list, global_anomaly_loss_list, loss_list, clip_model, start_time, train_data, device, model, Dino_model):
    model.train() 
    for idx, image_info in enumerate(train_data):
        # 接收 4 个返回值
        anomaly_awareness, mask, anomaly_map_cross_modal, global_anomaly_score = get_anomaly_map(
            clip_model, image_info, device, model, Dino_model
        )
        
        # 1. Pixel Loss
        anomaly_awareness_loss = loss_focal(anomaly_awareness, mask) + loss_dice(anomaly_awareness[:, 1, :, :], mask)
        
        seg_prob = anomaly_map_cross_modal.unsqueeze(1)
        seg_input = torch.cat([1 - seg_prob, seg_prob], dim=1) 
        seg_loss = loss_focal(seg_input, mask) + loss_dice(anomaly_map_cross_modal, mask)
        
        # 2. Global Loss (BCE)
        labels = image_info["is_anomaly"].to(device).float()
        global_anomaly_loss = F.binary_cross_entropy(global_anomaly_score, labels)

        # Total Loss
        loss = 0.25 * anomaly_awareness_loss + 0.5 * seg_loss + 0.25 * global_anomaly_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{10} | Batch {idx+1}/{len(train_data)} | loss: {loss.item():.4f} | aware: {anomaly_awareness_loss.item():.4f} | seg: {seg_loss.item():.4f} | glob: {global_anomaly_loss.item():.4f}", end="\r", flush=True)
        
        anomaly_awareness_loss_list.append(anomaly_awareness_loss.item())
        seg_loss_list.append(seg_loss.item())
        global_anomaly_loss_list.append(global_anomaly_loss.item())
        loss_list.append(loss.item())
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default="./Result", help="path to result")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--dataset", type=str, default="visa", help="dataset")
    parser.add_argument("--epoch", type=int, default=10, help="epoch")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    # Compat args
    parser.add_argument("--use_caa", action="store_true", default=False)
    parser.add_argument("--use_moga", action="store_true", default=False)
    parser.add_argument("--use_progressive_fusion", action="store_true", default=False)
    parser.add_argument("--use_lgag", action="store_true", default=False)
    args = parser.parse_args()

    if torch.cuda.is_available():
        if args.device.startswith("cuda:"):
            try:
                device_id = int(args.device.split(":")[1])
                if device_id >= torch.cuda.device_count(): device = torch.device("cuda:0")
                else: device = torch.device(args.device)
            except: device = torch.device("cuda:0")
        else: device = torch.device(args.device)
    else: device = torch.device("cpu")
    
    print(f"✅ 使用设备: {device}")
    os.makedirs(f"{args.result_path}", exist_ok=True)
    
    repo_dir = './dinov3'
    Dinov3_model_path = './dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
    print("⏳ Loading DINOv3...")
    Dino_model = torch.hub.load(repo_dir, 'dinov3_vitl16', source='local', weights=Dinov3_model_path)
    Dino_model.to(device)
    Dino_model.eval()

    print("⏳ Loading CLIP...")
    clip_model = create_model(model_name='ViT-L-14-336', img_size=512, device=device, pretrained='openai', require_pretrained=True)
    clip_model.to(device)
    clip_model.eval()

    print("🚀 Initializing Model (VCP + SGA + LRE + FSC Standard)...")
    model = model_adapter(c_in=1024, device=device)
    model.to(device)
    model.train()

    params_to_update = model.parameters()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔥 Trainable Parameters: {num_params / 1e6:.2f} M")

    train_data = prepare_data(args.dataset, 'ALL', args, **{'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {})
    optimizer = torch.optim.AdamW(params_to_update, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    
    total_steps = args.epoch * len(train_data)
    warmup_steps = int(0.1 * total_steps)
    scheduler = LambdaLR(optimizer, lambda s: 0.1 + 0.9 * (1.0 + math.cos(math.pi * (s - warmup_steps) / max(1, total_steps - warmup_steps))) / 2.0 if s >= warmup_steps else float(s) / max(1, warmup_steps))

    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    print("🏁 Start Training...")
    for epoch in range(args.epoch):
        start_time = time.time()
        awareness_loss_list, seg_loss_list, loss_list, global_anomaly_loss_list = [], [], [], []

        train_epoch(optimizer, loss_focal, loss_dice, epoch, awareness_loss_list, seg_loss_list, global_anomaly_loss_list, loss_list, clip_model, start_time, train_data, device, model, Dino_model)
        print() 
        scheduler.step()

        os.makedirs(f"{args.result_path}/ckpt", exist_ok=True)
        torch.save(model.state_dict(), f"{args.result_path}/ckpt/{epoch}.pth")
        
        with open(f"{args.result_path}/loss.txt", "a") as f:
            f.write(f"epoch_{epoch}: aware={np.mean(awareness_loss_list):.4f} seg={np.mean(seg_loss_list):.4f} glob={np.mean(global_anomaly_loss_list):.4f} total={np.mean(loss_list):.4f}\n")