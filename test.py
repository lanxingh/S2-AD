from utils import encode_text_with_prompt_ensemble
import torch
import cv2
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.ndimage import label
import os
import glob
import sys
import time
from CLIP.clip import create_model
import torch.nn.functional as F
from CLIP.adapter import CLIP_Inplanted as model_adapter
import argparse
from tools.utils import get_anomaly_map
from tools.visualization import visualization
from Datasets import DATASET_REGISTRY, DATASET_CLASSES
from torch.utils.data.dataloader import default_collate

use_cuda = torch.cuda.is_available()
kwargs = {'num_workers': 0, 'pin_memory': False} if use_cuda else {}

def safe_collate(batch):
    new_batch =[]
    for item in batch:
        new_item = {}
        for k, v in item.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.bool:
                new_item[k] = v.float()
            else:
                new_item[k] = v
        new_batch.append(new_item)
    return default_collate(new_batch)

def prepare_data(dataset_name, category, args, **kwargs):
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_REGISTRY: raise ValueError(f"❌ Unsupported dataset: {dataset_name}")
    dataset_cls, split_cls, root_path = DATASET_REGISTRY[dataset_name]
    test_dataset = dataset_cls(source=root_path, split=split_cls.TEST, classname=category, resize=512, imagesize=512)
    print(f"✅ Loaded[{dataset_name}] ({category}) test set, size: {len(test_dataset)}")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=safe_collate, **kwargs)
    return test_loader

# =========================================================================
# PRO-Score 计算函数 (修复维度 Bug 且优化边界版)
# =========================================================================
def calculate_pro_score(masks, amaps, num_th=200):
    """
    计算 PRO-Score (积分至 FPR=0.3)
    masks: (N, H, W) 真实的二进制掩码
    amaps: (N, H, W) 预测的异常热力图
    """
    min_th, max_th = amaps.min(), amaps.max()
    thresholds = np.linspace(min_th, max_th, num_th)
    
    gt_comps =[]
    for mask in masks:
        labeled, n_comps = label(mask)
        comps = [labeled == i for i in range(1, n_comps + 1)]
        gt_comps.append(comps)
        
    pros, fprs = [],[]
    
    # 保持 3D 矩阵形态进行逻辑运算，避免 Broadcast Error
    bg_mask = (masks == 0)
    tn = bg_mask.sum()
    
    for th in thresholds:
        binary_amaps = (amaps >= th)
        
        # 逻辑与运算直接在 3D 空间进行
        fp = np.logical_and(binary_amaps, bg_mask).sum()
        fpr = fp / tn if tn > 0 else 0
        fprs.append(fpr)
        
        pro_per_img =[]
        for b_amap, comps in zip(binary_amaps, gt_comps):
            for comp in comps:
                overlap = np.logical_and(b_amap, comp).sum()
                comp_size = comp.sum()
                pro_per_img.append(overlap / comp_size if comp_size > 0 else 0)
        pros.append(np.mean(pro_per_img) if len(pro_per_img) > 0 else 0)
        
    fprs = np.array(fprs)
    pros = np.array(pros)
    
    # 排序以备计算 AUC
    sort_idx = np.argsort(fprs)
    fprs, pros = fprs[sort_idx], pros[sort_idx]
    
    # 积分至 FPR=0.3
    max_fpr = 0.3
    valid_idx = fprs <= max_fpr
    if not np.any(valid_idx): return 0.0
    
    fprs_sel = fprs[valid_idx]
    pros_sel = pros[valid_idx]
    
    # 线性插值补齐 0.3 边界，让分数更精确
    idx_greater = np.argmax(fprs > max_fpr)
    if idx_greater > 0 and fprs[idx_greater] > max_fpr:
        x1, y1 = fprs[idx_greater-1], pros[idx_greater-1]
        x2, y2 = fprs[idx_greater], pros[idx_greater]
        if x2 > x1:
            y_interp = y1 + (y2 - y1) * (max_fpr - x1) / (x2 - x1)
            fprs_sel = np.append(fprs_sel, max_fpr)
            pros_sel = np.append(pros_sel, y_interp)
        else:
            fprs_sel = np.append(fprs_sel, max_fpr)
            pros_sel = np.append(pros_sel, pros_sel[-1])
    else:
        fprs_sel = np.append(fprs_sel, max_fpr)
        pros_sel = np.append(pros_sel, pros_sel[-1])
    
    pro_auc = auc(fprs_sel, pros_sel) / max_fpr
    return pro_auc

# =========================================================================

def test(clip_model, result_path, epoch, args, device, model, Dino_model):
    P_AUROC, P_F1, P_PRO = [], [],[]
    total_infer_time = 0.0
    total_images = 0

    print(f"--------------------------------------Testing epoch {epoch}--------------------------------------")
    if args.dataset in DATASET_CLASSES: categories = sorted(DATASET_CLASSES[args.dataset])
    else: categories = [args.dataset]

    with torch.no_grad():
        for category in categories:
            os.makedirs(result_path + f"/{category}", exist_ok=True)
            pixel_pred, pixel_gt, img_list =[], [],[]
            
            try: test_data = prepare_data(args.dataset, category, args, **kwargs)
            except Exception as e:
                print(f"Skipping {category}: {e}")
                continue

            # 预热 GPU (防止第一次测速不准)
            dummy_image = next(iter(test_data))
            _ = get_anomaly_map(clip_model, dummy_image, device, model, Dino_model)
            
            for image_info in tqdm(test_data, desc=f"Testing {category}"):
                batch_size_curr = image_info["image"].shape[0]
                
                # --- CUDA 精准测速开始 ---
                if use_cuda: torch.cuda.synchronize()
                t_start = time.time()
                
                # 接收 4 个返回值
                _, mask, anomaly_map_cross_modal, _ = get_anomaly_map(
                    clip_model, image_info, device, model, Dino_model
                )
                
                if use_cuda: torch.cuda.synchronize()
                t_end = time.time()
                # --- CUDA 精准测速结束 ---
                
                total_infer_time += (t_end - t_start)
                total_images += batch_size_curr
                
                if mask.dim() == 4: mask = mask.squeeze(1)
                
                # 直接强转 int，防止 sklearn 与 PRO 计算报错
                pixel_gt.append((mask.cpu().detach().numpy() > 0.5).astype(int)) 
                img_list.extend(image_info["image_path"])
                pixel_pred.append(anomaly_map_cross_modal.cpu().detach().numpy())
            
            if use_cuda: torch.cuda.empty_cache()

            # 拼接
            gt_masks_np = np.concatenate(pixel_gt, axis=0)
            pred_masks_np = np.concatenate(pixel_pred, axis=0)
            
            # 可视化前归一化
            if pred_masks_np.max() - pred_masks_np.min() > 0:
                pred_masks_vis = (pred_masks_np - pred_masks_np.min()) / (pred_masks_np.max() - pred_masks_np.min())
            else:
                pred_masks_vis = pred_masks_np
                
            visualization(img_list, pred_masks_vis, gt_masks_np, category, result_path)

            # --- 计算 3 个指标 ---
            # 1. AUROC
            try: auc_val = roc_auc_score(gt_masks_np.flatten(), pred_masks_np.flatten())
            except ValueError: auc_val = 0.5 
            P_AUROC.append(auc_val)

            # 2. F1
            pre, rec, _ = precision_recall_curve(gt_masks_np.flatten(), pred_masks_np.flatten())
            f1_scores = (2 * pre * rec) / (pre + rec + 1e-8)
            f1_val = np.max(f1_scores[np.isfinite(f1_scores)]) if len(f1_scores) > 0 else 0
            P_F1.append(f1_val)
            
            # 3. PRO-Score
            try:
                pro_val = calculate_pro_score(gt_masks_np, pred_masks_np)
            except Exception as e:
                print(f"\n[WARNING] PRO Calculation Error for {category}: {e}")
                pro_val = 0.0
            P_PRO.append(pro_val)

            print(f"{category}: AUC={auc_val:.4f} F1={f1_val:.4f} PRO={pro_val:.4f}")
    
    # 计算平均分
    mean_auc = np.mean(P_AUROC) if len(P_AUROC) > 0 else 0
    mean_f1 = np.mean(P_F1) if len(P_F1) > 0 else 0
    mean_pro = np.mean(P_PRO) if len(P_PRO) > 0 else 0
    
    # 计算推理耗时 (ms/image)
    avg_time_ms = (total_infer_time / total_images) * 1000 if total_images > 0 else 0

    print(f"\n[Final Results] Epoch {epoch}")
    print(f"Mean Pixel-AUC: {mean_auc:.4f}")
    print(f"Mean Pixel-F1:  {mean_f1:.4f}")
    print(f"Mean PRO-Score: {mean_pro:.4f}")
    print(f"Avg Inference Time: {avg_time_ms:.2f} ms/image")

    metric_file = f"{result_path}/metric_pro.txt"
    with open(metric_file, "a") as f:
        f.write(f"----------Dataset: {args.dataset}----------\n")
        f.write(f"{'Classname':<14s}{'P-AUC':<10s}{'P-F1':<10s}{'PRO':<10s} (epoch_{epoch})\n")
        for i, cname in enumerate(categories):
            if i < len(P_AUROC): 
                f.write(f"{cname:<14s}{P_AUROC[i]:.4f}    {P_F1[i]:.4f}    {P_PRO[i]:.4f}\n")
        f.write(f"{'mean':<14s}{mean_auc:.4f}    {mean_f1:.4f}    {mean_pro:.4f}\n")
        f.write(f"Inference Speed: {avg_time_ms:.2f} ms per image\n\n")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default="./Result", help="path to result")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--dataset", type=str, default="mvtec", help="dataset")
    parser.add_argument("--epoch", type=int, default=9, help="epoch")
    parser.add_argument("--checkpoint", type=str, default=None, help="ckpt")
    
    # 占位参数
    parser.add_argument("--use_caa", action="store_true", default=False)
    parser.add_argument("--use_moga", action="store_true", default=False)
    parser.add_argument("--use_progressive_fusion", action="store_true", default=False)
    parser.add_argument("--use_lgag", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(f"{args.result_path}/{args.dataset}", exist_ok=True)
    
    Dino_model = torch.hub.load('./dinov3', 'dinov3_vitl16', source='local', weights='./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth').to(device).eval()
    clip_model = create_model(model_name='ViT-L-14-336', img_size=512, device=device, pretrained='openai', require_pretrained=True).to(device).eval()
    
    model = model_adapter(c_in=1024, device=device)
    ckpt = args.checkpoint if args.checkpoint else f"{args.result_path}/ckpt/{args.epoch}.pth"
    
    if os.path.exists(ckpt):
        print(f"\n[INFO] 加载模型: {ckpt}")
        # 如果是不同版本训练的，允许 strict=False 跳过不匹配的层
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
        model.to(device).eval()
        test(clip_model, args.result_path, args.epoch, args, device, model, Dino_model)
    else:
        print(f"[ERROR] 找不到权重文件: {ckpt}")