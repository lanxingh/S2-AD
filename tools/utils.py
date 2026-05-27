import torch
import torch.nn.functional as F
import os
from utils import encode_text_with_prompt_ensemble 

def get_feature_dinov3(image_path, batch_img, device, Dino_model):
    with torch.no_grad():
        layers = [5, 11, 17, 23]
        patch_tokens_dict = {i: [] for i in layers}
        cls_tokens_dict = {i: [] for i in layers}
        for j in range(len(image_path)):
            patch_dict, tokens_dict, cls_dict = {}, {}, {}
            handles = []
            image = batch_img[j].unsqueeze(0).to(device)
            anchor = getattr(Dino_model, "norm", None) or getattr(Dino_model, "fc_norm", None)
            for i in layers:
                def _mk_hook(idx):
                    def _hook(module, inp, out):
                        tokens_dict[idx] = anchor(out[0]).detach().cpu()
                    return _hook
                handles.append(Dino_model.blocks[i].register_forward_hook(_mk_hook(i)))
            with torch.inference_mode(): _ = Dino_model(image)
            for h in handles: h.remove()
            for i, toks in tokens_dict.items():
                tokens = toks[:, 5:, :]
                tokens = (tokens - tokens.mean(dim=1, keepdim=True)) / (tokens.std(dim=1, keepdim=True) + 1e-6)
                patch_dict[i] = tokens
                cls_dict[i] = toks[:, 0, :].unsqueeze(1)
            for i in layers:
                patch_tokens_dict[i].append(patch_dict[i])
                cls_tokens_dict[i].append(cls_dict[i])
        patch_tokens = [torch.cat(patch_tokens_dict[i], dim=0).to(device) for i in layers]
        cls_token = [torch.cat(cls_tokens_dict[i], dim=0).to(device) for i in layers]
        return cls_token, patch_tokens

def get_anomaly_map(clip_model, image_info, device, model, Dino_model, **kwargs):
    image = image_info["image"].to(device)
    image_path = image_info["image_path"]
    mask = image_info["mask"].to(device)
    mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
    y = image_info["is_anomaly"]

    text_feature_raw = torch.zeros(len(image_path), 768, 2).to(device)
    with torch.no_grad():
        for i in range(len(image_path)):
            normalized_path = os.path.normpath(image_path[i])
            path_parts = normalized_path.split(os.sep)
            if len(path_parts) >= 4: classname = path_parts[-4]
            elif len(path_parts) >= 2: classname = path_parts[-2]
            else: classname = os.path.splitext(os.path.basename(image_path[i]))[0]
            text_feature_raw[i] = encode_text_with_prompt_ensemble(clip_model, classname, device, '', y)

    text_static = model.forward_text(text_feature_raw)
    cls_token_raw, patch_tokens_raw = get_feature_dinov3(image_path, image, device, Dino_model)
    
    # 接收 FSC 校准后的 Map (Prob)
    calib_maps_list, calib_scores_list = model.forward_visual(
        patch_tokens_raw, cls_token_raw, text_static
    )

    anomaly_maps = []
    for i in range(4):
        sim_map = calib_maps_list[i] # (B, N)
        B, N = sim_map.shape
        H = int(N**0.5)
        
        sim_map = sim_map.view(B, 1, H, H)
        sim_map = F.interpolate(sim_map, size=512, mode='bilinear', align_corners=True)
        # 已经是 Prob
        anomaly_maps.append(sim_map[:, 0, :, :])

    final_anomaly_map = torch.mean(torch.stack(anomaly_maps, dim=0), dim=0) # (B, H, W)
    final_score = torch.mean(torch.stack(calib_scores_list, dim=0), dim=0) # (B,)
    
    am_prob = final_anomaly_map.unsqueeze(1)
    anomaly_awareness = torch.cat([1 - am_prob, am_prob], dim=1)

    return anomaly_awareness, mask, final_anomaly_map, final_score