import torch
from torch import nn
from torch.nn import functional as F
import math

# =========================================================================
# [模块 0] VCP: Visual-Conditioned Prompting
# 作用: 利用全局视觉上下文动态调整文本特征
# =========================================================================
class VisualConditionedPrompting(nn.Module):
    def __init__(self, visual_dim=1024, text_dim=768, hidden_dim=256):
        super().__init__()
        self.meta_net = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, text_dim)
        )
        # 零初始化
        nn.init.constant_(self.meta_net[-1].weight, 0)
        nn.init.constant_(self.meta_net[-1].bias, 0)

    def forward(self, text_feat, visual_context):
        if visual_context.dim() == 3:
            visual_context = visual_context.squeeze(1)
        bias = self.meta_net(visual_context)
        bias = bias.unsqueeze(2)
        return text_feat + bias

# =========================================================================
# [模块 1] SGA: Semantic Guidance Attention
# 作用: 语义对齐
# =========================================================================
class LanguageGuidedAttention(nn.Module):
    def __init__(self, visual_dim=1024, text_dim=768, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = text_dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q_proj = nn.Linear(visual_dim, text_dim)
        self.k_proj = nn.Linear(text_dim, text_dim)
        self.v_proj = nn.Linear(text_dim, text_dim)
        self.out_proj = nn.Linear(text_dim, visual_dim)
        
        self.gate = nn.Parameter(torch.zeros(1)) 
        self.layer_norm = nn.LayerNorm(visual_dim)

    def forward(self, x_visual, x_text):
        B, N, C = x_visual.shape
        residual = x_visual

        if x_text.dim() == 2:
            x_text = x_text.unsqueeze(1)
            
        q = self.q_proj(x_visual).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k_proj(x_text).reshape(B, 1, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v_proj(x_text).reshape(B, 1, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1) 
        
        out = attn @ v 
        out = out.transpose(1, 2).reshape(B, N, -1)
        out = self.out_proj(out)

        return self.layer_norm(residual + self.gate * out)

# =========================================================================
# [模块 2] LRE: Local Refinement Encoder
# 作用: 空间平滑
# =========================================================================
class LocalPerceptionFFN(nn.Module):
    def __init__(self, dim=1024, expansion_ratio=4):
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )
        self.layer_norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.out_channels == 1024 and m.kernel_size == (1, 1):
                nn.init.constant_(m.weight, 0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, N, C = x.shape
        H = int(math.sqrt(N)) 
        residual = x
        x_img = x.permute(0, 2, 1).view(B, C, H, H)
        out = self.net(x_img)
        out = out.flatten(2).permute(0, 2, 1)
        return self.layer_norm(residual + out)

# =========================================================================
# [核心模块] TG-AMS-FSC: Text-Guided Adaptive Multi-Scale FSC
# 策略：利用文本语义引导 + 受限自适应权重预测
# =========================================================================
class TextGuidedAdaptiveMultiScaleFSC(nn.Module):
    def __init__(self, feature_dim=768, high_res=32, low_res=16):
        super().__init__()
        self.high_res = high_res
        self.low_res = low_res
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        self.alpha = nn.Parameter(torch.tensor(0.2))
        
        self.proj = nn.Linear(feature_dim, feature_dim // 4)
        
        # 文本投影层 (768 -> 128)
        self.text_proj = nn.Linear(feature_dim, 128)
        
        # 权重预测器 (Meta-Net)
        # Input: Visual(768) + Text(128) = 896
        self.class_aware_weight = nn.Sequential(
            nn.Linear(feature_dim + 128, 128),
            nn.LayerNorm(128), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),   
            nn.Linear(128, 1)
        )
        
        # 初始化为 0
        nn.init.constant_(self.class_aware_weight[-1].weight, 0)
        nn.init.constant_(self.class_aware_weight[-1].bias, 0)
        
        # 约束范围 [0.25, 0.65]
        self.beta_min = 0.25
        self.beta_max = 0.65
        
        self.downsample = nn.AdaptiveAvgPool2d((low_res, low_res))
        
    def compute_smooth(self, feats, probs):
        """计算特征亲和力并平滑"""
        feat_norm = F.normalize(feats, p=2, dim=-1)
        affinity = torch.bmm(feat_norm, feat_norm.transpose(1, 2))
        affinity = F.softmax(affinity / self.temperature.clamp(min=0.01), dim=-1)
        smoothed = torch.bmm(affinity, probs.unsqueeze(2)).squeeze(2)
        return smoothed

    def forward(self, visual_feats, anomaly_probs, text_guide):
        """
        visual_feats: (B, N, C)
        anomaly_probs: (B, N)
        text_guide: (B, 768) - 动态异常文本特征
        """
        B, N, C = visual_feats.shape
        H, W = self.high_res, self.high_res
        
        # 1. 预测融合权重 beta
        # Global Visual Context
        global_feat = visual_feats.mean(dim=1) # (B, 768)
        
        # Text Context
        text_ctx = self.text_proj(text_guide) # (B, 128)
        
        # Feature Fusion
        fusion_feat = torch.cat([global_feat, text_ctx], dim=1) # (B, 896)
        
        # Predict Raw Beta
        raw_out = self.class_aware_weight(fusion_feat)
        
        # Constrain Beta [0.25, 0.65]
        beta = self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(raw_out)
        
        # 2. 准备特征
        feats_proj = self.proj(visual_feats)
        
        # --- High Res Path (32x32) ---
        smoothed_high = self.compute_smooth(feats_proj, anomaly_probs)
        
        # --- Low Res Path (16x16) ---
        feats_img = feats_proj.permute(0, 2, 1).view(B, -1, H, W)
        probs_img = anomaly_probs.view(B, 1, H, W)
        
        # Downsample
        feats_low_img = self.downsample(feats_img)
        probs_low_img = self.downsample(probs_img)
        
        feats_low = feats_low_img.flatten(2).permute(0, 2, 1)
        probs_low = probs_low_img.flatten(2).permute(0, 2, 1).squeeze(2)
        
        # Smooth at Low Res (强力去噪)
        smoothed_low_small = self.compute_smooth(feats_low, probs_low)
        
        # Upsample back
        L = self.low_res
        smoothed_low_img = smoothed_low_small.view(B, 1, L, L)
        smoothed_low_high_res_img = F.interpolate(smoothed_low_img, size=(H, W), mode='bilinear', align_corners=False)
        smoothed_low_high_res = smoothed_low_high_res_img.view(B, N)
        
        # 3. 自适应融合
        # beta 是 High-Res 的权重 (Transistor 想要大 beta，Cable 想要小 beta)
        smoothed_combined = beta * smoothed_high + (1 - beta) * smoothed_low_high_res
        
        # 4. 最终与原始图融合 (Residual)
        alpha = torch.sigmoid(self.alpha)
        final_probs = (1 - alpha) * anomaly_probs + alpha * smoothed_combined
        
        return final_probs

class SimpleAdapter(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // 4, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(c_in // 4, c_in, bias=False)
        )
        self.scale = nn.Parameter(torch.ones(1) * 1e-4)
    def forward(self, x): return x + self.scale * self.fc(x)

# =========================================================================
# 主模型: CLIP_Inplanted
# =========================================================================
class CLIP_Inplanted(nn.Module):
    def __init__(self, c_in=1024, device='cuda', **kwargs):
        super().__init__()
        self.device = device
        
        self.vcp = VisualConditionedPrompting(visual_dim=1024, text_dim=768)
        self.map_projector = nn.Linear(1024, 768, bias=False)
        
        # 初始化 TG-AMS-FSC
        self.fsc = TextGuidedAdaptiveMultiScaleFSC(feature_dim=768)
        
        self.blocks = nn.ModuleList()
        for _ in range(4):
            self.blocks.append(nn.ModuleList([
                SimpleAdapter(1024),
                LanguageGuidedAttention(visual_dim=1024, text_dim=768),
                LocalPerceptionFFN(1024)
            ]))
        self.prompt_adapters = nn.ModuleList([SimpleAdapter(768) for _ in range(2)])

    def forward_text(self, text_feat):
        f_norm = self.prompt_adapters[0](text_feat[:, :, 0])
        f_anom = self.prompt_adapters[1](text_feat[:, :, 1])
        return torch.stack([f_norm, f_anom], dim=2)

    def forward_visual(self, patch_list, cls_list, text_static):
        processed_patches = []
        processed_cls = []
        
        global_visual_context = torch.mean(torch.stack(cls_list, dim=0), dim=0)
        text_dynamic = self.vcp(text_static, global_visual_context) 
        anomaly_text_dynamic = text_dynamic[:, :, 1] # (B, 768)
        
        for i in range(4):
            patches = patch_list[i]
            cls_token = cls_list[i]
            adapter, lga, ffn = self.blocks[i]
            
            x = adapter(patches)
            x = lga(x, anomaly_text_dynamic)
            x = ffn(x)                        
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-6) 
            
            c = adapter(cls_token) 
            c = c / (c.norm(dim=-1, keepdim=True) + 1e-6)
            
            x_proj = self.map_projector(x)
            c_proj = self.map_projector(c)
            
            x_proj = x_proj / (x_proj.norm(dim=-1, keepdim=True) + 1e-6)
            c_proj = c_proj / (c_proj.norm(dim=-1, keepdim=True) + 1e-6)
            
            processed_patches.append(x_proj)
            processed_cls.append(c_proj)
            
        # --- TG-AMS-FSC Calibration ---
        calibrated_maps = []
        calibrated_scores = []
        
        for i in range(4):
            p_feat = processed_patches[i] 
            c_feat = processed_cls[i]
            if c_feat.dim() == 3: c_feat = c_feat.squeeze(1)
            
            # 1. 原始相似度
            sim_map = 100 * torch.bmm(p_feat, text_dynamic)
            prob_map = torch.softmax(sim_map, dim=-1)[:, :, 1]
            
            # 2. 文本引导的自适应多尺度校准
            # 传入 anomaly_text_dynamic 帮助 Meta-Net 决策
            calib_map = self.fsc(p_feat, prob_map, anomaly_text_dynamic)
            calibrated_maps.append(calib_map)
            
            # Image Score
            sim_cls = 100 * torch.bmm(c_feat.unsqueeze(1), text_dynamic).squeeze(1)
            prob_cls = torch.softmax(sim_cls, dim=-1)[:, 1]
            calibrated_scores.append(prob_cls)
            
        return calibrated_maps, calibrated_scores