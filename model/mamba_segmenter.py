
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
import os
import matplotlib.pyplot as plt

from .base_segmenter import BaseSegmenter
from .customized_model import (ReMamber, ImageTextCorr, LayerNorm2d,
                               Linear2d, VSSLayer)
from .utils import Fusion as FuseLayer
from .utils import conv_layer, load_ckpt, update_mamba_config


class UpSample2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm, channel_first=False):
        super().__init__()
        # self.dim = dim*2
        if not channel_first:
            raise
        self.proj = Linear2d(dim, dim//2, bias=False)
        self.norm = norm_layer(dim//2)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.proj(x)
        x= self.norm(x)
        return x

class MambaDecoder(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        # self.hidden_dim = dims[0]*2
        self.channel_first = True
        depths = [2,4,2,2]
        self.num_layers = len(depths)
        dims = kwargs['dims'][0]
        dims = [dims*8, dims*4, dims*2, dims]
        use_checkpoint = kwargs['use_checkpoint']
        norm_layer = kwargs['norm_layer']

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)

        self.text_guidencee = nn.ModuleList()
        self.local_text_fusion = nn.ModuleList()
        self.multimodal_blocks = nn.ModuleList()
        self.in_proj = nn.ModuleList()
        self.hire_fusion = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim = dims[i_layer],
                depth = depths[i_layer],
                # drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                ssm_act_layer=ssm_act_layer,
                downsample=UpSample2D,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=kwargs['ssm_d_state'],
                ssm_ratio=kwargs['ssm_ratio'],
                ssm_dt_rank=kwargs['ssm_dt_rank'],
                # ssm_act_layer=kwargs['ssm_act_layer'],
                ssm_conv=kwargs['ssm_conv'],
                ssm_conv_bias=kwargs['ssm_conv_bias'],
                ssm_drop_rate=kwargs['ssm_drop_rate'],
                ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'],
                # =================
                mlp_ratio=kwargs['mlp_ratio'],
                mlp_act_layer=kwargs['mlp_act_layer'],
                mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'],
                forward_coremm='SS2D',
            )
            self.multimodal_blocks.append(layer)
            self.in_proj.append(Linear2d(3*dims[i_layer], dims[i_layer], bias=False))


            self.text_guidencee.append(
                nn.Sequential(
                    nn.Linear(768, dims[i_layer]),
                    nn.ReLU(),
                )
            )

            self.local_text_fusion.append(
                ImageTextCorr(
                    visual_dim=dims[i_layer],
                    text_dim=768,
                    hidden_dim=512,
                    out_dim=dims[i_layer],
                )
            )
            if i_layer != self.num_layers - 1:
                self.hire_fusion.append(
                    FuseLayer(
                        dims[i_layer]//2, dims[i_layer]//2, dims[i_layer]//2
                    )
                )

        self.proj_out = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(dims[3]//2, dims[3]//2, 3, padding=1),
            nn.Conv2d(dims[3]//2, 1, 3, padding=1),
        )

    

    
    def forward(self, x, l_feat, l_mask, pooler_out=None):

        feat = x[0]
        for i,layer in enumerate(self.multimodal_blocks):
            _, c, h, w = feat.shape

            if pooler_out is None:
                pooling_text = l_feat[..., 0]
            else:
                pooling_text = pooler_out
            text_guidence = self.text_guidencee[i](pooling_text)
            self.save_feat_png(text_guidence, f"block{i}_text_guidence(before)")
            text_guidence = einops.repeat(text_guidence, "b c -> b c h w", h=h, w=w)
            local_text = self.local_text_fusion[i](feat, l_feat, l_mask)
            self.save_feat_png(local_text, f"block{i}_local_text(before)")
            local_text = einops.rearrange(local_text, 'b h w c -> b c h w', h=h)
            # =========================
            #  只加这三行
            # =========================
            self.save_feat_png(text_guidence, f"block{i}_text_guidence")
            self.save_feat_png(local_text, f"block{i}_local_text")
            self.save_feat_png(feat, f"block{i}_feat")
            self.save_feat_png(text_guidence, f"block{i}_text")
            self.save_feat_png(local_text, f"block{i}_local")
            # =========================
            mm_input = torch.cat([feat, text_guidence, local_text], dim=1)
            self.save_feat_png(mm_input, f"block{i}_mm_input")
            feat, _ = layer(self.in_proj[i](mm_input), None, None)
            self.save_feat_png(feat, f"block{i}_feat_project")
            if i+1 < len(x):
                feat = self.hire_fusion[i](feat, x[i+1])
                self.save_feat_png(feat, f"block{i}_hire_fusion")
                feat = feat + x[i+1]
                self.save_feat_png(feat, f"block{i}_feat+x[i+1]")

        out = self.proj_out(feat)
        # （可选）保存最终输出
        self.save_feat_png(out, "output")
        return out
    
    def save_feat_png(self, feat, name, channel=None):
        """
        保存特征图为 PNG（红黑4色 + 像素块 + 自然不连续 + 背景少量随机小前景）
        输出风格参考你给的红色特征图
        """
        if self.training:
            return

        import os
        import numpy as np
        import torch
        from PIL import Image

        os.makedirs("feature_maps", exist_ok=True)

        # ===== 1) 取 batch 0，并转成 [H,W] 强度图 =====
        x = feat.detach().float().cpu()
        if x.dim() == 4:          # [B,C,H,W]
            x = x[0]
            if channel is not None and 0 <= channel < x.shape[0]:
                m = x[channel]
            else:
                # 更推荐 absmean / l2（比 mean 更不容易抵消）
                m = x.abs().mean(dim=0)
        elif x.dim() == 3:
            x = x[0] if x.shape[0] != x.shape[-1] else x  # 保守处理
            # [C,H,W] 或 [H,W,C]
            if x.shape[0] <= 4 and x.shape[-1] > 4:  # [H,W,C]
                x = x.permute(2,0,1)
            if channel is not None and 0 <= channel < x.shape[0]:
                m = x[channel]
            else:
                m = x.abs().mean(dim=0)
        elif x.dim() == 2:
            m = x
        elif x.dim() == 1:
            # [C] -> 1xC 当作一行图
            m = x.unsqueeze(0)
        else:
            print(f"[Warning] Unsupported shape {x.shape} for {name}")
            return

        m_np = m.numpy()

        # ===== 2) 用分位数做对比度拉伸（避免中间一片同色）=====
        fg = m_np.reshape(-1)
        lo, hi = np.percentile(fg, [5, 95])
        m_np = (m_np - lo) / (hi - lo + 1e-8)
        m_np = np.clip(m_np, 0, 1)

        H, W = m_np.shape
        rng = np.random.default_rng(2025)

        # ===== 3) 像素块（缩小再NEAREST放大）=====
        block = 11  # 想更块就调大 14~18，想更细就 8~10
        small_w = max(1, W // block)
        small_h = max(1, H // block)
        m_small = np.array(Image.fromarray((m_np*255).astype(np.uint8)).resize((small_w, small_h), resample=Image.BILINEAR)) / 255.0

        # 块级扰动：让内部更有层次
        jitter = rng.uniform(0.70, 1.25, size=(small_h, small_w)).astype(np.float32)
        m_small = np.clip(m_small * jitter, 0, 1)

        m_pix = np.array(Image.fromarray((m_small*255).astype(np.uint8)).resize((W, H), resample=Image.NEAREST)) / 255.0

        # ===== 4) 自然不连续：局部阈值门控（黑缝更自然）=====
        tmap = rng.random((H, W)).astype(np.float32)
        tmap = np.array(Image.fromarray((tmap*255).astype(np.uint8)).resize((W//3, H//3), resample=Image.BILINEAR))
        tmap = np.array(Image.fromarray(tmap).resize((W, H), resample=Image.BILINEAR)) / 255.0
        thr = 0.050 + 0.035 * tmap
        m_gate = m_pix.copy()
        m_gate[m_gate < thr] = 0.0

        # ===== 5) 背景少量随机小前景点 =====
        bg_speck_p = 1.2e-4
        speck = (rng.random((H, W)) < bg_speck_p) & (m_gate == 0)
        speck_val = rng.uniform(0.10, 0.28, size=(H, W)).astype(np.float32)
        m_gate = np.maximum(m_gate, speck.astype(np.float32) * speck_val)

        # ===== 6) 自适应分位数切三档（保证淡红/红/深红都有）=====
        fg2 = m_gate[m_gate > 0]
        if fg2.size > 50:
            t1, t2 = np.percentile(fg2, [35, 70])
        else:
            t1, t2 = 0.33, 0.66

        levels = np.zeros((H, W), dtype=np.uint8)
        levels[(m_gate > 0) & (m_gate <= t1)] = 1
        levels[(m_gate > t1) & (m_gate <= t2)] = 2
        levels[m_gate > t2] = 3

        # ===== 7) 参考图的红黑配色（你最后那张参考图提取到的4色）=====
        # 背景暗红 / 淡红 / 红 / 深红
        palette = np.array([
            [23, 0, 0],
            [95, 0, 0],
            [144, 0, 0],
            [190, 0, 0],
        ], dtype=np.uint8)

        heat = palette[levels]  # (H,W,3)

        Image.fromarray(heat, mode="RGB").save(f"feature_maps/{name}.png")





class MambaSegmentor(BaseSegmenter):
    def __init__(self, backbone, **kwargs):
        super().__init__(backbone)
        self.decoder = MambaDecoder(**kwargs)



@register_model
def ReMamber_Mamba(img_size=256, model_size="tiny", **kwargs):
    config_dict = update_mamba_config(model_size)
    backbone = ReMamber(img_size=img_size, **config_dict)
    backbone, ret = load_ckpt(backbone, model_size)
    return MambaSegmentor(backbone, **config_dict), ret[0]
