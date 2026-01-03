import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizerFast

from .utils import dice_loss, sigmoid_focal_loss


class BaseSegmenter(nn.Module):
    def __init__(self, backbone, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.decoder = None
        self.tokenizer = CLIPTokenizerFast.from_pretrained('/root/autodl-fs/openai/clip-vit-large-patch14')
        self.text_encoder = CLIPTextModel.from_pretrained('/root/autodl-fs/openai/clip-vit-large-patch14')

    def forward(self, x, text, mask=None, **kwargs):
        encode_text = self.tokenizer(text, padding='max_length', truncation=True, max_length=20, return_tensors='pt')
        text = encode_text['input_ids'].to(x.device, non_blocking=True)
        l_mask = encode_text['attention_mask'].to(x.device, non_blocking=True)

        input_shape = x.shape[-2:]
        ret = self.text_encoder(text, attention_mask=l_mask)  # (6, 10, 768)
        l_feats = ret['last_hidden_state']
        l_feats = l_feats.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        l_mask = l_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)
        if 'pooler_output' in ret:
            pooler_out = ret['pooler_output']
        else:
            pooler_out = None
        
        features = self.backbone(x, l_feats, l_mask, pooler_out=pooler_out)
        x_c1, x_c2, x_c3, x_c4 = features
        pred = self.decoder([x_c4, x_c3, x_c2, x_c1], l_feats, l_mask)
        pred = F.interpolate(pred, input_shape, mode='bilinear', align_corners=True)
        
        # loss
        if self.training:
            loss = dice_loss(pred, mask) + sigmoid_focal_loss(pred, mask, alpha=-1, gamma=0)
            return pred.detach(), mask, loss
        else:
            return pred.detach()

# #自己修改的   假设您已经加载了模型
# class BaseSegmenter(nn.Module):
#     def __init__(self, backbone, **kwargs):
#         super().__init__()
#         self.backbone = backbone
#         self.decoder = None
#         self.tokenizer = CLIPTokenizerFast.from_pretrained('/root/autodl-fs/openai/clip-vit-large-patch14')
#         self.text_encoder = CLIPTextModel.from_pretrained('/root/autodl-fs/openai/clip-vit-large-patch14')

#         # 创建文件夹来保存特征图
#         self.output_dir = './feature_maps'
#         if not os.path.exists(self.output_dir):
#             os.makedirs(self.output_dir)

#     def save_feature_map(self, feature_map, layer_name, index):
#         """保存特征图到本地文件夹"""
#         num_features = feature_map.shape[1]  # 通道数
#         for i in range(min(10, num_features)):  # 最多保存前10个特征图
#             feature_map_image = feature_map[0, i, :, :].cpu().numpy()
#             plt.imsave(f"{self.output_dir}/{layer_name}_feature_{index}_{i}.png", feature_map_image, cmap='viridis')

#     def forward(self, x, text, mask=None, **kwargs):
#         encode_text = self.tokenizer(text, padding='max_length', truncation=True, max_length=20, return_tensors='pt')
#         text = encode_text['input_ids'].to(x.device, non_blocking=True)
#         l_mask = encode_text['attention_mask'].to(x.device, non_blocking=True)

#         input_shape = x.shape[-2:]
#         ret = self.text_encoder(text, attention_mask=l_mask)  # (6, 10, 768)
#         l_feats = ret['last_hidden_state']
#         l_feats = l_feats.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
#         l_mask = l_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)
#         if 'pooler_output' in ret:
#             pooler_out = ret['pooler_output']
#         else:
#             pooler_out = None
        
#         # 获取中间特征图
#         features = self.backbone(x, l_feats, l_mask, pooler_out=pooler_out)
#         x_c1, x_c2, x_c3, x_c4 = features
        
#         # 保存特征图
#         self.save_feature_map(x_c1, 'backbone_x_c1', 1)
#         self.save_feature_map(x_c2, 'backbone_x_c2', 2)
#         self.save_feature_map(x_c3, 'backbone_x_c3', 3)
#         self.save_feature_map(x_c4, 'backbone_x_c4', 4)

#         # 后续的解码和预测
#         pred = self.decoder([x_c4, x_c3, x_c2, x_c1], l_feats, l_mask)
#         pred = F.interpolate(pred, input_shape, mode='bilinear', align_corners=True)
        
#         # loss
#         if self.training:
#             loss = dice_loss(pred, mask) + sigmoid_focal_loss(pred, mask, alpha=-1, gamma=0)
#             return pred.detach(), mask, loss
#         else:
#             return pred.detach()
