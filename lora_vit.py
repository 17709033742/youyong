import torch
import torch.nn as nn
import timm
import loralib as lora


def replace_linear_with_lora(module: nn.Module, r=8, lora_alpha=16, lora_dropout=0.0):
    """递归将所有 nn.Linear 替换为 lora.Linear，并拷贝预训练权重。"""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            new_layer = lora.Linear(
                child.in_features,
                child.out_features,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=(child.bias is not None),
            )
            # 拷贝底座权重
            with torch.no_grad():
                new_layer.weight.copy_(child.weight)
                if child.bias is not None:
                    new_layer.bias.copy_(child.bias)
            setattr(module, name, new_layer)
        else:
            replace_linear_with_lora(child, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)


class ViTWithLoRA(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, r=8, lora_alpha=16, lora_dropout=0.0):
        super().__init__()
        # num_classes=0 让 timm 返回特征而不是分类 logits
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        replace_linear_with_lora(self.backbone, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        # 冻结所有参数，仅训练 LoRA 参数（lora_ 前缀）
        for p in self.backbone.parameters():
            p.requires_grad = False
        for n, p in self.backbone.named_parameters():
            if 'lora_' in n:
                p.requires_grad = True

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        feats = self.backbone(x)  # [B, D]
        return nn.functional.normalize(feats, dim=-1)

    def forward(self, x):
        return self.backbone(x)