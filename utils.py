import random
import numpy as np
import torch

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ========== 新增：SimSiam对比学习工具 ==========
import torch
import torch.nn as nn
import torch.nn.functional as F

def simsiam\_aug(img\_size=224):
    """
    SimSiam用的强增强：同一张图变两次，每次不一样
    """
    from torchvision import transforms
    return transforms.Compose(\[
        transforms.Resize(img\_size + 32),
        transforms.RandomCrop(img\_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(\[
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        \], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=\[0.485, 0.456, 0.406\],
                           std=\[0.229, 0.224, 0.225\]),
    \])

class TwoViewDataset(torch.utils.data.Dataset):
    """
    包装原有数据集，返回同一张图的两个增强视图
    """
    def \_\_init\_\_(self, base\_dataset, transform):
        self.base = base\_dataset
        self.transform = transform
    
    def \_\_len\_\_(self):
        return len(self.base)
    
    def \_\_getitem\_\_(self, idx):
        img, label = self.base\[idx\]
        x1 = self.transform(img)  # 第一次增强
        x2 = self.transform(img)  # 第二次增强（随机性导致不同）
        return x1, x2, label

def simsiam\_loss(p, z):
    """
    SimSiam核心损失：预测器输出p去逼近投影输出z（z停止梯度）
    """
    z = z.detach()  # 关键！z不更新
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return -(p \* z).sum(dim=1).mean()

class SimSiamHead(nn.Module):
    """
    SimSiam的投影头+预测头
    """
    def \_\_init\_\_(self, dim=768, proj\_dim=2048, pred\_dim=512):
        super().\_\_init\_\_()
        
        # 投影头：768 → 2048 → 2048
        self.projector = nn.Sequential(
            nn.Linear(dim, proj\_dim),
            nn.BatchNorm1d(proj\_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj\_dim, proj\_dim),
        )
        
        # 预测头：2048 → 512 → 2048（SimSiam关键，防止坍塌）
        self.predictor = nn.Sequential(
            nn.Linear(proj\_dim, pred\_dim),
            nn.BatchNorm1d(pred\_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred\_dim, proj\_dim),
        )
    
    def forward(self, feat):
        z = self.projector(feat)   # 投影特征
        p = self.predictor(z)       # 预测特征
        return p, z
        def total\_sim\_loss(z1, p1, z2, p2):
    """双向对称SimSiam总损失"""
    loss\_1 = simsiam\_loss(p1, z2)
    loss\_2 = simsiam\_loss(p2, z1)
    return (loss\_1 + loss\_2) / 2

def sep\_reg(feat\_batch):
    """类间分离正则，缓解多视图带来特征全部趋同问题"""
    b\_size = feat\_batch.shape\[0\]
    feat\_norm = F.normalize(feat\_batch, dim=-1)
    gram\_matrix = torch.matmul(feat\_norm, feat\_norm.T)
    eye\_mask = torch.eye(b\_size, device=feat\_batch.device)
    # 只计算不同样本间相似度
    cross\_sim = gram\_matrix \* (1 - eye\_mask)
    return cross\_sim.sum() / (b\_size \* (b\_size - 1))
