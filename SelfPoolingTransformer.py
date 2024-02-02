# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=16, kernel_size=1, proj_ratio=2.):
        super(PatchEmbedding, self).__init__()
        embedding_dim = int(in_channels * proj_ratio)
        if kernel_size == 1:
            kernel_size = (kernel_size,) * 2
            self.proj = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim, kernel_size=kernel_size)
        else:
            padding = kernel_size // 2
            kernel_size = (kernel_size,) * 2
            self.proj = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        x = self.proj(x)
        return x


class CTM(nn.Module):
    def __init__(self, num_head=4):
        super(CTM, self).__init__()
        self.num_head = num_head

    def forward(self, x):
        B, C, H, W = x.shape

        x = x.reshape(B, self.num_head, C // self.num_head, -1)
        # x.shape = [B, num_head, C // num_head, H * W]

        # -------- Attention mode 1 -------- #
        x_ = x
        x_ = x_ / torch.linalg.norm(x_, dim=2).unsqueeze(2)
        # x_.shape = [B, num_head, C // num_head, H * W]

        mid = x_.shape[-1] // 2
        x0 = x_[..., mid]
        # x0.shape = [B, num_head, C // num_head]

        attn = x0.unsqueeze(2) @ x_
        # attn.shape = [B, num_head, 1, H * W]
        attn = attn.softmax(dim=-1)

        x = x * attn * H * W
        x = x.reshape(B, C, -1).reshape(B, C, H, W)
        # -------- Attention mode 1 -------- #

        # -------- Attention mode 2 -------- #
        # # x.shape = [B, num_head, C // num_head, H * W]
        # attn = x.transpose(-2, -1) @ x
        # # attn.shape = [B, num_head, H * W, H * W]
        # attn = attn.softmax(dim=-1)
        # x = x @ attn
        # x = x.reshape(B, C, -1).reshape(B, C, H, W)
        # -------- Attention mode 2 -------- #

        return x


class ChannelShuffle(nn.Module):
    def __init__(self, group):
        super(ChannelShuffle, self).__init__()
        self.group = group

    def forward(self, x):
        B, C, H, W = x.shape
        assert C % self.group == 0
        group_C = C // self.group
        x = x.view(B, self.group, group_C, H, W)
        x = x.transpose(1, 2).contiguous()
        x = x.view(B, -1, H, W)
        return x


class MHSP(nn.Module):
    def __init__(self, dim, num_head=4, kernel_size=1, stride=1, padding=0,
                 channel_shuffle=True, sparse_mapping=True):
        super(MHSP, self).__init__()
        self.num_head = num_head
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.channel_shuffle = ChannelShuffle(dim // num_head) if channel_shuffle else nn.Identity()
        self.sparse_mapping = sparse_mapping
        self.sm = self.SM(n=dim // num_head)

    def SM(self, n):
        indices = [[i * n + j for i in range(n) for j in range(i, n)]]
        indices.append(list(range(len(indices[0]))))
        indices = torch.Tensor(indices)
        values = torch.ones(len(indices[0]))
        sparse_mapping = torch.sparse_coo_tensor(
            indices=indices, values=values,
            size=[n ** 2, values.shape[0]],
            requires_grad=False
        ).to_dense() if self.sparse_mapping and n ** 2 > values.shape[0] else torch.eye(n ** 2)
        return sparse_mapping

    def forward(self, x):
        B, C, H, W = x.shape
        assert C % self.num_head == 0
        x = self.channel_shuffle(x)
        # x.shape = [B, C, H, W]
        x = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        # x.shape = [B, C * kernel_size[0] * kernel_size[1], L]
        x = x.transpose(-2, -1)
        # x.shape = [B, L, C * kernel_size[0] * kernel_size[1]]
        # C * kernel_size[0] * kernel_size[1] -> C
        B, L, C = x.shape
        x = x.reshape(B, L, self.num_head, C // self.num_head)
        x = x.unsqueeze(-1)
        poolfeature = x @ x.transpose(-2, -1)
        # poolfeature.shape = [B, L, num_head, C // self.num_head, C // self.num_head]
        poolfeature = poolfeature.reshape(B, L, self.num_head, -1)
        poolfeature = poolfeature @ self.sm.to(x.device)
        # poolfeature.shape = [B, L, num_head, D_feature]
        poolfeature = poolfeature.reshape(B, L, -1)
        poolfeature = poolfeature.transpose(-2, -1).reshape(B, -1, H, W)
        return poolfeature.contiguous()


class LightMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Conv2d(in_features, out_features, (1, 1))
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        return x


class SPFBlock(nn.Module):
    def __init__(self, dim, kernel_size=1, stride=1, padding=0, proj_ratio=2.):
        super(SPFBlock, self).__init__()

        self.norm1 = nn.LayerNorm(int(dim * proj_ratio))
        num_head = 4
        self.multiheadselfpooling = MHSP(dim=int(dim * proj_ratio), num_head=num_head, kernel_size=kernel_size,
                                         stride=stride, padding=padding, channel_shuffle=True, sparse_mapping=True)
        self.token_mixer = CTM(num_head=num_head)
        self.dimension_reduction = nn.Conv2d(
            in_channels=int((dim * proj_ratio + num_head) * (dim * proj_ratio) / num_head / 2),
            out_channels=dim, kernel_size=(1, 1)
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = LightMlp(in_features=dim, out_features=dim)

    def forward(self, x):

        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.multiheadselfpooling(x)
        x = self.token_mixer(x)
        x = self.dimension_reduction(x)

        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)
        x = self.mlp(x)

        return x


class SelfPoolingTransformer(nn.Module):
    def __init__(self, dim, autoencoder_path, kernel_size=1, n_class=9):

        super(SelfPoolingTransformer, self).__init__()
        ae = torch.load(autoencoder_path)
        in_channels = ae['encoder.0.weight'].shape[1]
        from AutoEncoder import AutoEncoder
        drmlp = AutoEncoder(in_channels=in_channels, out_channels=dim)
        drmlp.load_state_dict(ae, strict=True)
        self.encoder = drmlp.encoder

        for p in self.encoder.parameters():
            p.requires_grad = False

        # stage 1
        proj_ratio1 = 2
        self.patchembed1 = PatchEmbedding(in_channels=dim, kernel_size=kernel_size, proj_ratio=proj_ratio1)
        self.block1 = SPFBlock(dim=dim, proj_ratio=proj_ratio1)

        # stage 2
        proj_ratio2 = 2
        self.patchembed2 = PatchEmbedding(in_channels=dim, kernel_size=kernel_size, proj_ratio=proj_ratio2)
        self.block2 = SPFBlock(dim=dim, proj_ratio=proj_ratio2)

        # classify
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(dim, n_class)

    def forward(self, x):

        x = self.encoder(x)

        # stage 1
        x = self.patchembed1(x)
        x = self.block1(x)

        # stage 2
        x = self.patchembed2(x)
        x = self.block2(x)

        # classify
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


if __name__ == '__main__':
    x = torch.arange(1 * 8 * 3 * 3)
    x = x.float()
    x = x.reshape(1, 8, 3, 3)
    ctm = CTM(num_head=4)
    y = ctm(x)
