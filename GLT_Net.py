# -*- coding:utf-8 -*-
# @Time       :2022/9/7 上午9:57
# @AUTHOR     :DingKexin
# @FileName   :GLT_Net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def img2seq(x):
    [b, c, h, w] = x.shape
    x = x.reshape((b, c, h*w))
    return x


def seq2img(x):
    [b, c, d] = x.shape
    p = int(d ** .5)
    x = x.reshape((b, c, p, p))
    return x


class CNN_Encoder(nn.Module):
    def __init__(self, l1, l2):
        super(CNN_Encoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(l1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),  # No effect on order
            # nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(l2, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),  # No effect on order
            # nn.MaxPool2d(2),
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # No effect on order
            nn.MaxPool2d(2),
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # No effect on order
            nn.MaxPool2d(2),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # No effect on order
            nn.MaxPool2d(2),
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # No effect on order
            nn.MaxPool2d(2),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # No effect on order
            nn.MaxPool2d(2),
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # No effect on order
            nn.MaxPool2d(2),
        )
        self.xishu1 = torch.nn.Parameter(torch.Tensor([0.5]))  # lamda
        self.xishu2 = torch.nn.Parameter(torch.Tensor([0.5]))  # 1 - lamda

    def forward(self, x11, x21, x12, x22, x13, x23):

        x11 = self.conv1(x11)
        x21 = self.conv2(x21)
        x12 = self.conv1(x12)
        x22 = self.conv2(x22)
        x13 = self.conv1(x13)
        x23 = self.conv2(x23)

        x1_1 = self.conv1_1(x11)
        x2_1 = self.conv2_1(x21)
        x_add1 = x1_1 * self.xishu1 + x2_1 * self.xishu2

        x1_2 = self.conv1_2(x12)
        x2_2 = self.conv2_2(x22)
        x_add2 = x1_2 * self.xishu1 + x2_2 * self.xishu2

        x1_3 = self.conv1_3(x13)
        x2_3 = self.conv2_3(x23)
        x_add3 = x1_3 * self.xishu1 + x2_3 * self.xishu2

        return x_add1, x_add2, x_add3


class CNN_Decoder(nn.Module):
    def __init__(self, l1, l2):
        super(CNN_Decoder, self).__init__()

        self.dconv1 = nn.Sequential(
            nn.Conv2d(64, l1, 3, 1, 1),
            nn.Sigmoid(),

        )
        self.dconv2 = nn.Sequential(
            nn.Conv2d(64, l2, 3, 1, 1),
            nn.Sigmoid(),

        )
        self.dconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2),  # add Upsample
            nn.Conv2d(64, l1, 3, 1, 1),
            nn.Sigmoid(),

        )
        self.dconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2),  # add Upsample
            nn.Conv2d(64, l2, 3, 1, 1),
            nn.Sigmoid(),

        )
        self.dconv5 = nn.Sequential(
            nn.Upsample(scale_factor=3),  # add Upsample
            nn.Conv2d(64, l1, 3, 1, 1),
            nn.Sigmoid(),

        )
        self.dconv6 = nn.Sequential(
            nn.Upsample(scale_factor=3),  # add Upsample
            nn.Conv2d(64, l2, 3, 1, 1),
            nn.Sigmoid(),

        )

    def forward(self, x_con1):
        x1 = self.dconv1(x_con1)
        x2 = self.dconv2(x_con1)

        x3 = self.dconv3(x_con1)
        x4 = self.dconv4(x_con1)

        x5 = self.dconv5(x_con1)
        x6 = self.dconv6(x_con1)
        return x1, x2, x3, x4, x5, x6


class CNN_Classifier(nn.Module):
    def __init__(self, Classes):
        super(CNN_Classifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, Classes, 1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x_out = F.softmax(x, dim=1)

        return x_out


class SA_GDR(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA_GDR, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, x3, dim):
        p = int(x1.shape[2] ** .5)
        x1 = x1.reshape((x1.shape[0], x1.shape[1], p, p))
        x2 = x2.reshape((x2.shape[0], x2.shape[1], p, p))
        x3 = x3.reshape((x3.shape[0], x3.shape[1], p, p))
        num1 = x1.shape[1] // dim
        num2 = x2.shape[1] // dim
        num3 = x3.shape[1] // dim
        x_out = torch.empty(x1.shape[0], dim, p, p).cuda()
        for i in range(dim):
            x1_tmp = x1[:, i * num1:(i + 1) * num1, :, :]
            x2_tmp = x2[:, i * num2:(i + 1) * num2, :, :]
            x3_tmp = x3[:, i * num3:(i + 1) * num3, :, :]
            x_tmp1 = torch.cat((x1_tmp, x2_tmp, x3_tmp), dim=1)
            avgout = torch.mean(x_tmp1, dim=1, keepdim=True)
            maxout, _ = torch.max(x_tmp1, dim=1, keepdim=True)
            x_tmp2 = torch.cat([avgout, maxout], dim=1)
            x_tmp3 = self.conv(x_tmp2)
            x_tmp4 = self.sigmoid(x_tmp3)
            x_out[:, i:i+1, :, :] = x_tmp4
        x_out = x_out.reshape(x_out.shape[0], dim, p*p)
        return x_out


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
            ]))

        self.skipcat = nn.ModuleList([])
        for _ in range(depth - 2):
            self.skipcat.append(nn.Conv2d(num_channel + 1, num_channel + 1, [1, 2], 1, 0))

    def forward(self, x, mask=None):

        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class GLT(nn.Module):
    def __init__(self, l1, l2, patch_size, num_patches, num_classes, encoder_embed_dim, decoder_embed_dim, en_depth, en_heads,
                 de_depth, de_heads, mlp_dim, dim_head=16, dropout=0., emb_dropout=0.):
        super().__init__()
        self.cnn_encoder = CNN_Encoder(l1, l2)
        self.cnn_decoder = CNN_Decoder(l1, l2)
        self.cnn_classifier = CNN_Classifier(num_classes)
        self.coefficient1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.coefficient2 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.sa_gdr = SA_GDR()
        self.loss_fun2 = nn.MSELoss()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_pos_embed = nn.Parameter(torch.randn(1, self.patch_size ** 2 + 1, encoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.patch_size ** 2 + 1, decoder_embed_dim))
        self.encoder_embedding1 = nn.Linear(((patch_size // 2) * 1) ** 2, self.patch_size ** 2)
        self.encoder_embedding2 = nn.Linear(((patch_size // 2) * 2) ** 2, self.patch_size ** 2)
        self.encoder_embedding3 = nn.Linear(((patch_size // 2) * 3) ** 2, self.patch_size ** 2)
        self.decoder_embedding = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.cls_token = nn.Parameter(torch.randn(1, 1, encoder_embed_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.en_transformer = Transformer(encoder_embed_dim, en_depth, en_heads, dim_head, mlp_dim, dropout,
                                          num_patches)
        self.de_transformer = Transformer(decoder_embed_dim, de_depth, de_heads, dim_head, mlp_dim, dropout,
                                          num_patches)
        self.decoder_pred1 = nn.Linear(decoder_embed_dim, 64, bias=True)  # decoder to patch
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(encoder_embed_dim),
            nn.Linear(encoder_embed_dim, num_classes)
        )

    def encoder(self, x11, x21, x12, x22, x13, x23):

        x_fuse1, x_fuse2, x_fuse3 = self.cnn_encoder(x11, x21, x12, x22, x13, x23)  # x_fuse1:64*64*8*8, x_fuse2:64*64*4*4, x_fuse2:64*64*2*2
        x_flat1 = x_fuse1.flatten(2)
        x_flat2 = x_fuse2.flatten(2)
        x_flat3 = x_fuse3.flatten(2)

        x_1 = self.encoder_embedding1(x_flat1)
        x_2 = self.encoder_embedding2(x_flat2)
        x_3 = self.encoder_embedding3(x_flat3)

        x_cnn = self.sa_gdr(x_1, x_2, x_3, self.encoder_embed_dim)

        x_cnn = torch.einsum('nld->ndl', x_cnn)

        b, n, _ = x_cnn.shape
        # add pos embed w/o cls token
        x = x_cnn + self.encoder_pos_embed[:, 1:, :]

        # append cls token
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        # add position embedding
        x += self.encoder_pos_embed[:, :1]
        x = self.dropout(x)

        x = self.en_transformer(x, mask=None)

        return x, x_cnn

    def classifier(self, x, x_cnn):
        # classification: using cls_token output
        x = self.to_latent(x[:, 0])

        # MLP classification layer
        x_cls1 = self.mlp_head(x)

        x_cnn = torch.einsum('ndl->nld', x_cnn)

        x_cls2 = self.cnn_classifier(seq2img(x_cnn))

        x_cls = x_cls1 * self.coefficient1 + x_cls2 * self.coefficient2
        return x_cls

    def decoder(self, x, imgs11, imgs21, imgs12, imgs22, imgs13, imgs23):

        # embed tokens
        x = self.decoder_embedding(x)
        """ with or without decoder_pos_embed"""
        # add pos embed
        x += self.decoder_pos_embed

        x = self.de_transformer(x, mask=None)

        # predictor projection
        x_1 = self.decoder_pred1(x)

        # remove cls token
        x_con1 = x_1[:, 1:, :]

        x_con1 = torch.einsum('nld->ndl', x_con1)

        x_con1 = x_con1.reshape((x_con1.shape[0], x_con1.shape[1], self.patch_size, self.patch_size))

        x1, x2, x3, x4, x5, x6 = self.cnn_decoder(x_con1)

        # con_loss
        con_loss1 = 0.5 * self.loss_fun2(x1, imgs11) + 0.5 * self.loss_fun2(x2, imgs21)
        con_loss2 = 0.5 * self.loss_fun2(x3, imgs12) + 0.5 * self.loss_fun2(x4, imgs22)
        con_loss3 = 0.5 * self.loss_fun2(x5, imgs13) + 0.5 * self.loss_fun2(x6, imgs23)
        con_loss = 1/3 * con_loss1 + 1/3 * con_loss2 + 1/3 * con_loss3

        return con_loss

    def forward(self, img11, img21, img12, img22, img13, img23):

        x_vit, x_cnn = self.encoder(img11, img21, img12, img22, img13, img23)
        con_loss = self.decoder(x_vit, img11, img21, img12, img22, img13, img23)
        x_cls = self.classifier(x_vit, x_cnn)
        return x_cls, con_loss


