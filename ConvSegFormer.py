import torch.nn as nn
import torch
import numpy as np
import os
import cv2 as cv
import torchvision
import matplotlib
#matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.utils as vutils
from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset
import scipy.io as io
from torchsummary import summary
import glob
import torch.nn.functional as F
from PIL import Image
import PIL
import random
import time
import sys
#from torch_receptive_field import receptive_field
from patchify import patchify, unpatchify
import os
import random
import pandas as pd
from PIL import Image, ImageOps
from torch.nn import init
import math
import copy
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from einops.layers.torch import Reduce, Rearrange

from DataHandling import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from mmcv.runner import load_checkpoint
import math

class DoubleConv(nn.Module):
	"""(convolution => [BN] => ReLU) * 2"""

	def __init__(self, in_channels, out_channels, kernel_size = 3, mid_channels=None):
		super().__init__()
		if not mid_channels:
			mid_channels = out_channels
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, mid_channels, kernel_size= kernel_size, padding=kernel_size//2, dilation=1),
			nn.BatchNorm2d(mid_channels),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.double_conv(x)

class Down(nn.Module):
	"""Downscaling with maxpool then double conv"""

	def __init__(self, in_channels, out_channels, kernel_size = 3):
		super().__init__()
		self.maxpool_conv = nn.Sequential(
			nn.MaxPool2d(2),
			DoubleConv(in_channels, out_channels, kernel_size)
		)

	def forward(self, x):
		return self.maxpool_conv(x)

class Up(nn.Module):
	"""Upscaling then double conv"""

	def __init__(self, in_channels, out_channels, kernel_size = 3, bilinear=True):
		super().__init__()

		# if bilinear, use the normal convolutions to reduce the number of channels
		if bilinear:
			self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
			self.conv = DoubleConv(in_channels, out_channels, 3)#, in_channels // 2)
		else:
			self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
			self.conv = DoubleConv(in_channels, out_channels, kernel_size)


	def forward(self, x1, x2):
		x1 = self.up(x1)
		# input is CHW
		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]

		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
						diffY // 2, diffY - diffY // 2])
		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)

class Mlp(nn.Module):
	def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
		super().__init__()
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features
		self.fc1 = nn.Linear(in_features, hidden_features)
		self.dwconv = DWConv(hidden_features)
		self.act = act_layer()
		self.fc2 = nn.Linear(hidden_features, out_features)
		self.drop = nn.Dropout(drop)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)
		elif isinstance(m, nn.Conv2d):
			fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			fan_out //= m.groups
			m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
			if m.bias is not None:
				m.bias.data.zero_()

	def forward(self, x, H, W):
		x = self.fc1(x)
		x = self.dwconv(x, H, W)
		x = self.act(x)
		x = self.drop(x)
		x = self.fc2(x)
		x = self.drop(x)
		return x


class Attention(nn.Module):
	def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
		super().__init__()
		assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

		self.dim = dim
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = qk_scale or head_dim ** -0.5

		self.q = nn.Linear(dim, dim, bias=qkv_bias)
		self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

		self.sr_ratio = sr_ratio
		if sr_ratio > 1:
			self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
			self.norm = nn.LayerNorm(dim)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)
		elif isinstance(m, nn.Conv2d):
			fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			fan_out //= m.groups
			m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
			if m.bias is not None:
				m.bias.data.zero_()

	def forward(self, x, H, W):
		B, N, C = x.shape
		# print(B, N, C, self.dim, x.shape)
		q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

		if self.sr_ratio > 1:
			x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
			x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
			x_ = self.norm(x_)
			kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		else:
			kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		k, v = kv[0], kv[1]

		attn = (q @ k.transpose(-2, -1)) * self.scale
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)

		x = (attn @ v).transpose(1, 2).reshape(B, N, C)
		x = self.proj(x)
		x = self.proj_drop(x)

		return x

class MESA(nn.Module):
	def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
		super().__init__()
		assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

		self.dim = dim
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = qk_scale or head_dim ** -0.5

		self.q = nn.Linear(dim, dim, bias=qkv_bias)
		self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

		self.sr_ratio = sr_ratio
		if sr_ratio > 1:
			self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
			self.norm = nn.LayerNorm(dim)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)
		elif isinstance(m, nn.Conv2d):
			fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			fan_out //= m.groups
			m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
			if m.bias is not None:
				m.bias.data.zero_()

	def forward(self, x, y, H, W):
		x = x.flatten(2).transpose(1, 2)
		B, N, C = x.shape
		q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

		if self.sr_ratio > 1:
			# x_ = y.permute(0, 2, 1).reshape(B, C, H, W)
			x_ = self.sr(y).reshape(B, C, -1).permute(0, 2, 1)
			x_ = self.norm(x_)
			kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		else:
			y = y.flatten(2).transpose(1, 2)
			kv = self.kv(y).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		k, v = kv[0], kv[1]

		attn = (q @ k.transpose(-2, -1)) * self.scale
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)

		x = (attn @ v).transpose(1, 2).reshape(B, N, C)
		x = self.proj(x)
		x = self.proj_drop(x)

		return x

class Block(nn.Module):

	def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
				 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
		super().__init__()
		self.norm1 = norm_layer(dim)
		self.attn = Attention(
			dim,
			num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
			attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
		# NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
		self.norm2 = norm_layer(dim)
		mlp_hidden_dim = int(dim * mlp_ratio)
		self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)
		elif isinstance(m, nn.Conv2d):
			fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			fan_out //= m.groups
			m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
			if m.bias is not None:
				m.bias.data.zero_()

	def forward(self, x, H, W):
		x = x + self.drop_path(self.attn(self.norm1(x), H, W))
		x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

		return x


class OverlapPatchEmbed(nn.Module):
	""" Image to Patch Embedding
	"""

	def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
		super().__init__()
		img_size = to_2tuple(img_size)
		patch_size = to_2tuple(patch_size)

		self.img_size = img_size
		self.patch_size = patch_size
		self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
		self.num_patches = self.H * self.W
		self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
							  padding=(patch_size[0] // 2, patch_size[1] // 2))
		self.norm = nn.LayerNorm(embed_dim)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)
		elif isinstance(m, nn.Conv2d):
			fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			fan_out //= m.groups
			m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
			if m.bias is not None:
				m.bias.data.zero_()

	def forward(self, x):
		x = self.proj(x)
		_, _, H, W = x.shape
		x = x.flatten(2).transpose(1, 2)
		x = self.norm(x)

		return x, H, W

class OverlapPatchEmbedMod(nn.Module):
	""" Modified Image to Patch Embedding by adding convolution features.
	"""

	def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
		super().__init__()
		img_size = to_2tuple(img_size)
		patch_size = to_2tuple(patch_size)
		self.conv = nn.Conv2d(embed_dim, embed_dim, 3, padding = 1)

		self.img_size = img_size
		self.patch_size = patch_size
		self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
		self.num_patches = self.H * self.W
		self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
							  padding=(patch_size[0] // 2, patch_size[1] // 2))
		self.norm = nn.LayerNorm(embed_dim)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)
		elif isinstance(m, nn.Conv2d):
			fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			fan_out //= m.groups
			m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
			if m.bias is not None:
				m.bias.data.zero_()

	def forward(self, x, y):
		x = self.proj(x) + self.conv(y)
		_, _, H, W = x.shape
		x = x.flatten(2).transpose(1, 2)
		x = self.norm(x)

		return x, H, W

class MixVisionTransformer(nn.Module):
	def __init__(self, img_size=512, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
				 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
				 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
				 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
		super().__init__()
		self.num_classes = num_classes
		self.depths = depths

		# patch_embed
		self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
											  embed_dim=embed_dims[0])
		self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
											  embed_dim=embed_dims[1])
		self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
											  embed_dim=embed_dims[2])
		self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
											  embed_dim=embed_dims[3])

		# transformer encoder
		dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
		cur = 0
		self.block1 = nn.ModuleList([Block(
			dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
			drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
			sr_ratio=sr_ratios[0])
			for i in range(depths[0])])
		self.norm1 = norm_layer(embed_dims[0])

		cur += depths[0]
		self.block2 = nn.ModuleList([Block(
			dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
			drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
			sr_ratio=sr_ratios[1])
			for i in range(depths[1])])
		self.norm2 = norm_layer(embed_dims[1])

		cur += depths[1]
		self.block3 = nn.ModuleList([Block(
			dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
			drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
			sr_ratio=sr_ratios[2])
			for i in range(depths[2])])
		self.norm3 = norm_layer(embed_dims[2])

		cur += depths[2]
		self.block4 = nn.ModuleList([Block(
			dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
			drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
			sr_ratio=sr_ratios[3])
			for i in range(depths[3])])
		self.norm4 = norm_layer(embed_dims[3])

		# classification head
		# self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)
		elif isinstance(m, nn.Conv2d):
			fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			fan_out //= m.groups
			m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
			if m.bias is not None:
				m.bias.data.zero_()

	def init_weights(self, pretrained=None):
		if isinstance(pretrained, str):
			logger = get_root_logger()
			load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

	def reset_drop_path(self, drop_path_rate):
		dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
		cur = 0
		for i in range(self.depths[0]):
			self.block1[i].drop_path.drop_prob = dpr[cur + i]

		cur += self.depths[0]
		for i in range(self.depths[1]):
			self.block2[i].drop_path.drop_prob = dpr[cur + i]

		cur += self.depths[1]
		for i in range(self.depths[2]):
			self.block3[i].drop_path.drop_prob = dpr[cur + i]

		cur += self.depths[2]
		for i in range(self.depths[3]):
			self.block4[i].drop_path.drop_prob = dpr[cur + i]

	def freeze_patch_emb(self):
		self.patch_embed1.requires_grad = False

	@torch.jit.ignore
	def no_weight_decay(self):
		return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

	def get_classifier(self):
		return self.head

	def reset_classifier(self, num_classes, global_pool=''):
		self.num_classes = num_classes
		self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

	def forward_features(self, x):
		B = x.shape[0]
		outs = []

		# stage 1
		x, H, W = self.patch_embed1(x)
		for i, blk in enumerate(self.block1):
			x = blk(x, H, W)
		x = self.norm1(x)
		x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
		outs.append(x)

		# stage 2
		x, H, W = self.patch_embed2(x)
		for i, blk in enumerate(self.block2):
			x = blk(x, H, W)
		x = self.norm2(x)
		x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
		outs.append(x)

		# stage 3
		x, H, W = self.patch_embed3(x)
		for i, blk in enumerate(self.block3):
			x = blk(x, H, W)
		x = self.norm3(x)
		x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
		outs.append(x)

		# stage 4
		x, H, W = self.patch_embed4(x)
		for i, blk in enumerate(self.block4):
			x = blk(x, H, W)
		x = self.norm4(x)
		x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
		outs.append(x)

		return outs

	def forward(self, x):
		x = self.forward_features(x)
		# x = self.head(x)

		return x

class MixVisionTransformerMod(nn.Module):
	def __init__(self, img_size=512, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
				 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
				 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
				 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
		super().__init__()
		self.num_classes = num_classes
		self.depths = depths

		# convolution_features
		self.inc = DoubleConv(3, 16)
		self.down1 = Down(16, 32)
		self.down2 = Down(32, 32)
		self.down3 = Down(32, 64)
		self.down4 = Down(64, 160)
		self.down5 = Down(160, 256)

		# patch_embed
		self.patch_embed1 = OverlapPatchEmbedMod(img_size=img_size, patch_size=1, stride=1, in_chans=embed_dims[0],
											  embed_dim=embed_dims[0])
		self.patch_embed2 = OverlapPatchEmbedMod(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
											  embed_dim=embed_dims[1])
		self.patch_embed3 = OverlapPatchEmbedMod(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
											  embed_dim=embed_dims[2])
		self.patch_embed4 = OverlapPatchEmbedMod(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
											  embed_dim=embed_dims[3])

		# transformer encoder
		dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
		cur = 0
		self.block1 = nn.ModuleList([Block(
			dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
			drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
			sr_ratio=sr_ratios[0])
			for i in range(depths[0])])
		self.norm1 = norm_layer(embed_dims[0])

		cur += depths[0]
		self.block2 = nn.ModuleList([Block(
			dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
			drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
			sr_ratio=sr_ratios[1])
			for i in range(depths[1])])
		self.norm2 = norm_layer(embed_dims[1])

		cur += depths[1]
		self.block3 = nn.ModuleList([Block(
			dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
			drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
			sr_ratio=sr_ratios[2])
			for i in range(depths[2])])
		self.norm3 = norm_layer(embed_dims[2])

		cur += depths[2]
		self.block4 = nn.ModuleList([Block(
			dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
			drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
			sr_ratio=sr_ratios[3])
			for i in range(depths[3])])
		self.norm4 = norm_layer(embed_dims[3])

		# classification head
		# self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)
		elif isinstance(m, nn.Conv2d):
			fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			fan_out //= m.groups
			m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
			if m.bias is not None:
				m.bias.data.zero_()

	def init_weights(self, pretrained=None):
		if isinstance(pretrained, str):
			logger = get_root_logger()
			load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

	def reset_drop_path(self, drop_path_rate):
		dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
		cur = 0
		for i in range(self.depths[0]):
			self.block1[i].drop_path.drop_prob = dpr[cur + i]

		cur += self.depths[0]
		for i in range(self.depths[1]):
			self.block2[i].drop_path.drop_prob = dpr[cur + i]

		cur += self.depths[1]
		for i in range(self.depths[2]):
			self.block3[i].drop_path.drop_prob = dpr[cur + i]

		cur += self.depths[2]
		for i in range(self.depths[3]):
			self.block4[i].drop_path.drop_prob = dpr[cur + i]

	def freeze_patch_emb(self):
		self.patch_embed1.requires_grad = False

	@torch.jit.ignore
	def no_weight_decay(self):
		return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

	def get_classifier(self):
		return self.head

	def reset_classifier(self, num_classes, global_pool=''):
		self.num_classes = num_classes
		self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

	def forward_features(self, x):

		conv_features_seg = []

		res = self.inc(x)
		conv_features_seg.append(res)
		res1 = self.down1(res)
		conv_features_seg.append(res1)
		res2 = self.down2(res1)
		conv_features_seg.append(res2)
		res3 = self.down3(res2)
		conv_features_seg.append(res3)
		res4 = self.down4(res3)
		conv_features_seg.append(res4)
		res5 = self.down5(res4)
		conv_features_seg.append(res5)

		B = x.shape[0]
		outs = []

		# stage 1
		x, H, W = self.patch_embed1(res2, res2)
		for i, blk in enumerate(self.block1):
			x = blk(x, H, W)
		x = self.norm1(x)
		x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
		outs.append(x)

		# stage 2
		x, H, W = self.patch_embed2(x, res3)
		for i, blk in enumerate(self.block2):
			x = blk(x, H, W)
		x = self.norm2(x)
		x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
		outs.append(x)

		# stage 3
		x, H, W = self.patch_embed3(x, res4)
		for i, blk in enumerate(self.block3):
			x = blk(x, H, W)
		x = self.norm3(x)
		x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
		outs.append(x)

		# stage 4
		x, H, W = self.patch_embed4(x, res5)
		for i, blk in enumerate(self.block4):
			x = blk(x, H, W)
		x = self.norm4(x)
		x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
		outs.append(x)

		return outs, conv_features_seg

	def forward(self, x):
		x, y = self.forward_features(x)
		# x = self.head(x)

		return x, y

class DWConv(nn.Module):
	def __init__(self, dim=768):
		super(DWConv, self).__init__()
		self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

	def forward(self, x, H, W):
		B, N, C = x.shape
		x = x.transpose(1, 2).view(B, C, H, W)
		x = self.dwconv(x)
		x = x.flatten(2).transpose(1, 2)

		return x

import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict

from mmseg.ops import resize
# from ..builder import HEADS
# from .decode_head import BaseDecodeHead
from mmseg.models.utils import *
import attr

from IPython import embed

class MLP(nn.Module):
	"""
	Linear Embedding
	"""
	def __init__(self, input_dim=2048, embed_dim=768):
		super().__init__()
		self.proj = nn.Linear(input_dim, embed_dim)

	def forward(self, x):
		x = x.flatten(2).transpose(1, 2)
		x = self.proj(x)
		return x


# @HEADS.register_module()
class SegFormerHead(nn.Module):
	"""
	SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
	"""
	def __init__(self):
		super(SegFormerHead, self).__init__()
		# assert len(feature_strides) == len(self.in_channels)
		# assert min(feature_strides) == feature_strides[0]
		# self.feature_strides = feature_strides

		c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = 32, 64, 160, 256

		# decoder_params = kwargs['decoder_params']
		embedding_dim = 256

		self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
		self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
		self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
		self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

		self.linear_fuse = ConvModule(
			in_channels=embedding_dim*4,
			out_channels=embedding_dim,
			kernel_size=1,
			norm_cfg=dict(type='BN', requires_grad=True)
		)

		self.linear_pred = nn.Conv2d(embedding_dim, 1, kernel_size=1)
		self.dropout = nn.Dropout2d(0.1)

	def forward(self, x):
		# x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
		# print([i.shape for i in x[::-1]])
		c1, c2, c3, c4 = x[::-1]

		############## MLP decoder on C1-C4 ###########
		n, _, h, w = c4.shape

		_c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
		_c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

		_c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
		_c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

		_c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
		_c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

		_c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

		_c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

		x = self.dropout(_c)
		x = self.linear_pred(x)

		return x

class SegFormerHeadMod(nn.Module):
	"""
	Modified SegFormer with the U-Net style upsampling.
	"""
	def __init__(self, deep_supervision = False):
		super(SegFormerHeadMod, self).__init__()
		dims = [32, 64, 160, 256]
		c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = 32, 64, 160, 256
		reduction_ratio = [8, 4, 2, 1]
		heads = [1, 2, 5, 8]

		self.upsampling = nn.ModuleList([
			Up(dims[-1] + dims[-2], dims[-2]),
			Up(dims[-2] + dims[-3], dims[-3]),
			Up(dims[-3] + dims[-4], dims[-4]),
			Up(96 + 32, 64),
			Up(64 + 16, 32)
		])
		embedding_dim = decoder_dim = 512
		num_classes = 1

		self.deep_supervision = deep_supervision

		if deep_supervision:

			self.to_segmentation = nn.Sequential(
				nn.Conv2d(dims[-4], decoder_dim, 1),
				nn.Conv2d(decoder_dim, num_classes, 1),
				nn.Upsample(scale_factor = 2)
			)
			self.to_segmentation1 = nn.Sequential(
				nn.Conv2d(dims[-3], decoder_dim, 1),
				nn.Conv2d(decoder_dim, num_classes, 1),
				nn.Upsample(scale_factor = 4)
			)
			self.to_segmentation2 = nn.Sequential(
				nn.Conv2d(dims[-2], decoder_dim, 1),
				nn.Conv2d(decoder_dim, num_classes, 1),
				nn.Upsample(scale_factor = 8)
			)

			self.outd = nn.Conv2d(64, 1, 1)

		self.outc = nn.Conv2d(32, 1, 1)
		self.segf2c = nn.Conv2d(dims[-4], 96, 1)

		self.convsegattn1 = MESA(dim = dims[-1], qkv_bias = True, num_heads = heads[-1], sr_ratio = reduction_ratio[-1])
		self.convsegattn2 = MESA(dim = dims[-2], qkv_bias = True, num_heads = heads[-2], sr_ratio = reduction_ratio[-2])
		self.convsegattn3 = MESA(dim = dims[-3], qkv_bias = True, num_heads = heads[-3], sr_ratio = reduction_ratio[-3])
		self.convsegattn4 = MESA(dim = dims[-4], qkv_bias = True, num_heads = heads[-4], sr_ratio = reduction_ratio[-4])

	def forward(self, layer_outputs, conv_features):
		res5, res4, res3, res2, res1, res = conv_features[::-1]

		# return x
		B, C, H, W = layer_outputs[-1].shape
		conv_attn1 = self.convsegattn1(layer_outputs[-1], res5, H, W)
		conv_attn1 = conv_attn1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
		layer_outputs[-1] = layer_outputs[-1] + conv_attn1

		B, C, H, W = layer_outputs[-2].shape
		conv_attn2 = self.convsegattn2(layer_outputs[-2], res4, H, W)
		conv_attn2 = conv_attn2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
		layer_outputs[-2] = layer_outputs[-2] + conv_attn2

		B, C, H, W = layer_outputs[-3].shape
		conv_attn3 = self.convsegattn3(layer_outputs[-3], res3, H, W)
		conv_attn3 = conv_attn3.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
		layer_outputs[-3] = layer_outputs[-3] + conv_attn3

		B, C, H, W = layer_outputs[-4].shape
		conv_attn4 = self.convsegattn4(layer_outputs[-4], res2, H, W)
		conv_attn4 = conv_attn4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
		layer_outputs[-4] = layer_outputs[-4] + conv_attn4

		outputs = []

		x = self.upsampling[0](layer_outputs[-1], layer_outputs[-2])
		if self.deep_supervision:
			outputs.append(self.to_segmentation2(x))
		
		x = self.upsampling[1](x, layer_outputs[-3])
		if self.deep_supervision:
			outputs.append(self.to_segmentation1(x))
		
		x = self.upsampling[2](x, layer_outputs[-4])
		if self.deep_supervision:
			outputs.append(self.to_segmentation(x))

		x = self.upsampling[3](self.segf2c(x), res1)
		if self.deep_supervision:
			outputs.append(self.outd(x))

		x = self.upsampling[4](x, res)
		outputs.append(self.outc(x))

		return outputs

# @BACKBONES.register_module()
class mit_b0(MixVisionTransformer):
	def __init__(self, **kwargs):
		super(mit_b0, self).__init__(
			patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
			qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
			drop_rate=0.0, drop_path_rate=0.1)

class mit_b0mod(MixVisionTransformerMod):
	def __init__(self, **kwargs):
		super(mit_b0mod, self).__init__(
			patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
			qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
			drop_rate=0.0, drop_path_rate=0.1)

class SegFormerB0(nn.Module):
	def __init__(self):
		super(SegFormerB0, self).__init__()
		self.encoder = mit_b0()
		self.decoder = SegFormerHead()

	def forward(self, x):
		feat1, feat2, feat3, feat4 = self.encoder(x)
		return self.decoder([feat4, feat3, feat2, feat1])

class ConvSegFormer(nn.Module):
	def __init__(self, deep_supervision = False):
		super(ConvSegFormer, self).__init__()
		self.encoder = mit_b0mod()
		self.decoder = SegFormerHeadMod(deep_supervision = deep_supervision)

	def forward(self, x):
		transf_features, conv_features = self.encoder(x)
		return self.decoder(transf_features, conv_features)

# model = ConvSegFormer(deep_supervision = False)
# device = torch.device("cuda:0")
# model.to(device)
# from torchinfo import summary
# summary(model, input_size=(1, 3, 512, 512))

## Changing the kernel_size parameter in classes in line 69 and 82 to '1' will give
## parameters and GMACs for the smaller version.
