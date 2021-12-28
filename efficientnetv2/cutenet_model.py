#All most all the code here have been taken from Automl and efficientnet open source implementations in github

import copy
import itertools
import math
import os

from absl import logging
import numpy as np
import tensorflow as tf

import effnetv2_configs
import hparams
import utils

from tensorflow.keras.layers import (
    Dropout,
    Softmax,
    LayerNormalization,
    Conv2D,
    Layer,
    Dense,
    Activation
)
from tensorflow import keras
from tensorflow.keras import Model, Sequential
import tensorflow_addons as tfa
import collections


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


class DropPath(Layer):
    def __init__(self, prob):
        super().__init__()
        self.drop_prob = prob

    def call(self, x, training=None):
        if self.drop_prob == 0. or not training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = tf.random.uniform(shape=shape)
        random_tensor = tf.where(random_tensor < keep_prob, 1, 0)
        output = x / keep_prob * random_tensor
        return output


class TruncatedDense(Dense):
    def __init__(self, units, use_bias=True, initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=.02)):
        super().__init__(units, use_bias=use_bias, kernel_initializer=initializer)


class Mlp(Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=Activation(tf.nn.gelu), drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = TruncatedDense(hidden_features)
        self.act = act_layer
        self.fc2 = TruncatedDense(out_features)
        self.drop = Dropout(drop)

    def call(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
        

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = tf.reshape(x, [B, H // window_size, window_size, W // window_size, window_size, C]) 
    # TODO contiguous memory access?
    windows = tf.reshape(tf.transpose(x, perm=[0, 1, 3, 2, 4, 5]), [-1, window_size, window_size, C])
    return windows


@tf.function
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = tf.reshape(windows, [B, H // window_size, W // window_size, window_size, window_size, -1])
    x = tf.reshape(tf.transpose(x, perm=[0, 1, 3, 2, 4, 5]), [B, H, W, -1])
    return x


class WindowAttention(Layer):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=.02)
        self.relative_position_bias_table = tf.Variable(
            initializer(shape=((2*self.window_size[0]-1) * (2*self.window_size[1]-1), num_heads)), name="relative_position_bias_table")  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = tf.range(self.window_size[0])
        coords_w = tf.range(self.window_size[1])
        coords = tf.stack(tf.meshgrid(coords_h, coords_w))  # 2, Wh, Ww
        coords_flatten = tf.reshape(coords, [2, -1])  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = tf.transpose(relative_coords, perm=[1,2,0]) # Wh*Ww, Wh*Ww, 2
        relative_coords = relative_coords + [self.window_size[0] - 1, self.window_size[1] - 1]  # shift to start from 0
        relative_coords = relative_coords * [2*self.window_size[1] - 1, 1]
        self.relative_position_index = tf.math.reduce_sum(relative_coords,-1)  # Wh*Ww, Wh*Ww

        self.qkv = TruncatedDense(dim * 3, use_bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj = TruncatedDense(dim)
        self.proj_drop = Dropout(proj_drop)
        self.softmax = Softmax(axis=-1)

    def call(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = tf.transpose(tf.reshape(self.qkv(x), [B_, N, 3, self.num_heads, C // self.num_heads]), perm=[2, 0, 3, 1, 4]) # [3, B_, num_head, Ww*Wh, C//num_head]
        q, k, v = tf.unstack(qkv)  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = tf.einsum('...ij,...kj->...ik', q, k)

        relative_position_bias = tf.reshape(tf.gather(self.relative_position_bias_table, tf.reshape(self.relative_position_index, -1)),
            [self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1])  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = tf.transpose(relative_position_bias, perm=[2, 0, 1])  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias

        if mask is not None:
            nW = mask.shape[0] # every window has different mask [nW, N, N]
            attn = tf.reshape(attn, [B_ // nW, nW, self.num_heads, N, N]) + mask[:, None, :, :] # add mask: make each component -inf or just leave it
            attn = tf.reshape(attn, [-1, self.num_heads, N, N])
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = tf.reshape(tf.transpose(attn @ v, perm=[0, 2, 1, 3]), [B_, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(Layer):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (tf.keras.layers.Activation, optional): Activation layer. Default: tf.keras.layers.Activation(tf.nn.gelu)
        norm_layer (tf.keras.layers.Layer, optional): Normalization layer.  Default: tf.keras.layers.LayerNormalization
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=Activation(tf.nn.gelu), norm_layer=LayerNormalization):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(epsilon=1e-5)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else tf.identity
        self.norm2 = norm_layer(epsilon=1e-5)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = np.zeros([1, H, W, 1])  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            img_mask = tf.constant(img_mask)
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = tf.reshape(mask_windows, [-1, self.window_size * self.window_size])
            attn_mask = mask_windows[:, None, :] - mask_windows[:, :, None]
            self.attn_mask = tf.where(attn_mask==0, -100., 0.)
        else:
            self.attn_mask = None

    def call(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = tf.reshape(x, [B, H, W, C])

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = tf.reshape(x_windows, [-1, self.window_size * self.window_size, C])  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = tf.reshape(attn_windows, [-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=[self.shift_size, self.shift_size], axis=(1, 2))
        else:
            x = shifted_x
        x = tf.reshape(x, [B, H * W, C])

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(Layer):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (tf.keras.layers.Layer, optional): Normalization layer.  Default: tf.keras.layers.LayerNormalization
    """

    def __init__(self, input_resolution, dim, norm_layer=LayerNormalization):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = TruncatedDense(2 * dim, use_bias=False)
        self.norm = norm_layer(epsilon=1e-5)

    def call(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = tf.reshape(x, [B, H, W, C])

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = tf.concat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = tf.reshape(x, [B, -1, 4 * C])  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim # merging
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim # reduction
        return flops


class BasicLayer(Layer):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (tf.keras.layers.Layer, optional): Normalization layer. Default: tf.keras.layers.LayerNormalization
        downsample (tf.keras.layers.Layer | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=LayerNormalization, downsample=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = [
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)]

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def call(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(Layer):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (tf.keras.layers.Layer, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = Conv2D(filters=embed_dim, kernel_size=patch_size, strides=patch_size, data_format="channels_first")
        if norm_layer is not None:
            # TODO: Check impact of epsilon
            self.norm = norm_layer(epsilon=1e-5)
        else:
            self.norm = None

    def call(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = tf.transpose(tf.reshape(self.proj(x), [B, self.embed_dim, -1]), perm=[0, 2, 1])
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class ReversedPatchEmbed(Layer):
  def __init__(self,input, patch_size=4):
    super().__init__()
    self.patch_size=4
    self.trans_conv2d = layers.Conv2DTranspose(3, self.patch_size, self.patch_size)
  def call(self,input):
    B, L, C = input.shape
    x = tf.reshape(input, [B, np.sqrt(L), np.sqrt(L), C])
    x = self.trans_conv2d(x)
    return x




#-----------------------------------------------------EFFICIENTNET_PART---------------------------------------------------------------------------------
      
def conv_kernel_initializer(shape, dtype=None, partition_info=None):
  """Initialization for convolutional kernels.

  The main difference with tf.variance_scaling_initializer is that
  tf.variance_scaling_initializer uses a truncated normal with an uncorrected
  standard deviation, whereas here we use a normal distribution. Similarly,
  tf.initializers.variance_scaling uses a truncated normal with
  a corrected standard deviation.

  Args:
    shape: shape of variable
    dtype: dtype of variable
    partition_info: unused

  Returns:
    an initialization for the variable
  """
  del partition_info
  kernel_height, kernel_width, _, out_filters = shape
  fan_out = int(kernel_height * kernel_width * out_filters)
  return tf.random.normal(
      shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


def dense_kernel_initializer(shape, dtype=None, partition_info=None):
  """Initialization for dense kernels.
  This initialization is equal to
    tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                    distribution='uniform').
  It is written out explicitly here for clarity.
  Args:
    shape: shape of variable
    dtype: dtype of variable
    partition_info: unused
  Returns:
    an initialization for the variable
  """
  del partition_info
  init_range = 1.0 / np.sqrt(shape[1])
  return tf.random.uniform(shape, -init_range, init_range, dtype=dtype)


def round_filters(filters, mconfig, skip=False):
  """Round number of filters based on depth multiplier."""
  multiplier = mconfig.width_coefficient
  divisor = mconfig.depth_divisor
  min_depth = mconfig.min_depth
  if skip or not multiplier:
    return filters

  filters *= multiplier
  min_depth = min_depth or divisor
  new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
  return int(new_filters)


def round_repeats(repeats, multiplier, skip=False):
  """Round number of filters based on depth multiplier."""
  if skip or not multiplier:
    return repeats
  return int(math.ceil(multiplier * repeats))


class SE(tf.keras.layers.Layer):
  """Squeeze-and-excitation layer."""

  def __init__(self, mconfig, se_filters, output_filters, name=None):
    super().__init__(name=name)

    self._local_pooling = mconfig.local_pooling
    self._data_format = mconfig.data_format
    self._act = utils.get_act_fn(mconfig.act_fn)

    # Squeeze and Excitation layer.
    self._se_reduce = tf.keras.layers.Conv2D(
        se_filters,
        kernel_size=1,
        strides=1,
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        data_format=self._data_format,
        use_bias=True,
        name='conv2d')
    self._se_expand = tf.keras.layers.Conv2D(
        output_filters,
        kernel_size=1,
        strides=1,
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        data_format=self._data_format,
        use_bias=True,
        name='conv2d_1')

  def call(self, inputs):
    h_axis, w_axis = [2, 3] if self._data_format == 'channels_first' else [1, 2]
    if self._local_pooling:
      se_tensor = tf.nn.avg_pool(
          inputs,
          ksize=[1, inputs.shape[h_axis], inputs.shape[w_axis], 1],
          strides=[1, 1, 1, 1],
          padding='VALID')
    else:
      se_tensor = tf.reduce_mean(inputs, [h_axis, w_axis], keepdims=True)
    se_tensor = self._se_expand(self._act(self._se_reduce(se_tensor)))
    logging.info('Built SE %s : %s', self.name, se_tensor.shape)
    return tf.sigmoid(se_tensor) * inputs


class MBConvBlock(tf.keras.layers.Layer):
  """A class of MBConv: Mobile Inverted Residual Bottleneck.
  Attributes:
    endpoints: dict. A list of internal tensors.
  """

  def __init__(self, block_args, mconfig, name=None):
    """Initializes a MBConv block.
    Args:
      block_args: BlockArgs, arguments to create a Block.
      mconfig: GlobalParams, a set of global parameters.
      name: layer name.
    """
    super().__init__(name=name)

    self._block_args = copy.deepcopy(block_args)
    self._mconfig = copy.deepcopy(mconfig)
    self._local_pooling = mconfig.local_pooling
    self._data_format = mconfig.data_format
    self._channel_axis = 1 if self._data_format == 'channels_first' else -1

    self._act = utils.get_act_fn(mconfig.act_fn)
    self._has_se = (
        self._block_args.se_ratio is not None and
        0 < self._block_args.se_ratio <= 1)

    self.endpoints = None

    # Builds the block accordings to arguments.
    self._build()

  @property
  def block_args(self):
    return self._block_args

  def _build(self):
    """Builds block according to the arguments."""
    # pylint: disable=g-long-lambda
    bid = itertools.count(0)
    get_norm_name = lambda: 'tpu_batch_normalization' + ('' if not next(
        bid) else '_' + str(next(bid) // 2))
    cid = itertools.count(0)
    get_conv_name = lambda: 'conv2d' + ('' if not next(cid) else '_' + str(
        next(cid) // 2))
    # pylint: enable=g-long-lambda

    mconfig = self._mconfig
    filters = self._block_args.input_filters * self._block_args.expand_ratio
    kernel_size = self._block_args.kernel_size

    # Expansion phase. Called if not using fused convolutions and expansion
    # phase is necessary.
    if self._block_args.expand_ratio != 1:
      self._expand_conv = tf.keras.layers.Conv2D(
          filters=filters,
          kernel_size=1,
          strides=1,
          kernel_initializer=conv_kernel_initializer,
          padding='same',
          data_format=self._data_format,
          use_bias=False,
          name=get_conv_name())
      self._norm0 = utils.normalization(
          mconfig.bn_type,
          axis=self._channel_axis,
          momentum=mconfig.bn_momentum,
          epsilon=mconfig.bn_epsilon,
          groups=mconfig.gn_groups,
          name=get_norm_name())

    # Depth-wise convolution phase. Called if not using fused convolutions.
    self._depthwise_conv = tf.keras.layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=self._block_args.strides,
        depthwise_initializer=conv_kernel_initializer,
        padding='same',
        data_format=self._data_format,
        use_bias=False,
        name='depthwise_conv2d')

    self._norm1 = utils.normalization(
        mconfig.bn_type,
        axis=self._channel_axis,
        momentum=mconfig.bn_momentum,
        epsilon=mconfig.bn_epsilon,
        groups=mconfig.gn_groups,
        name=get_norm_name())

    if self._has_se:
      num_reduced_filters = max(
          1, int(self._block_args.input_filters * self._block_args.se_ratio))
      self._se = SE(self._mconfig, num_reduced_filters, filters, name='se')
    else:
      self._se = None

    # Output phase.
    filters = self._block_args.output_filters
    self._project_conv = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=1,
        strides=1,
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        data_format=self._data_format,
        use_bias=False,
        name=get_conv_name())
    self._norm2 = utils.normalization(
        mconfig.bn_type,
        axis=self._channel_axis,
        momentum=mconfig.bn_momentum,
        epsilon=mconfig.bn_epsilon,
        groups=mconfig.gn_groups,
        name=get_norm_name())

  def residual(self, inputs, x, training, survival_prob):
    if (self._block_args.strides == 1 and
        self._block_args.input_filters == self._block_args.output_filters):
      # Apply only if skip connection presents.
      if survival_prob:
        x = utils.drop_connect(x, training, survival_prob)
      x = tf.add(x, inputs)

    return x

  def call(self, inputs, training, survival_prob=None):
    """Implementation of call().
    Args:
      inputs: the inputs tensor.
      training: boolean, whether the model is constructed for training.
      survival_prob: float, between 0 to 1, drop connect rate.
    Returns:
      A output tensor.
    """
    logging.info('Block %s input shape: %s (%s)', self.name, inputs.shape,
                 inputs.dtype)
    x = inputs
    if self._block_args.expand_ratio != 1:
      x = self._act(self._norm0(self._expand_conv(x), training=training))
      logging.info('Expand shape: %s', x.shape)

    x = self._act(self._norm1(self._depthwise_conv(x), training=training))
    logging.info('DWConv shape: %s', x.shape)

    if self._mconfig.conv_dropout and self._block_args.expand_ratio > 1:
      x = tf.keras.layers.Dropout(self._mconfig.conv_dropout)(
          x, training=training)

    if self._se:
      x = self._se(x)

    self.endpoints = {'expansion_output': x}

    x = self._norm2(self._project_conv(x), training=training)
    x = self.residual(inputs, x, training, survival_prob)

    logging.info('Project shape: %s', x.shape)
    return x


class FusedMBConvBlock(MBConvBlock):
  """Fusing the proj conv1x1 and depthwise_conv into a conv2d."""

  def _build(self):
    """Builds block according to the arguments."""
    # pylint: disable=g-long-lambda
    bid = itertools.count(0)
    get_norm_name = lambda: 'tpu_batch_normalization' + ('' if not next(
        bid) else '_' + str(next(bid) // 2))
    cid = itertools.count(0)
    get_conv_name = lambda: 'conv2d' + ('' if not next(cid) else '_' + str(
        next(cid) // 2))
    # pylint: enable=g-long-lambda

    mconfig = self._mconfig
    block_args = self._block_args
    filters = block_args.input_filters * block_args.expand_ratio
    kernel_size = block_args.kernel_size
    if block_args.expand_ratio != 1:
      # Expansion phase:
      self._expand_conv = tf.keras.layers.Conv2D(
          filters,
          kernel_size=kernel_size,
          strides=block_args.strides,
          kernel_initializer=conv_kernel_initializer,
          padding='same',
          use_bias=False,
          name=get_conv_name())
      self._norm0 = utils.normalization(
          mconfig.bn_type,
          axis=self._channel_axis,
          momentum=mconfig.bn_momentum,
          epsilon=mconfig.bn_epsilon,
          groups=mconfig.gn_groups,
          name=get_norm_name())

    if self._has_se:
      num_reduced_filters = max(
          1, int(block_args.input_filters * block_args.se_ratio))
      self._se = SE(mconfig, num_reduced_filters, filters, name='se')
    else:
      self._se = None
    # Output phase:
    filters = block_args.output_filters
    self._project_conv = tf.keras.layers.Conv2D(
        filters,
        kernel_size=1 if block_args.expand_ratio != 1 else kernel_size,
        strides=1 if block_args.expand_ratio != 1 else block_args.strides,
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False,
        name=get_conv_name())
    self._norm1 = utils.normalization(
        mconfig.bn_type,
        axis=self._channel_axis,
        momentum=mconfig.bn_momentum,
        epsilon=mconfig.bn_epsilon,
        groups=mconfig.gn_groups,
        name=get_norm_name())

  def call(self, inputs, training, survival_prob=None):
    """Implementation of call().
    Args:
      inputs: the inputs tensor.
      training: boolean, whether the model is constructed for training.
      survival_prob: float, between 0 to 1, drop connect rate.
    Returns:
      A output tensor.
    """
    logging.info('Block %s  input shape: %s', self.name, inputs.shape)
    x = inputs
    if self._block_args.expand_ratio != 1:
      x = self._act(self._norm0(self._expand_conv(x), training=training))
    logging.info('Expand shape: %s', x.shape)

    self.endpoints = {'expansion_output': x}

    if self._mconfig.conv_dropout and self._block_args.expand_ratio > 1:
      x = tf.keras.layers.Dropout(self._mconfig.conv_dropout)(x, training)

    if self._se:
      x = self._se(x)

    x = self._norm1(self._project_conv(x), training=training)
    if self._block_args.expand_ratio == 1:
      x = self._act(x)  # add act if no expansion.

    x = self.residual(inputs, x, training, survival_prob)
    logging.info('Project shape: %s', x.shape)
    return x


class Stem(tf.keras.layers.Layer):
  """Stem layer at the begining of the network."""

  def __init__(self, mconfig, stem_filters, name=None):
    super().__init__(name=name)
    self._conv_stem = tf.keras.layers.Conv2D(
        filters=round_filters(stem_filters, mconfig),
        kernel_size=3,
        strides=2,
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        data_format=mconfig.data_format,
        use_bias=False,
        name='conv2d')
    self._norm = utils.normalization(
        mconfig.bn_type,
        axis=(1 if mconfig.data_format == 'channels_first' else -1),
        momentum=mconfig.bn_momentum,
        epsilon=mconfig.bn_epsilon,
        groups=mconfig.gn_groups)
    self._act = utils.get_act_fn(mconfig.act_fn)

  def call(self, inputs, training):
    return self._act(self._norm(self._conv_stem(inputs), training=training))


class Head(tf.keras.layers.Layer):
  """Head layer for network outputs."""

  def __init__(self, mconfig, name=None):
    super().__init__(name=name)

    self.endpoints = {}
    self._mconfig = mconfig

    self._conv_head = tf.keras.layers.Conv2D(
        filters=round_filters(mconfig.feature_size or 1280, mconfig),
        kernel_size=1,
        strides=1,
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        data_format=mconfig.data_format,
        use_bias=False,
        name='conv2d')
    self._norm = utils.normalization(
        mconfig.bn_type,
        axis=(1 if mconfig.data_format == 'channels_first' else -1),
        momentum=mconfig.bn_momentum,
        epsilon=mconfig.bn_epsilon,
        groups=mconfig.gn_groups)
    self._act = utils.get_act_fn(mconfig.act_fn)

    self._avg_pooling = tf.keras.layers.GlobalAveragePooling2D(
        data_format=mconfig.data_format)

    if mconfig.dropout_rate > 0:
      self._dropout = tf.keras.layers.Dropout(mconfig.dropout_rate)
    else:
      self._dropout = None

    self.h_axis, self.w_axis = ([2, 3] if mconfig.data_format
                                == 'channels_first' else [1, 2])

  def call(self, inputs, training):
    """Call the layer."""
    outputs = self._act(self._norm(self._conv_head(inputs), training=training))
    self.endpoints['head_1x1'] = outputs

    if self._mconfig.local_pooling:
      shape = outputs.get_shape().as_list()
      kernel_size = [1, shape[self.h_axis], shape[self.w_axis], 1]
      outputs = tf.nn.avg_pool(
          outputs, ksize=kernel_size, strides=[1, 1, 1, 1], padding='VALID')
      self.endpoints['pooled_features'] = outputs
      if self._dropout:
        outputs = self._dropout(outputs, training=training)
      self.endpoints['global_pool'] = outputs
      if self._fc:
        outputs = tf.squeeze(outputs, [self.h_axis, self.w_axis])
        outputs = self._fc(outputs)
      self.endpoints['head'] = outputs
    else:
      outputs = self._avg_pooling(outputs)
      self.endpoints['pooled_features'] = outputs
      if self._dropout:
        outputs = self._dropout(outputs, training=training)
      self.endpoints['head'] = outputs
    return outputs

 
class MyDense(tf.keras.layers.Dense):
  def __init__(self,cnn=True):
    super().__init__()
    self.cnn = cnn
  def call(self, input):
    if cnn:
      B, W, H, C = tf.shape(input)
    else:
      B, L, C = tf.shape(input)
    return self.call(C/2)


  
  
  
#------------------------------------------------------THE_MODEL----------------------------------------------------------------------  
  

  
class CuteNetModel(tf.keras.Model):
  """A class implements tf.keras.Model.

    Reference: https://arxiv.org/abs/1807.11626
  """
  def __init__(self,
               effnet_model_name='efficientnetv2-s',
               effnet_model_config=None,
               name='cutenet',
              
               img_size=384, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                 window_size=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=LayerNormalization, ape=False, patch_norm=True,
                 **kwargs):
    """Initializes an `Model` instance.

    Args:
      model_name: A string of model name for efficinetnet model.
      model_config: A dict of efficientnet model configurations or a string of hparams.
      name: A string of layer name.

    Raises:
      ValueError: when blocks_args is not specified as a list.
    """
    super().__init__(name=name)
    cfg = copy.deepcopy(hparams.base_config)
    if effnet_model_name:
      cfg.override(effnetv2_configs.get_model_config(effnet_model_name))
    cfg.model.override(effnet_model_config)
    self.cfg = cfg
    self._mconfig = cfg.model
    self.endpoints = None
    self.in_chans = in_chans
    self.img_size = img_size
    self.patch_size = patch_size
    self.num_classes = num_classes
    self.num_layers = len(depths)
    self.embed_dim = embed_dim
    self.ape = ape
    self.patch_norm = patch_norm
    self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
    self.mlp_ratio = mlp_ratio  
    self.qkv_bias = qkv_bias
    self.qk_scale = qk_scale
    self.drop_rate = drop_rate
    self.attn_drop_rate = attn_drop_rate 
    self.drop_path_rate = drop_path_rate
    self.norm_layer = LayerNormalization 
    self.ape = ape
    self.patch_norm = patch_norm
    self.depths = depths
    self.num_heads = num_heads
    self.window_size = window_size
    self._build()

  def _build(self):
    
    

    # split image into non-overlapping patches
    self.patch_embed = PatchEmbed(
            img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim,
            norm_layer=self.norm_layer if self.patch_norm else None)
    num_patches = self.patch_embed.num_patches
    patches_resolution = self.patch_embed.patches_resolution
    self.patches_resolution = patches_resolution
    self.patch_embed_1 = PatchEmbed(
            img_size=192, patch_size=4, in_chans=self.in_chans, embed_dim=256,
            norm_layer=self.norm_layer if self.patch_norm else None)
    self.patch_embed_2 = PatchEmbed(
            img_size=96, patch_size=4, in_chans=self.in_chans, embed_dim=512,
            norm_layer=self.norm_layer if self.patch_norm else None)
    self.patch_embed_3 = PatchEmbed(
            img_size=48, patch_size=4, in_chans=self.in_chans, embed_dim=1024,
            norm_layer=self.norm_layer if self.patch_norm else None)
    self.patch_embed_4 = PatchEmbed(
            img_size=24, patch_size=4,in_chans=self.in_chans, embed_dim=1024,
            norm_layer=self.norm_layer if self.patch_norm else None)
    self.embeder = [self.patch_embed_1,self.patch_embed_2,self.patch_embed_3,self.patch_embed_4]

    # absolute position embedding
    if self.ape:
        initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=.02)
        # TODO: Check to make sure that this variable is supposed to not be trainable
        self.absolute_pos_embed = tf.Variable(initializer(shape = (1, num_patches, self.embed_dim)), trainable=False)

    self.pos_drop = tf.keras.layers.Dropout(rate=self.drop_rate)

        # stochastic depth
    dpr = [x for x in np.linspace(0, self.drop_path_rate, sum(self.depths))]  # stochastic depth decay rule

    # build layers
    self.blocks = [tf.keras.models.Sequential(BasicLayer(dim=int(self.embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=self.depths[i_layer],
                               num_heads=self.num_heads[i_layer],
                               window_size=self.window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                               drop=self.drop_rate, attn_drop=self.attn_drop_rate,
                               drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                               norm_layer=self.norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)) for i_layer in range(self.num_layers)]
    

        # TODO: Check impact of epsilon
    self.norm = self.norm_layer(epsilon=1e-5)
    self.avgpool = tfa.layers.AdaptiveAveragePooling1D(1) 
    self.swin_input = tf.keras.layers.Conv2D(3,1)
    
    
    """Builds a model."""
    self._blocks = []

    # Stem part.
    self._stem = Stem(self._mconfig, self._mconfig.blocks_args[0].input_filters)

    # Builds blocks.
    block_id = itertools.count(0)
    block_name = lambda: 'blocks_%d' % next(block_id)
    for block_args in self._mconfig.blocks_args:
      assert block_args.num_repeat > 0
      # Update block input and output filters based on depth multiplier.
      input_filters = round_filters(block_args.input_filters, self._mconfig)
      output_filters = round_filters(block_args.output_filters, self._mconfig)
      repeats = round_repeats(block_args.num_repeat,
                              self._mconfig.depth_coefficient)
      block_args.update(
          dict(
              input_filters=input_filters,
              output_filters=output_filters,
              num_repeat=repeats))

      # The first block needs to take care of stride and filter size increase.
      conv_block = {0: MBConvBlock, 1: FusedMBConvBlock}[block_args.conv_type]
      self._blocks.append(
          conv_block(block_args, self._mconfig, name=block_name()))
      if block_args.num_repeat > 1:  # rest of blocks with the same block_arg
        # pylint: disable=protected-access
        block_args.input_filters = block_args.output_filters
        block_args.strides = 1
        # pylint: enable=protected-access
      for _ in range(block_args.num_repeat - 1):
        self._blocks.append(
            conv_block(block_args, self._mconfig, name=block_name()))

    # Head part.
    self._head = Head(self._mconfig)


  def summary(self, input_shape=(224, 224, 3), **kargs):
    x = tf.keras.Input(shape=input_shape)
    model = tf.keras.Model(inputs=[x], outputs=self.call(x, training=True))
    return model.summary()

  def get_model_with_inputs(self, inputs, **kargs):
    model = tf.keras.Model(
        inputs=[inputs], outputs=self.call(inputs, training=True))
    return model

  
  
  def call(self, inputs, training=False, with_endpoints=False):
    """Implementation of call().
    Args:
      inputs: input tensors.
      training: boolean, whether the model is constructed for training.
      with_endpoints: If true, return a list of endpoints.
    Returns:
      output tensors.
    """
    outputs = None
    self.endpoints = {}
    reduction_idx = 0

    # Calls Stem layers
    outputs = self._stem(inputs, training)
    swin_output = self.swin_input(outputs)
    swin_output = self.patch_embed(swin_output)
    if self.ape:
            swin_output = swin_output + self.absolute_pos_embed
        swin_output = self.pos_drop(swin_output)
    
    logging.info('Built stem: %s (%s)', outputs.shape, outputs.dtype)
    self.endpoints['stem'] = outputs

    # Calls blocks.
    for idx, block in enumerate(self._blocks):
      is_reduction = False  # reduction flag for blocks after the stem layer
      if ((idx == len(self._blocks) - 1) or
          self._blocks[idx + 1].block_args.strides > 1):
        is_reduction = True
        reduction_idx += 1

      survival_prob = self._mconfig.survival_prob
      if survival_prob:
        drop_rate = 1.0 - survival_prob
        survival_prob = 1.0 - drop_rate * float(idx) / len(self._blocks)
        logging.info('block_%s survival_prob: %s', idx, survival_prob)
      outputs = block(outputs, training=training, survival_prob=survival_prob)
      self.endpoints['block_%s' % idx] = outputs
      if is_reduction:
        self.endpoints['reduction_%s' % reduction_idx] = outputs
        if reduction_idx > 1:
          swin_output = self.blocks[reduction_idx-1](swin_output)
          path_embed = self.embeder[reduction_idx-1](outputs)
          trans_embed = self.TransposePatchEmbed(swin_output)
          outputs = layers.Concatenate()[trans_embed,outputs]
          outputs = MyDense()(outputs)
          swin_output = layers.Concatenate()[swin_output, patch_embed]
          outputs = MyDense(cnn=False)(swin_output)
          
      if block.endpoints:
        for k, v in block.endpoints.items():
          self.endpoints['block_%s/%s' % (idx, k)] = v
          if is_reduction:
            self.endpoints['reduction_%s/%s' % (reduction_idx, k)] = v
    self.endpoints['features'] = outputs

    # Head to obtain the final feature.
    outputs = self._head(outputs, training)
    swin_outputs = self.avgpool(swin_outputs)
    outputs = layers.Concatenate()[outputs, swin_outputs]
    self.endpoints.update(self._head.endpoints)

   

    if with_endpoints:  # Use for building sequential models.
      return [outputs] + list(
          filter(lambda endpoint: endpoint is not None, [
              self.endpoints.get('reduction_1'),
              self.endpoints.get('reduction_2'),
              self.endpoints.get('reduction_3'),
              self.endpoints.get('reduction_4'),
              self.endpoints.get('reduction_5'),
          ]))

    return outputs


def get_model(model_name,
              model_config=None,
              include_top=True,
              weights='imagenet',
              training=True,
              with_endpoints=False,
              **kwargs):
  """Get a EfficientNet V1 or V2 model instance.
  This is a simply utility for finetuning or inference.
  Args:
    model_name: a string such as 'efficientnetv2-s' or 'efficientnet-b0'.
    model_config: A dict of model configurations or a string of hparams.
    include_top: whether to include the final dense layer for classification.
    weights: One of None (random initialization),
      'imagenet' (pretrained on ImageNet),
      'imagenet21k' (pretrained on Imagenet21k),
      'imagenet21k-ft1k' (pretrained on 21k and finetuned on 1k), 
      'jft' (trained with non-labelled JFT-300),
      or the path to the weights file to be loaded. Defaults to 'imagenet'.
    training: If true, all model variables are trainable.
    with_endpoints: whether to return all intermedia endpoints.
    **kwargs: additional parameters for keras model, such as name=xx.
  Returns:
    A single tensor if with_endpoints if False; otherwise, a list of tensor.
  """
  net = EffNetV2Model(model_name, model_config, include_top, **kwargs)
  net(tf.keras.Input(shape=(None, None, 3)),
      training=training,
      with_endpoints=with_endpoints)

  if not weights:  # pylint: disable=g-bool-id-comparison
    return net

  v2url = 'https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/'
  v1url = 'https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/advprop/'
  v1jfturl = 'https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/noisystudent/'
  pretrained_ckpts = {
      # EfficientNet V2.
      'efficientnetv2-s': {
          'imagenet': v2url + 'efficientnetv2-s.tgz',
          'imagenet21k': v2url + 'efficientnetv2-s-21k.tgz',
          'imagenet21k-ft1k': v2url + 'efficientnetv2-s-21k-ft1k.tgz',
      },
      'efficientnetv2-m': {
          'imagenet': v2url + 'efficientnetv2-m.tgz',
          'imagenet21k': v2url + 'efficientnetv2-m-21k.tgz',
          'imagenet21k-ft1k': v2url + 'efficientnetv2-m-21k-ft1k.tgz',
      },
      'efficientnetv2-l': {
          'imagenet': v2url + 'efficientnetv2-l.tgz',
          'imagenet21k': v2url + 'efficientnetv2-l-21k.tgz',
          'imagenet21k-ft1k': v2url + 'efficientnetv2-l-21k-ft1k.tgz',
      },
      'efficientnetv2-xl': {
          # no imagenet ckpt.
          'imagenet21k': v2url + 'efficientnetv2-xl-21k.tgz',
          'imagenet21k-ft1k': v2url + 'efficientnetv2-xl-21k-ft1k.tgz',
      },

      'efficientnetv2-b0': {
          'imagenet': v2url + 'efficientnetv2-b0.tgz',
          'imagenet21k': v2url + 'efficientnetv2-b0-21k.tgz',
          'imagenet21k-ft1k': v2url + 'efficientnetv2-b0-21k-ft1k.tgz',
      },
      'efficientnetv2-b1': {
          'imagenet': v2url + 'efficientnetv2-b1.tgz',
          'imagenet21k': v2url + 'efficientnetv2-b1-21k.tgz',
          'imagenet21k-ft1k': v2url + 'efficientnetv2-b1-21k-ft1k.tgz',
      },
      'efficientnetv2-b2': {
          'imagenet': v2url + 'efficientnetv2-b2.tgz',
          'imagenet21k': v2url + 'efficientnetv2-b2-21k.tgz',
          'imagenet21k-ft1k': v2url + 'efficientnetv2-b2-21k-ft1k.tgz',
      },
      'efficientnetv2-b3': {
          'imagenet': v2url + 'efficientnetv2-b3.tgz',
          'imagenet21k': v2url + 'efficientnetv2-b3-21k.tgz',
          'imagenet21k-ft1k': v2url + 'efficientnetv2-b3-21k-ft1k.tgz',
      },

      # EfficientNet V1.
      'efficientnet-b0': {
          'imagenet': v1url + 'efficientnet-b0.tar.gz',
          'jft': v1jfturl + 'noisy_student_efficientnet-b0.tar.gz',
      },
      'efficientnet-b1': {
          'imagenet': v1url + 'efficientnet-b1.tar.gz',
          'jft': v1jfturl + 'noisy_student_efficientnet-b1.tar.gz',
      },
      'efficientnet-b2': {
          'imagenet': v1url + 'efficientnet-b2.tar.gz',
          'jft': v1jfturl + 'noisy_student_efficientnet-b2.tar.gz',
      },
      'efficientnet-b3': {
          'imagenet': v1url + 'efficientnet-b3.tar.gz',
          'jft': v1jfturl + 'noisy_student_efficientnet-b3.tar.gz',
      },
      'efficientnet-b4': {
          'imagenet': v1url + 'efficientnet-b4.tar.gz',
          'jft': v1jfturl + 'noisy_student_efficientnet-b4.tar.gz',
      },
      'efficientnet-b5': {
          'imagenet': v1url + 'efficientnet-b5.tar.gz',
          'jft': v1jfturl + 'noisy_student_efficientnet-b5.tar.gz',
      },
      'efficientnet-b6': {
          'imagenet': v1url + 'efficientnet-b6.tar.gz',
          'jft': v1jfturl + 'noisy_student_efficientnet-b6.tar.gz',
      },
      'efficientnet-b7': {
          'imagenet': v1url + 'efficientnet-b7.tar.gz',
          'jft': v1jfturl + 'noisy_student_efficientnet-b7.tar.gz',
      },
      'efficientnet-b8': {
          'imagenet': v1url + 'efficientnet-b8.tar.gz',
      },
      'efficientnet-l2': {
          'jft': v1jfturl + 'noisy_student_efficientnet-l2_475.tar.gz',
      },
  }

  if model_name in pretrained_ckpts and weights in pretrained_ckpts[model_name]:
    url = pretrained_ckpts[model_name][weights]
    fname = os.path.basename(url).split('.')[0]
    pretrained_ckpt= tf.keras.utils.get_file(fname, url , untar=True)
  else:
    pretrained_ckpt = weights

  if tf.io.gfile.isdir(pretrained_ckpt):
    pretrained_ckpt = tf.train.latest_checkpoint(pretrained_ckpt)
  net.load_weights(pretrained_ckpt)
  return net
