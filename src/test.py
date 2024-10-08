import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple
from torch import Tensor
from typing import Optional
from functools import partial




try:
    from mamba_smm.ops.triton.layernorm import RMSNorm,layer_norm_fn,rms_norm_fn
except ImportError:
    RMSNorm ,rms_norm_fn,layer_norm_fn = None


#切方块 patch_size为切的个数
#B C H W -> B embed_dim grid_size,grid_size-> B embed_dim num_patches (四维变三维）
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224,patch_size=16,stride=16,in_channels=3,embed_dim=768,norm_layer=None,flatten=True):
        super(PatchEmbed,self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

        self.grid_size = ((img_size[0]-patch_size[0])//stride+1,(img_size[0]-patch_size[0])//stride+1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_channels,embed_dim,kernel_size=patch_size,stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    def forward(self,x):
        B,C,H,W = x.shape  #H W是图片的长宽
        assert H == self.img_size[0] and W == self.img_size[1],\
        f"Input img size{(H)*(W)} dosen't match model({self.image[0]} * {self.imahe[1]})"
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1,2)
        x = self.norm(x)
        return x


class Block(nn.Module):
    def __init__(
            self,dim,mixer_cls,
            norm_cls=nn.LayerNorm,
            fused_add_norm=False,residual_in_fp32=False,drop_path=0
    ):
        super(Block,self).__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)

        self.drop_path = DropPath(drop_path)

        if self.fused_add_norm:
            assert RMSNorm is not None,"RMSNorm import Fails"
            assert isinstance(
                self.norm,(nn.LayerNorm,RMSNorm)
            ),"Only LayerNorm and RMSNorm are supported for fused_add_norm"

        def forward(self,
                    hidden_states: Tensor, residual:Optional[Tensor] = None,
                    inference_params=None):
            if not self.fused_add_norm:
                if residual is None:
                    residual = hidden_states
                else:
                    residual = residual+self.drop_path(hidden_states)
                hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)

            else:
                fused_add_norm = rms_norm_fn if isinstance(self.norm,RMSNorm) else layer_norm_fn
                if residual is None:
                    hidden_states,residual = fused_add_norm(
                        hidden_states,
                        self.norm.weight,
                        self.norm.bias,
                        residual = residual,
                        prenorm = True,
                        residual_in_fp32 = self.residual_in_fp32,
                        eps = self.norm.eps,
                    )
                else:
                    hidden_states, residual = fused_add_norm(
                        self.drop_path(hidden_states),
                        self.norm.weight,
                        self.norm.bias,
                        residual = residual,
                        prenorm=True,
                        residual_in_fp32=self.residual_in_fp32,
                    )
            hidden_states = self.mixer(hidden_states,inference_params=inference_params)
            return hidden_states,residual

def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon = 1e-5,
        drop_path = 0.,
        rms_norm = False,
        residual_in_fp32 = False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
        if_bimamba=None,
        bimamba_type="none",
        if_device_out=False,
        init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type="v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device":device,"dtype":dtype}
    mixer_cls = partial(
        Mamab,
        layer_idx=layer_idx,
        bimamba_type=bimamba_type,
        if_device_out = if_device_out,
        init_layer_scale = init_layer_scale,
        **ssm_cfg,
        **factory_kwargs
    )
    norm_cls=partial(
        nn.LayerNorm if not rms_norm else RMSNorm,eps=norm_epsilon,**factory_kwargs
    )
    block = Block(
        dim=d_model,
        mixer_cls=mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32
    )
    block.layer_idx = layer_idx
    return block