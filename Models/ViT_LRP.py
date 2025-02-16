""" Vision Transformer (ViT) in PyTorch
Hacked together by / Copyright 2020 Ross Wightman
"""

import os
import torch
import torch.nn as nn
from einops import rearrange
import networkx as nx
from tqdm import tqdm

from .layers_ours import *
from .weight_init import trunc_normal_
from .layer_helpers import to_2tuple
from Utils.Misc_utils import get_stat


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    # patch models
    "vit_small_patch16_224": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth",
    ),
    "vit_base_patch16_224": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth",
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "vit_large_patch16_224": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth",
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
}


def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = (
        torch.eye(num_tokens)
        .expand(batch_size, num_tokens, num_tokens)
        .to(all_layer_matrices[0].device)
    )
    all_layer_matrices = [
        all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))
    ]
    # all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
    #                       for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer + 1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = GELU()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def relprop(self, cam, **kwargs):
        cam = self.drop.relprop(cam, **kwargs)
        cam = self.fc2.relprop(cam, **kwargs)
        cam = self.act.relprop(cam, **kwargs)
        cam = self.fc1.relprop(cam, **kwargs)
        return cam


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        add_hook=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.add_hook = add_hook
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim**-0.5

        # A = Q*K^T
        self.matmul1 = einsum("bhid,bhjd->bhij")
        # attn = A*V
        self.matmul2 = einsum("bhij,bhjd->bhid")

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(proj_drop)
        self.softmax = Softmax(dim=-1)

        self.attn_map = None
        self.attn_cam = None
        self.attn_gradients = None
        self.attn_pruning_mask = None
        self.v = None
        self.v_cam = None

    def get_attn_map(self):
        return self.attn_map

    def save_attn_map(self, attn):
        self.attn_map = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def set_attn_pruning_mask(self, mask, method="AND"):
        if self.attn_pruning_mask is None:
            self.attn_pruning_mask = mask
        else:
            assert (
                self.attn_pruning_mask.shape == mask.shape
            ), "Attention class set_attn_pruning_mask(): The shape of the mask is not correct."
            assert method in [
                "AND",
                "LAST",
            ], "Invalid method for mask generation. Choose from ['AND', 'LAST'.]"
            if method == "AND":
                self.attn_pruning_mask = self.attn_pruning_mask * mask
            elif method == "LAST":
                self.attn_pruning_mask = mask

    def get_attn_pruning_mask(self):
        return self.attn_pruning_mask

    def get_v(self):
        return self.v

    def save_v(self, v):
        self.v = v

    def save_v_cam(self, cam):
        self.v_cam = cam

    def get_v_cam(self):
        return self.v_cam

    def forward(self, x):
        b, n, _, h = *x.shape, self.num_heads
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=h)

        self.save_v(v)

        dots = self.matmul1([q, k]) * self.scale

        attn = self.softmax(dots)
        attn = self.attn_drop(attn)
        self.save_attn_map(attn)

        if self.attn_pruning_mask is not None:
            # element-wise multiplication of the mask of all h heads with the attention matrices of all h heads for all of the #batch_size records
            # (we expand the mask to match the shape of the attention matrices which includes the batch demension)
            attn = attn * self.attn_pruning_mask.expand(attn.shape)

        if x.requires_grad and self.add_hook:
            attn.register_hook(self.save_attn_gradients)

        out = self.matmul2([attn, v])

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.proj(out)
        out = self.proj_drop(out)

        return out

    def relprop(self, cam, **kwargs):
        cam = self.proj_drop.relprop(cam, **kwargs)
        cam = self.proj.relprop(cam, **kwargs)
        cam = rearrange(cam, "b n (h d) -> b h n d", h=self.num_heads)

        # attn = A*V
        (cam1, cam_v) = self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam_v /= 2

        self.save_v_cam(cam_v)
        self.save_attn_cam(cam1)

        cam1 = self.attn_drop.relprop(cam1, **kwargs)
        cam1 = self.softmax.relprop(cam1, **kwargs)

        # A = Q*K^T
        (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)
        cam_q /= 2
        cam_k /= 2

        cam_qkv = rearrange(
            [cam_q, cam_k, cam_v],
            "qkv b h n d -> b n (qkv h d)",
            qkv=3,
            h=self.num_heads,
        )

        return self.qkv.relprop(cam_qkv, **kwargs)


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        add_hook=False,
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            add_hook=add_hook,
        )
        self.norm2 = LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()

    def save_all(self, all):
        self.all = all

    def get_all(self):
        return self.all

    def forward(self, x):
        x1, x2 = self.clone1(x, 2)
        x = self.add1([x1, self.attn(self.norm1(x2))])
        x1, x2 = self.clone2(x, 2)
        x = self.add2([x1, self.mlp(self.norm2(x2))])
        self.save_all(x)
        return x

    def relprop(self, cam, **kwargs):
        (cam1, cam2) = self.add2.relprop(cam, **kwargs)
        cam2 = self.mlp.relprop(cam2, **kwargs)
        cam2 = self.norm2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        cam2 = self.attn.relprop(cam2, **kwargs)
        cam2 = self.norm1.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)
        return cam


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

    def relprop(self, cam, **kwargs):
        cam = cam.transpose(1, 2)
        cam = cam.reshape(
            cam.shape[0],
            cam.shape[1],
            (self.img_size[0] // self.patch_size[0]),
            (self.img_size[1] // self.patch_size[1]),
        )
        return self.proj.relprop(cam, **kwargs)


class VisionTransformer(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        mlp_head=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        add_hook=False,
        need_ig=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depth = depth
        self.add_hook = add_hook
        self.need_ig = need_ig
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.depth = depth
        self.num_heads = num_heads

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    add_hook=add_hook,
                )
                for i in range(depth)
            ]
        )

        self.norm = LayerNorm(embed_dim)
        if mlp_head:
            # paper diagram suggests 'MLP head', but results in 4M extra parameters vs paper
            if num_classes == 2:
                self.head = Mlp(embed_dim, int(embed_dim * mlp_ratio), 1)
            else:
                self.head = Mlp(embed_dim, int(embed_dim * mlp_ratio), num_classes)
        else:
            # with a single Linear layer as head, the param count within rounding of paper
            if num_classes == 2:
                self.head = Linear(embed_dim, 1)
            else:
                self.head = Linear(embed_dim, num_classes)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        self.pool = IndexSelect()
        self.add = Add()

        self.inp_grad = None
        self.ig = None
        self.graph = None
        self.edge_type = None

    def build_graph(
        self,
        edge_weights,
        branch="main",
        edge_type="attention_weights",
        prun_iter_cnt=None,
        output_folder_path="",
    ):

        assert edge_type in [
            "attention_weights",
            "gradients",
        ], "Invalid edge_type. Choose from 'attention_weights', or 'gradients'"

        self.edge_type = edge_type
        num_blocks = len(self.blocks)
        num_heads = self.num_heads
        num_tokens = 197  # Typically 196 patches + 1 class token

        graph_file_path = os.path.join(
            output_folder_path,
            f"VTGraph_B{num_blocks}_H{num_heads}_T197_B({branch})_E({edge_type})_PIter({prun_iter_cnt}).graphml",
        )

        # Check if the graph already exists
        if os.path.exists(graph_file_path):
            self.graph = nx.read_graphml(graph_file_path)
            print(f"Loaded graph from {graph_file_path}")
            return

        G = nx.DiGraph()  # Directed graph to match the flow in the transformer

        total_iterations = (
            num_blocks * num_heads * num_tokens * num_tokens
            + (num_blocks - 1) * num_heads * num_tokens
        )

        with tqdm(total=total_iterations, desc=f"Building the {branch} Graph") as pbar:
            # Step 1: Create nodes and intra-block edges for each head in each block
            for block_idx in range(num_blocks):
                for head_idx in range(num_heads):
                    # Add nodes
                    nodes = [
                        (block_idx, head_idx, token_idx)
                        for token_idx in range(num_tokens)
                    ]
                    G.add_nodes_from(nodes)

                    # Add intra-block edges using adjacency matrix
                    for i in range(num_tokens):
                        for j in range(num_tokens):
                            G.add_edge(
                                nodes[i],
                                nodes[j],
                                weight=edge_weights[block_idx, head_idx, i, j],
                            )
                            pbar.update(1)

            # Step 2: Add inter-block edges to connect corresponding tokens across blocks
            for block_idx in range(num_blocks - 1):
                for head_idx in range(num_heads):
                    for token_idx in range(num_tokens):
                        current_node = (block_idx, head_idx, token_idx)
                        next_node = (block_idx + 1, head_idx, token_idx)
                        G.add_edge(
                            current_node, next_node, weight=1.0
                        )  # uniform weight for inter-block connections
                        pbar.update(1)

        # Save the graph to file
        if len(output_folder_path) > 0:
            nx.write_graphml(G, graph_file_path)
            print(f"Saved graph to {graph_file_path}")

        self.graph = G

    def get_graph(self):
        return self.graph

    def set_add_hook(self):
        self.add_hook = True

    def save_inp_grad(self, grad):
        self.inp_grad = grad

    def get_inp_grad(self):
        return self.inp_grad

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def set_attn_pruning_mask(self, mask, method="AND"):
        assert (
            mask.shape[0] == self.depth
        ), "ERROR: Mask shape doesn't match the depth of the model."
        for i in range(self.depth):
            self.blocks[i].attn.set_attn_pruning_mask(mask[i], method=method)

    def get_attn_pruning_mask(self):
        attn_pruning_mask = []
        for i in range(self.depth):
            mask = self.blocks[i].attn.get_attn_pruning_mask()
            if mask is None:
                return None
            attn_pruning_mask.append(mask.detach())
        attn_pruning_mask = torch.stack(attn_pruning_mask, dim=0)

        return attn_pruning_mask

    def clear_gradients(self):
        # Iterate through all parameters of the model and set their gradients to None
        for param in self.parameters():
            if param.grad is not None:
                param.grad.detach_()
                param.grad = None

    def save_ig(self, ig):
        self.ig = ig

    def get_ig(self):
        return self.ig

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.add([x, self.pos_embed])

        if x.requires_grad and self.add_hook:
            x.register_hook(self.save_inp_grad)

        if self.need_ig:
            ig = []

            for blk in self.blocks:
                x = blk(x)
                y = self.norm(x)
                y = self.pool(y, dim=1, indices=torch.tensor(0, device=y.device))
                y = y.squeeze(1)
                y = self.head(y)
                ig.append(y)

            self.save_ig(ig)
        else:
            for blk in self.blocks:
                x = blk(x)

        x = self.norm(x)
        x = self.pool(x, dim=1, indices=torch.tensor(0, device=x.device))
        x = x.squeeze(1)
        x = self.head(x)
        return x

    def relprop(
        self,
        cam=None,
        method="transformer_attribution",
        is_ablation=False,
        start_layer=0,
        **kwargs,
    ):

        cam = self.head.relprop(cam, **kwargs)
        cam = cam.unsqueeze(1)
        cam = self.pool.relprop(cam, **kwargs)
        cam = self.norm.relprop(cam, **kwargs)
        for blk in reversed(self.blocks):
            cam = blk.relprop(cam, **kwargs)

        if method == "full":
            (cam, _) = self.add.relprop(cam, **kwargs)
            cam = cam[:, 1:]
            cam = self.patch_embed.relprop(cam, **kwargs)
            # sum on channels
            cam = cam.sum(dim=1)
            return cam

        elif method == "rollout":
            # cam rollout
            attn_cams = []
            for blk in self.blocks:
                attn_heads = blk.attn.get_attn_cam().clamp(min=0)
                avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
                attn_cams.append(avg_heads)
            cam = compute_rollout_attention(attn_cams, start_layer=start_layer)
            cam = cam[:, 0, 1:]
            return cam

        elif method == "transformer_attribution":
            cams = []
            blk_attrs = []
            for blk in self.blocks:
                grad = blk.attn.get_attn_gradients()
                cam = blk.attn.get_attn_cam()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0)
                blk_attrs.append(cam)
                cam = cam.mean(dim=0)
                cams.append(cam.unsqueeze(0))
            rollout = compute_rollout_attention(cams, start_layer=start_layer)
            cam = rollout[:, 0, 1:]
            blk_attrs = torch.stack(blk_attrs)
            return cam, blk_attrs

        elif method == "grad":
            blk_grads = []
            for blk in self.blocks:
                grad = blk.attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                grad = grad.clamp(min=0)
                blk_grads.append(grad)

            blk_grads = torch.stack(blk_grads)
            return None, blk_grads

        elif method == "attr":
            blk_attrs = []
            for blk in self.blocks:
                cam = blk.attn.get_attn_cam()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                cam = cam.clamp(min=0)
                blk_attrs.append(cam)
            blk_attrs = torch.stack(blk_attrs)

            return None, blk_attrs

        elif method == "last_layer":
            cam = self.blocks[-1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.blocks[-1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "last_layer_attn":
            cam = self.blocks[-1].attn.get_attn()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "second_layer":
            cam = self.blocks[1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.blocks[1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam
        else:
            print("ERROR: Invalid method name.")


def _conv_filter(state_dict, patch_size=16):
    """convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if "patch_embed.proj.weight" in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


# def vit_base_patch16_224(pretrained=False, **kwargs):
#     model = VisionTransformer(
#         patch_size=16,
#         embed_dim=768,
#         depth=12,
#         num_heads=12,
#         mlp_ratio=4,
#         qkv_bias=True,
#         **kwargs,
#     )
#     model.default_cfg = default_cfgs["vit_base_patch16_224"]
#     if pretrained:
#         load_pretrained(
#             model,
#             num_classes=model.num_classes,
#             in_chans=kwargs.get("in_chans", 3),
#             filter_fn=_conv_filter,
#         )
#     return model


# def vit_large_patch16_224(pretrained=False, **kwargs):
#     model = VisionTransformer(
#         patch_size=16,
#         embed_dim=1024,
#         depth=24,
#         num_heads=16,
#         mlp_ratio=4,
#         qkv_bias=True,
#         **kwargs,
#     )
#     model.default_cfg = default_cfgs["vit_large_patch16_224"]
#     if pretrained:
#         load_pretrained(
#             model, num_classes=model.num_classes, in_chans=kwargs.get("in_chans", 3)
#         )
#     return model


# def deit_base_patch16_224(pretrained=False, **kwargs):
#     model = VisionTransformer(
#         patch_size=16,
#         embed_dim=768,
#         depth=12,
#         num_heads=12,
#         mlp_ratio=4,
#         qkv_bias=True,
#         **kwargs,
#     )
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = torch.hub.load_state_dict_from_url(
#             url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
#             map_location="cpu",
#             check_hash=True,
#         )
#         model.load_state_dict(checkpoint["model"])
#     return model


def deit_small_patch16_224(
    pretrained=False,
    pretrained_path=None,
    weight_path=None,
    num_classes=3,
    add_hook=False,
    need_ig=False,
    device="cuda",
    **kwargs,
):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        num_classes=num_classes,
        qkv_bias=True,
        add_hook=add_hook,
        need_ig=need_ig,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        if pretrained_path:
            checkpoint = torch.load(pretrained_path)
        else:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
                map_location="cpu",
                check_hash=True,
            )
        model.load_state_dict(
            {k: v for k, v in checkpoint["model"].items() if "head" not in k},
            strict=False,
        )
        print("Pre-trained weights on ImageNet loaded...")
    elif weight_path is not None:
        checkpoint = torch.load(weight_path, map_location=device)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Weights loaded from {weight_path} from the state_dict")
        else:
            print(
                'Couldn\'t find "model_state_dict" in the given checkpoint. No weights loaded, using random weights...'
            )
    else:
        print("No weights loaded, using random weights...")
    return model
