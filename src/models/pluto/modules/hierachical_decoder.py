from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..layers.embedding import PointsEncoder
from ..layers.fourier_embedding import FourierEmbedding
from ..layers.mlp_layer import MLPLayer


class DecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, dropout) -> None:
        super().__init__()
        self.dim = dim

        self.r2r_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.m2m_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt,
        memory,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        m_pos: Optional[Tensor] = None,
    ):
        """
        tgt: (bs, R, M, dim)
        tgt_key_padding_mask: (bs, R)
        """
        bs, R, M, D = tgt.shape

        tgt = tgt.transpose(1, 2).reshape(bs * M, R, D)
        tgt2 = self.norm1(tgt)
        tgt2 = self.r2r_attn(
            tgt2, tgt2, tgt2, key_padding_mask=tgt_key_padding_mask.repeat(M, 1)
        )[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt_tmp = tgt.reshape(bs, M, R, D).transpose(1, 2).reshape(bs * R, M, D)
        tgt_valid_mask = ~tgt_key_padding_mask.reshape(-1)
        tgt_valid = tgt_tmp[tgt_valid_mask]
        tgt2_valid = self.norm2(tgt_valid)
        tgt2_valid, _ = self.m2m_attn(
            tgt2_valid + m_pos, tgt2_valid + m_pos, tgt2_valid
        )
        tgt_valid = tgt_valid + self.dropout2(tgt2_valid)
        tgt = torch.zeros_like(tgt_tmp)
        tgt[tgt_valid_mask] = tgt_valid

        tgt = tgt.reshape(bs, R, M, D).view(bs, R * M, D)
        tgt2 = self.norm3(tgt)
        tgt2 = self.cross_attn(
            tgt2, memory, memory, key_padding_mask=memory_key_padding_mask
        )[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm4(tgt)
        tgt2 = self.ffn(tgt2)
        tgt = tgt + self.dropout3(tgt2)
        tgt = tgt.reshape(bs, R, M, D)

        return tgt


class HierachicalDecoder(nn.Module):
    def __init__(
        self,
        num_mode,
        decoder_depth,
        dim,
        num_heads,
        mlp_ratio,
        dropout,
        future_steps,
        history_steps,
        yaw_constraint=False,
        cat_x=False,
        recursive_decoder: bool = False,
        wtd_with_history: bool = False,
        multihead_decoder: bool = False,
    ) -> None:
        super().__init__()

        self.num_mode = num_mode
        self.future_steps = future_steps
        self.history_steps = history_steps
        self.yaw_constraint = yaw_constraint
        self.cat_x = cat_x

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderLayer(dim, num_heads, mlp_ratio, dropout)
                for _ in range(decoder_depth)
            ]
        )

        self.r_pos_emb = FourierEmbedding(3, dim, 64)
        self.r_encoder = PointsEncoder(6, dim)

        self.q_proj = nn.Linear(2 * dim, dim)

        self.m_emb = nn.Parameter(torch.Tensor(1, 1, num_mode, dim))
        self.m_pos = nn.Parameter(torch.Tensor(1, num_mode, dim))

        if self.cat_x:
            self.cat_x_proj = nn.Linear(2 * dim, dim)

        self.loc_head = MLPLayer(dim, 2 * dim, self.future_steps * 2)
        self.yaw_head = MLPLayer(dim, 2 * dim, self.future_steps * 2)
        self.vel_head = MLPLayer(dim, 2 * dim, self.future_steps * 2)
        self.pi_head = MLPLayer(dim, dim, 1)

        nn.init.normal_(self.m_emb, mean=0.0, std=0.01)
        nn.init.normal_(self.m_pos, mean=0.0, std=0.01)
        self.recursive_decoder = recursive_decoder
        self.multihead_decoder = multihead_decoder
        self.wtd_with_history = wtd_with_history
        self.time_steps = self.future_steps + self.history_steps if wtd_with_history else self.future_steps
        if recursive_decoder:
            self.detail_head = MLPLayer(dim, 2 * dim, self.time_steps * 2)
        if multihead_decoder:
            self.detail_head_0 = MLPLayer(dim, 2 * dim, self.time_steps * 2)
            self.detail_head_1 = MLPLayer(dim, 2 * dim, self.time_steps * 2)
            self.detail_head_2 = MLPLayer(dim, 2 * dim, self.time_steps * 2)
            self.detail_head_3 = MLPLayer(dim, 2 * dim, self.time_steps * 2)
            self.approximation_head = MLPLayer(dim, 2 * dim, self.time_steps * 2)
            self.detail_decoders = [self.detail_head_0, self.detail_head_1, self.detail_head_2, self.detail_head_3, self.approximation_head]
        if multihead_decoder and recursive_decoder:
            raise ValueError("multihead_decoder and recursive_decoder should not be both True")

    def forward(self, data, enc_data):
        enc_emb = enc_data["enc_emb"]
        # print('enc_emb', enc_emb.shape) # [12, 276, 128] emb of all the things
        enc_key_padding_mask = enc_data["enc_key_padding_mask"]

        r_position = data["reference_line"]["position"]
        r_vector = data["reference_line"]["vector"]
        r_orientation = data["reference_line"]["orientation"]
        r_valid_mask = data["reference_line"]["valid_mask"]
        r_key_padding_mask = ~r_valid_mask.any(-1)

        r_feature = torch.cat(
            [
                r_position - r_position[..., 0:1, :2],
                r_vector,
                torch.stack([r_orientation.cos(), r_orientation.sin()], dim=-1),
            ],
            dim=-1,
        )

        bs, R, P, C = r_feature.shape
        r_valid_mask = r_valid_mask.view(bs * R, P)
        r_feature = r_feature.reshape(bs * R, P, C)
        r_emb = self.r_encoder(r_feature, r_valid_mask).view(bs, R, -1)

        r_pos = torch.cat([r_position[:, :, 0], r_orientation[:, :, 0, None]], dim=-1)
        r_emb = r_emb + self.r_pos_emb(r_pos)

        r_emb = r_emb.unsqueeze(2).repeat(1, 1, self.num_mode, 1)
        m_emb = self.m_emb.repeat(bs, R, 1, 1)

        q = self.q_proj(torch.cat([r_emb, m_emb], dim=-1))

        residual = [q]
        # depth represents the information propagation distance
        for blk in self.decoder_blocks:
            # print('q', q.shape)# q torch.Size([12, 4, 6, 128]) the bs, ref_road_num, mode_num, dim
            # TODO: this can be recorded and used to compare with the decomposed trajectory(as has the property of different horizon)
            d_q = blk(
                q,
                enc_emb,
                tgt_key_padding_mask=r_key_padding_mask,
                memory_key_padding_mask=enc_key_padding_mask,
                m_pos=self.m_pos,
            )
            if self.recursive_decoder:
                residual.append(d_q)
                q = q + d_q
            else:
                q = d_q
            assert torch.isfinite(q).all()

        if self.cat_x:
            x = enc_emb[:, 0].unsqueeze(1).unsqueeze(2).repeat(1, R, self.num_mode, 1)
            q = self.cat_x_proj(torch.cat([q, x], dim=-1))

        loc = self.loc_head(q).view(bs, R, self.num_mode, self.future_steps, 2)
        yaw = self.yaw_head(q).view(bs, R, self.num_mode, self.future_steps, 2)
        vel = self.vel_head(q).view(bs, R, self.num_mode, self.future_steps, 2)
        pi = self.pi_head(q).squeeze(-1)

        traj = torch.cat([loc, yaw, vel], dim=-1)

        details = []
        if self.recursive_decoder:
            for r in residual:
                assert torch.isfinite(r).all()
                # detail_loc = self.detail_head(r).view(bs, R, self.num_mode, self.future_steps, 2)
                # detail_yaw = self.detail_head(r).view(bs, R, self.num_mode, self.future_steps, 2)
                detail = self.detail_head(r).view(bs, R, self.num_mode, self.time_steps, 2)
                # detail = torch.cat([detail_loc, detail_yaw, detail_vel], dim=-1)
                details.append(detail)
        if self.multihead_decoder:
            for decoder in self.detail_decoders:
                detail = decoder(q).view(bs, R, self.num_mode, self.time_steps, 2)
                details.append(detail)

        return traj, pi, details
