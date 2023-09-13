import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torchmetrics.classification import MulticlassAccuracy
import numpy as np


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels: 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels: 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels:] = emb_z

        self.cached_penc = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return self.cached_penc


class Block(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(Block, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(dim_feedforward, d_model)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_q, src_mask=None, src_key_padding_mask=None):
        src_q = src_q.permute(1, 0, 2)
        src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src_q = src_q + self.dropout12(src2)
        src_q = self.norm1(src_q)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
        src_q = src_q + self.dropout2(src2)
        src_q = self.norm2(src_q)
        return src_q.permute(1, 0, 2)


class CrossBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(CrossBlock, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cm_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src_q, src_v, src_mask=None, src_key_padding_mask=None):
        src_q = src_q.permute(1, 0, 2)
        src_v = src_v.permute(1, 0, 2)
        src1 = self.cm_attn(src_q, src_v, src_v, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src_q = src_q + self.dropout11(src1) + self.dropout12(src2)
        src_q = self.norm1(src_q)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
        src_q = src_q + self.dropout2(src2)
        src_q = self.norm2(src_q)
        return src_q.permute(1, 0, 2)


class UniModalTransformer(nn.Module):

    def __init__(self, args, encoder_layer, num_layers, norm=None):
        super(UniModalTransformer, self).__init__()

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm2 = nn.LayerNorm(512)
        self.norm = norm

    def forward(self, modality_1, mask=None, src_key_padding_mask=None):
        for i in range(self.num_layers):
            modality_1 = self.layers[i](modality_1, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm:
            modality_1 = self.norm2(modality_1)
        return modality_1


class CrossModalTransformer(nn.Module):

    def __init__(self, args, encoder_layer, num_layers, norm=None):
        super(CrossModalTransformer, self).__init__()

        self.args = args

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.norm = norm

    def forward(self, modality_1, modality_2, mask=None, src_key_padding_mask=None):
        for i in range(self.num_layers):
            modality_1 = self.layers[i](modality_1, modality_2, src_mask=mask,
                                        src_key_padding_mask=src_key_padding_mask)
            modality_2 = self.layers[i](modality_2, modality_1, src_mask=mask,
                                        src_key_padding_mask=src_key_padding_mask)
        if self.norm:
            modality_1 = self.norm1(modality_1)
            modality_2 = self.norm2(modality_2)
        return modality_1, modality_2


class TemporalSelection(nn.Module):

    def __init__(self, args):
        super(TemporalSelection, self).__init__()
        self.args = args
        self.attn_qst_query = nn.MultiheadAttention(512, 4, dropout=0.1)

    def soft_attention_select_2d(self, query_feat, kv_feat):
        kv_feat = kv_feat.permute(1, 0, 2)
        query_feat = query_feat.unsqueeze(0)
        _, temp_weights = self.attn_qst_query(query_feat, kv_feat, kv_feat,
                                              attn_mask=None, key_padding_mask=None)
        return temp_weights

    def find_top_k_fast(self, temp_weights, modality, clip_len, C):
        _, top_k_index_sort = torch.topk(temp_weights, k=self.args.top_k, dim=-1)
        modality_new = modality[torch.arange(modality.size(0)).unsqueeze(1), top_k_index_sort.squeeze(1)]
        return modality_new, top_k_index_sort

    def select_top_k_fast(self, patch_feat, top_k_index_sort):
        patch_select_new = patch_feat[torch.arange(patch_feat.size(0)).unsqueeze(1), top_k_index_sort.squeeze(1)]
        return patch_select_new

    def forward(self, query, key, value):
        B, T, C = query.size()
        clip_len = int(T / self.args.segs)
        temp_weights = self.soft_attention_select_2d(key, query)
        top_k_query, top_k_index_sort = self.find_top_k_fast(temp_weights, query,
                                                             clip_len, C)
        top_k_value = self.select_top_k_fast(value, top_k_index_sort)
        return top_k_query, top_k_value


class SpatialSelection(nn.Module):
    def __init__(self, args):
        super(SpatialSelection, self).__init__()
        self.args = args
        self.attn_qst_query = nn.MultiheadAttention(512, 4, dropout=0.1)

    def soft_attention_select_3d(self, query_feat, kv_feat):
        B, K, M, C = kv_feat.size()
        kv_feat = kv_feat.contiguous().view(B * K, M, C)
        kv_feat = kv_feat.permute(1, 0, 2)
        query_feat = query_feat.unsqueeze(0).repeat(1, K, 1)

        _, temp_weights = self.attn_qst_query(query_feat, kv_feat, kv_feat,
                                              attn_mask=None, key_padding_mask=None)
        temp_weights = temp_weights.contiguous().view(B, K, M)
        return temp_weights

    def find_and_select_top_m_fast(self, patch_weights, modality):
        B, K, M, C = modality.size()
        _, top_m_index_sort = torch.topk(patch_weights, k=self.args.top_m, dim=-1)
        modality = modality.contiguous().view(B * K, M, C)
        top_m_index_sort = top_m_index_sort.contiguous().view(B * K, self.args.top_m)
        modality_new = modality[torch.arange(modality.size(0)).unsqueeze(1), top_m_index_sort]
        modality_new = modality_new.contiguous().view(B, K, self.args.top_m, C)
        return modality_new

    def forward(self, filtered_qv, key):
        patch_weights = self.soft_attention_select_3d(key, filtered_qv)
        visual_patch_top_m = self.find_and_select_top_m_fast(patch_weights, filtered_qv)
        return visual_patch_top_m


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class EmbSimHead(nn.Module):
    def __init__(self):
        super(EmbSimHead, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.metric_fn = MulticlassAccuracy(num_classes=3, average='weighted')

    def calc_loss(self, scores, gt):
        """
        Calculate Cross-Entropy Loss
        :param scores: The unnormalised logits
        :param gt: A list of the true labels
        :return:
        """
        if not isinstance(gt, torch.Tensor):
            gt = torch.LongTensor(gt).to('cuda')
        if len(gt.size()) > 1:
            gt = gt.squeeze(1)

        return self.loss_fn(scores, gt)

    def calc_acc(self, scores, gt):
        """
        Calculate Accuracy
        :param scores: The unnormalised logits
        :param gt: A list of the true labels
        :return:
        """

        if not isinstance(gt, torch.Tensor):
            gt = torch.LongTensor(gt).to('cuda')
        if len(gt.size()) > 1:
            gt = gt.squeeze(1)
        scores = scores.argmax(dim=-1)
        if len(scores.size()) > 1:
            scores = scores.squeeze(1)
        return self.metric_fn(scores, gt)

    def forward(self, suggested_answer, options, gt_answer):
        score_0 = suggested_answer * options[:, 0, :]
        score_0 = score_0.sum(1).unsqueeze(1)
        score_1 = suggested_answer * options[:, 1, :]
        score_1 = score_1.sum(1).unsqueeze(1)
        score_2 = suggested_answer * options[:, 2, :]
        score_2 = score_2.sum(1).unsqueeze(1)
        scores = torch.cat([score_0, score_1, score_2], dim=1)
        if gt_answer is not None:
            loss = self.calc_loss(scores, gt_answer)
            metric = self.calc_acc(scores[:, :3], gt_answer)
            return scores, loss, metric
        return scores, None, None


class Cataphract(nn.Module):

    def __init__(self, args, hidden_size=512):
        super(Cataphract, self).__init__()
        self.args = args
        self.num_layers = args.num_layers  # Layers = 1
        self.fc_aud = nn.Linear(768, hidden_size)
        self.fc_vid = nn.Linear(768, hidden_size)
        self.fc_ocr = nn.Linear(768, hidden_size)

        self.fc_v = nn.Linear(512, hidden_size)
        self.fc_p = nn.Linear(768, hidden_size)

        self.fc_q = nn.Linear(512, hidden_size)
        self.fc_o = nn.Linear(512, hidden_size)
        self.fc_w = nn.Linear(512, hidden_size)

        # modules
        self.temporal_selector = TemporalSelection(args)
        self.spatial_selector = SpatialSelection(args)

        self.CT1 = CrossModalTransformer(args,
                                         CrossBlock(d_model=512, nhead=1, dim_feedforward=512),
                                         num_layers=self.num_layers)
        self.CT2 = CrossModalTransformer(args,
                                         CrossBlock(d_model=512, nhead=1, dim_feedforward=512),
                                         num_layers=self.num_layers)
        self.CT3 = CrossModalTransformer(args,
                                         CrossBlock(d_model=512, nhead=1, dim_feedforward=512),
                                         num_layers=self.num_layers)

        self.ST = UniModalTransformer(args,
                                      Block(d_model=512, nhead=1, dim_feedforward=512),
                                      num_layers=self.num_layers)

        self.time_encoder = PositionalEncoding1D(512)
        self.space_encoder = PositionalEncoding2D(512)

        self.EmbeddingSimilarityOutput = EmbSimHead()
        self.normv = nn.LayerNorm(512)
        self.normp = nn.LayerNorm(512)
        self.cls_n = 4
        self.cls = nn.Parameter(data=torch.randn(size=(self.cls_n, hidden_size), device='cuda'), requires_grad=True)
        self.fc_answer_pred = nn.Linear(512, 512)

    def forward(self, audio, visual, patch, video, ocr, question, qst_word, options, answer=None):
        if audio is not None:
            audio_feat = self.fc_aud(audio)
        if video is not None:
            video_feat = self.fc_vid(video)
        if ocr is not None:
            ocr_feat = self.fc_ocr(ocr)

        visual_feat = self.fc_v(visual)
        patch_feat = self.fc_p(patch)
        cls = self.cls.repeat(visual_feat.size()[0], 1, 1)
        qst_feat = self.fc_q(question).squeeze(-2)
        word_feat = self.fc_w(qst_word).squeeze(-3)
        options_feat = self.fc_o(options)

        ### Step 0: Time Space Encoding ###
        visual_feat = self.normv(visual_feat + self.time_encoder(visual_feat))
        patch_feat = self.normp(patch_feat + self.space_encoder(patch_feat))
        ### Step 1: Filter relevant frames ###
        _, top_k_patches = self.temporal_selector(query=visual_feat, key=qst_feat, value=patch_feat)
        ### Step 2: Filter relevant frame patches ###
        top_km_patches = self.spatial_selector(filtered_qv=top_k_patches, key=qst_feat)
        top_km_patches = top_km_patches.view(-1, self.args.top_k * self.args.top_m, 512)
        ### Step 5. Fusion module ###
        av_fusion_feat = torch.cat(
            (cls, top_km_patches, word_feat), dim=1)
        av_fusion_feat = self.ST(av_fusion_feat)
        av_fusion_feat = av_fusion_feat[:, 0:self.cls_n, :].mean(-2)
        avq_feat = torch.mul(av_fusion_feat, qst_feat)
        avq_feat = F.tanh(avq_feat)
        avq_feat = self.fc_answer_pred(avq_feat)
        ### 7. Answer prediction moudule *************************************************************
        logits, loss, metric = self.EmbeddingSimilarityOutput(avq_feat, options_feat, answer)
        return logits, loss, metric
