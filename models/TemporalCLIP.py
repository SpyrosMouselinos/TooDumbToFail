import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torchmetrics.classification import MulticlassAccuracy


class Block(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(Block, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

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
        self.qst_query_linear1 = nn.Linear(512, 512)
        self.qst_query_relu = nn.ReLU()
        self.qst_query_dropout1 = nn.Dropout(0.1)
        self.qst_query_linear2 = nn.Linear(512, 512)
        self.qst_query_dropout2 = nn.Dropout(0.1)
        self.qst_query_visual_norm = nn.LayerNorm(512)

    def soft_attention_select_2d(self, query_feat, kv_feat):
        kv_feat = kv_feat.permute(1, 0, 2)
        query_feat = query_feat.unsqueeze(0)
        attn_feat, temp_weights = self.attn_qst_query(query_feat, kv_feat, kv_feat,
                                                      attn_mask=None, key_padding_mask=None)
        attn_feat = attn_feat.squeeze(0)
        src = self.qst_query_linear1(attn_feat)
        src = self.qst_query_relu(src)
        src = self.qst_query_dropout1(src)
        src = self.qst_query_linear2(src)
        src = self.qst_query_dropout2(src)

        attn = attn_feat + src
        attn = self.qst_query_visual_norm(attn)

        return attn, temp_weights

    def find_top_k(self, temp_weights, modality, clip_len, C):
        sort_index = torch.argsort(temp_weights, dim=-1)
        top_k_index = sort_index[:, :, -self.args.top_k:]
        top_k_index_sort, indices = torch.sort(top_k_index)
        top_k_index_sort = top_k_index_sort.cpu().numpy()
        batch_num = int(top_k_index_sort.shape[0])
        output_visual = torch.zeros(1, self.args.top_k * clip_len, C).cuda()

        for batch_idx in range(batch_num):
            first_start_t = top_k_index_sort[batch_idx][-1][0] * clip_len
            first_end_t = first_start_t + clip_len
            visual_topk_feat = modality[batch_idx, first_start_t:first_end_t, :]
            for i in range(1, self.args.top_k):
                start_t = top_k_index_sort[batch_idx][-1][i] * clip_len
                end_t = start_t + clip_len
                current_visual_selet_feat = modality[batch_idx, start_t:end_t, :]
                visual_topk_feat = torch.cat((visual_topk_feat, current_visual_selet_feat), dim=-2)
            visual_topk_feat = visual_topk_feat.unsqueeze(0)
            output_visual = torch.cat((output_visual, visual_topk_feat), dim=0)
        output_visual = output_visual[1:, :, :]
        return output_visual, top_k_index_sort

    def select_top_k(self, patch_feat, top_k_index_sort):
        B, T, N, C = patch_feat.size()
        clip_len = int(T / self.args.segs)
        frames_nums = clip_len * self.args.top_k  # selected frames numbers
        patch_select = torch.zeros(B, frames_nums, N, C).cuda()
        for batch_idx in range(B):
            for k_idx in range(self.args.top_k):
                T_start = int(top_k_index_sort[batch_idx, :, k_idx]) * clip_len  # select start frame in a batch
                T_end = T_start + clip_len
                S_start = k_idx * clip_len  # S: Select, belong one temporal clip
                S_end = (k_idx + 1) * clip_len
                patch_select[batch_idx, S_start:S_end, :, :] = patch_feat[batch_idx, T_start:T_end, :, :]
        return patch_select

    def forward(self, query, key, value):
        B, T, C = query.size()
        clip_len = int(T / self.args.segs)
        temp_clip_attn_feat, temp_weights = self.soft_attention_select_2d(key, query)
        output, top_k_index_sort = self.find_top_k(temp_weights, query,
                                                   clip_len, C)
        visual_patch_top_k = self.select_top_k(value, top_k_index_sort)
        return visual_patch_top_k


class SpatialSelection(nn.Module):
    def __init__(self, args):
        super(SpatialSelection, self).__init__()
        self.args = args
        self.attn_qst_query = nn.MultiheadAttention(512, 4, dropout=0.1)
        self.qst_query_linear1 = nn.Linear(512, 512)
        self.qst_query_relu = nn.ReLU()
        self.qst_query_dropout1 = nn.Dropout(0.1)
        self.qst_query_linear2 = nn.Linear(512, 512)
        self.qst_query_dropout2 = nn.Dropout(0.1)
        self.qst_query_visual_norm = nn.LayerNorm(512)

        self.patch_fc = nn.Linear(512, 512)
        self.patch_relu = nn.ReLU()

    def soft_attention_select_3d(self, query_feat, modality):
        B, T, N, C = modality.size()
        patch_top_k_feat = torch.zeros(B, T, C).cuda()
        patch_top_k_weights = torch.zeros(B, T, N).cuda()

        query_feat = query_feat.unsqueeze(0)
        for T_idx in range(T):
            frames_idx = modality[:, T_idx, :, :]  # get patches, [B, N, C]

            kv_feat = frames_idx.permute(1, 0, 2)

            # output: [1, B, C] [B, 1, N]
            attn_feat, frames_patch_weights = self.attn_qst_query(query_feat, kv_feat, kv_feat,
                                                                  attn_mask=None, key_padding_mask=None)
            # attn_feat = attn_feat.squeeze(0)
            src = self.qst_query_linear1(attn_feat)
            src = self.qst_query_relu(src)
            src = self.qst_query_dropout1(src)
            src = self.qst_query_linear2(src)
            src = self.qst_query_dropout2(src)

            attn = attn_feat + src
            attn = self.qst_query_visual_norm(attn)

            patch_top_k_feat[:, T_idx, :] = attn[:, 0, :]  # [B, T, C]
            patch_top_k_weights[:, T_idx, :] = frames_patch_weights[:, 0, :]  # [B, T, N]

        return patch_top_k_feat, patch_top_k_weights

    def find_top_m(self, patch_weights, modality):
        B, T, N, C = modality.size()
        sort_index = torch.argsort(patch_weights, dim=-1)
        top_m_index = sort_index[:, :, -self.args.top_m:]

        top_m_index_sort, indices = torch.sort(top_m_index)
        top_m_index_sort = top_m_index_sort.cpu().numpy()

        output = torch.zeros(B, T, self.args.top_m, C).cuda()

        for batch_idx in range(B):
            for frame_idx in range(T):
                for patch_id in top_m_index_sort.tolist()[0][0]:
                    output[batch_idx, frame_idx, :, :] = modality[batch_idx, frame_idx, patch_id, :]
        return output, top_m_index_sort

    def forward(self, visual_patch_top_k, qst_feat):
        visual_spatial_feat, patch_weights = self.soft_attention_select_3d(qst_feat, visual_patch_top_k)
        B, T, N, C = visual_patch_top_k.size()
        visual_patch_top_m, top_m_idx_sort = self.find_top_m(patch_weights, visual_patch_top_k)
        visual_patch_feat = visual_patch_top_m.view(B, T * self.args.top_m, C)
        return visual_patch_top_m, visual_patch_feat


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


class TemporalNet(nn.Module):

    def __init__(self, args, hidden_size=512):
        super(TemporalNet, self).__init__()
        self.args = args
        self.num_layers = args.num_layers  # Layers = 1
        self.fc_a = nn.Linear(768, hidden_size)
        self.fc_v = nn.Linear(512, hidden_size)
        self.fc_p = nn.Linear(768, hidden_size)
        self.fc_vid = nn.Linear(768, hidden_size)
        self.fc_ocr = nn.Linear(768, hidden_size)
        self.fc_word = nn.Linear(512, hidden_size)
        self.fc_q = nn.Linear(512, hidden_size)
        self.fc_o = nn.Linear(512, hidden_size)
        self.relu = nn.ReLU()

        # modules
        self.TempSegsSelect_Module = TemporalSelection(args)
        self.SpatRegsSelect_Module = SpatialSelection(args)

        self.GlobalLocal_Module_q_p = CrossModalTransformer(args,
                                                           CrossBlock(d_model=512, nhead=1, dim_feedforward=512),
                                                           num_layers=self.num_layers)
        self.GlobalSelf_Module = UniModalTransformer(args,
                                                Block(d_model=512, nhead=1, dim_feedforward=512),
                                                num_layers=self.num_layers)

        self.EmbeddingSimilarityOutput = EmbSimHead()
        self.tanh = nn.Tanh()

    def forward(self, audio, visual, patch, video, ocr, question, qst_word, options, answer=None):
        audio_feat = self.fc_a(audio)
        visual_feat = self.fc_v(visual)
        patch_feat = self.fc_p(patch)
        video_feat = self.fc_vid(video)
        ocr_feat = self.fc_ocr(ocr)
        qst_feat = self.fc_q(question).squeeze(-2)
        word_feat = self.fc_word(qst_word).squeeze(-3)
        options_feat = self.fc_o(options)

        ### 2. Temporal segment selection module *******************************************************
        _, top_k_index_sort = self.TempSegsSelect_Module(visual_feat, qst_feat)
        ### 3. Spatial regions selection module ********************************************************
        _, visual_patch_feat = self.SpatRegsSelect_Module(
            patch_feat,
            qst_feat,
            top_k_index_sort)
        ### 4. Global-local question visual perception module **********************************************************
        qst_p, visual_p = self.GlobalLocal_Module_q_p(word_feat, visual_feat)

        ### 6. Fusion module **************************************************************************
        visual_feat_fusion = torch.cat((visual_patch_feat, visual_p, video_feat.unsqueeze(1), ocr_feat.unsqueeze(1)),
                                       dim=1)
        audio_feat_fusion = audio_feat.unsqueeze(1)
        qst_feat_fusion = word_feat + qst_p

        fusion_feat = torch.cat((visual_feat_fusion, audio_feat_fusion, qst_feat_fusion), dim=1)

        av_fusion_feat = self.GlobalSelf_Module(fusion_feat)
        av_fusion_feat = av_fusion_feat.mean(dim=-2)

        avq_feat = torch.mul(av_fusion_feat, qst_feat)

        ### 7. Answer prediction moudule *************************************************************
        logits, loss, metric = self.EmbeddingSimilarityOutput(avq_feat, options_feat, answer)
        return logits, loss, metric
