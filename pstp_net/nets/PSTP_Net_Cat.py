import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torchmetrics.classification import MulticlassAccuracy


class AVClipAttn(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(AVClipAttn, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cm_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

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


class TemporalSegmentSelection(nn.Module):

    def __init__(self, args):
        super(TemporalSegmentSelection, self).__init__()

        self.args = args

        # question as query on audio-visual clip
        self.attn_qst_query = nn.MultiheadAttention(512, 4, dropout=0.1)
        self.qst_query_linear1 = nn.Linear(512, 512)
        self.qst_query_relu = nn.ReLU()
        self.qst_query_dropout1 = nn.Dropout(0.1)
        self.qst_query_linear2 = nn.Linear(512, 512)
        self.qst_query_dropout2 = nn.Dropout(0.1)
        self.qst_query_visual_norm = nn.LayerNorm(512)

        # a-v attn
        self.clip_attn = AVClipAttn(d_model=512, nhead=1)
        self.clip_tanh = nn.Tanh()
        self.clip_fc = nn.Linear(1024, 512)

    ### audio-visual temporal attention
    def SegmentTempAttn(self, audios_feat_input, visual_feat_input):

        B, T, C = audios_feat_input.size()
        clip_len = int(T / self.args.segs)

        audio_feat = audios_feat_input.view(B, C, self.args.segs, clip_len)
        visual_feat = visual_feat_input.view(B, C, self.args.segs, clip_len)

        audio_feat = audio_feat.permute(0, 2, 3, 1)
        visual_feat = visual_feat.permute(0, 2, 3, 1)

        audio_feat = audio_feat.contiguous().view(B * self.args.segs, clip_len, C)
        visual_feat = visual_feat.contiguous().view(B * self.args.segs, clip_len, C)

        audio_clip_attn = self.clip_attn(audio_feat, visual_feat)
        visual_clip_attn = self.clip_attn(visual_feat, audio_feat)

        audio_clip_attn = audio_clip_attn.contiguous().view(B, T, C)
        visual_clip_attn = visual_clip_attn.contiguous().view(B, T, C)

        return audio_clip_attn, visual_clip_attn

    def AVClipGen(self, audios_input, visual_input, clip_len):

        audio_feat = audios_input.permute(0, 2, 1).unsqueeze(-1)
        visual_feat = visual_input.permute(0, 2, 1).unsqueeze(-1)

        audio_feat = F.avg_pool2d(audio_feat, (clip_len, 1)).squeeze(-1).permute(0, 2, 1)
        visual_feat = F.avg_pool2d(visual_feat, (clip_len, 1)).squeeze(-1).permute(0, 2, 1)

        av_clip_fusion = torch.cat([audio_feat, visual_feat], dim=-1)
        av_clip_fusion = self.clip_tanh(av_clip_fusion)
        av_clip_fusion = self.clip_fc(av_clip_fusion)

        return av_clip_fusion

    def QstQueryClipAttn(self, query_feat, kv_feat):

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

    def SelectTopK(self, temp_weights, audio_input, visual_input, clip_len, C):
        '''
            Function: Select top-k key temporal segments
        '''

        sort_index = torch.argsort(temp_weights, dim=-1)
        top_k_index = sort_index[:, :, -self.args.top_k:]

        top_k_index_sort, indices = torch.sort(top_k_index)
        top_k_index_sort = top_k_index_sort.cpu().numpy()

        batch_num = int(top_k_index_sort.shape[0])
        output_audio = torch.zeros(1, self.args.top_k * clip_len, C).cuda()
        output_visual = torch.zeros(1, self.args.top_k * clip_len, C).cuda()

        for batch_idx in range(batch_num):
            first_start_t = top_k_index_sort[batch_idx][-1][0] * clip_len  # start time in a batch
            first_end_t = first_start_t + clip_len  # end time in a batch
            audio_topk_feat = audio_input[batch_idx, first_start_t:first_end_t, :]
            visual_topk_feat = visual_input[batch_idx, first_start_t:first_end_t, :]

            for i in range(1, self.args.top_k):
                start_t = top_k_index_sort[batch_idx][-1][i] * clip_len
                end_t = start_t + clip_len
                current_audio_selet_feat = audio_input[batch_idx, start_t:end_t, :]
                current_visual_selet_feat = visual_input[batch_idx, start_t:end_t, :]
                audio_topk_feat = torch.cat((audio_topk_feat, current_audio_selet_feat), dim=-2)
                visual_topk_feat = torch.cat((visual_topk_feat, current_visual_selet_feat), dim=-2)

            audio_topk_feat = audio_topk_feat.unsqueeze(0)
            visual_topk_feat = visual_topk_feat.unsqueeze(0)
            output_audio = torch.cat((output_audio, audio_topk_feat), dim=0)
            output_visual = torch.cat((output_visual, visual_topk_feat), dim=0)

        # delete zero tensor
        output_audio = output_audio[1:, :, :]
        output_visual = output_visual[1:, :, :]

        return output_audio, output_visual, top_k_index_sort

    def forward(self, audio_input, visual_input, qst_input):

        B, T, C = audio_input.size()
        clip_len = int(T / self.args.segs)

        temp_clip_feat = self.AVClipGen(audio_input, visual_input, clip_len)
        temp_clip_attn_feat, temp_weights = self.QstQueryClipAttn(qst_input, temp_clip_feat)

        output_audio, output_visual, top_k_index_sort = self.SelectTopK(temp_weights, audio_input, visual_input,
                                                                        clip_len, C)

        return output_audio, output_visual, top_k_index_sort


class SpatioRegionSelection(nn.Module):

    def __init__(self, args, hidden_size=512):
        super(SpatioRegionSelection, self).__init__()

        self.args = args

        # question as query on audio-visual clip
        self.attn_qst_query = nn.MultiheadAttention(512, 4, dropout=0.1)
        self.qst_query_linear1 = nn.Linear(512, 512)
        self.qst_query_relu = nn.ReLU()
        self.qst_query_dropout1 = nn.Dropout(0.1)
        self.qst_query_linear2 = nn.Linear(512, 512)
        self.qst_query_dropout2 = nn.Dropout(0.1)
        self.qst_query_visual_norm = nn.LayerNorm(512)

        self.patch_fc = nn.Linear(512, 512)
        self.patch_relu = nn.ReLU()

    # select top-k frames
    def TopKSegs(self, audio_feat, patch_feat, top_k_index_sort):

        # patch feat: [B, T, N, C]
        B, T, N, C = patch_feat.size()

        clip_len = int(T / self.args.segs)
        frames_nums = clip_len * self.args.top_k  # selected frames numbers

        audio_select = torch.zeros(B, frames_nums, C).cuda()
        patch_select = torch.zeros(B, frames_nums, N, C).cuda()

        for batch_idx in range(B):
            for k_idx in range(self.args.top_k):
                T_start = int(top_k_index_sort[batch_idx, :, k_idx]) * clip_len  # select start frame in a batch
                T_end = T_start + clip_len

                S_start = k_idx * clip_len  # S: Select, belong one temporal clip
                S_end = (k_idx + 1) * clip_len

                patch_select[batch_idx, S_start:S_end, :, :] = patch_feat[batch_idx, T_start:T_end, :, :]
                audio_select[batch_idx, S_start:S_end, :] = audio_feat[batch_idx, T_start:T_end, :]

        return audio_select, patch_select

    def QstQueryPatchAttn(self, query_feat, visual_patch_top_k):

        # input visual: [B, T, N, C]
        B, T, N, C = visual_patch_top_k.size()
        patch_top_k_feat = torch.zeros(B, T, C).cuda()
        patch_top_k_weights = torch.zeros(B, T, N).cuda()

        query_feat = query_feat.unsqueeze(0)
        for T_idx in range(T):
            frames_idx = visual_patch_top_k[:, T_idx, :, :]  # get patches, [B, N, C]

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

    def SelectTopM(self, patch_weights, visual_patch_top_k, B, T, N, C):
        '''
            Function: select top-M key regions
        '''

        # B, T, N, C = visual_patch_top_k.size()
        sort_index = torch.argsort(patch_weights, dim=-1)
        top_m_index = sort_index[:, :, -self.args.top_m:]

        top_m_index_sort, indices = torch.sort(top_m_index)
        top_m_index_sort = top_m_index_sort.cpu().numpy()

        output_visual = torch.zeros(B, T, self.args.top_m, C).cuda()

        for batch_idx in range(B):
            for frame_idx in range(T):
                for patch_id in top_m_index_sort.tolist()[0][0]:
                    output_visual[batch_idx, frame_idx, :, :] = visual_patch_top_k[batch_idx, frame_idx, patch_id, :]

        visual_patch_top_m = output_visual

        return visual_patch_top_m, top_m_index_sort

    def forward(self, audio_feat, patch_feat, qst_feat, top_k_index_sort):

        # output: [B, top-k frames, C], [B, top-k frames, N, C]
        audio_top_k, visual_patch_top_k = self.TopKSegs(audio_feat, patch_feat, top_k_index_sort)

        # compute patch's weights
        # output: [B, T, C], [B, T, N], T: frames numbers, N: patch numbers
        visual_spatial_feat, patch_weights = self.QstQueryPatchAttn(qst_feat, visual_patch_top_k)

        # output: [B, T, top-M regions, C], T: frames numbers,
        B, T, N, C = visual_patch_top_k.size()
        visual_patch_top_m, top_m_idx_sort = self.SelectTopM(patch_weights, visual_patch_top_k, B, T, N, C)
        visual_patch_feat = visual_patch_top_m.view(B, T * self.args.top_m, C)

        return audio_top_k, visual_patch_top_m, visual_patch_feat


class AVHanLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(AVHanLayer, self).__init__()

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

        self.activation = nn.ReLU()

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


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class GlobalLocalPrecption(nn.Module):

    def __init__(self, args, encoder_layer, num_layers, norm=None):
        super(GlobalLocalPrecption, self).__init__()

        self.args = args

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.norm = norm

    def forward(self, src_a, src_v, mask=None, src_key_padding_mask=None):

        audio_output = src_a
        visual_output = src_v

        for i in range(self.num_layers):
            src_a = self.layers[i](src_a, src_v, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            src_v = self.layers[i](src_v, src_a, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            audio_output = self.norm1(src_a)
            visual_output = self.norm2(src_v)

        return audio_output, visual_output


class GlobalHanLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(GlobalHanLayer, self).__init__()

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

        self.activation = nn.ReLU()

    def forward(self, src_q, src_mask=None, src_key_padding_mask=None):
        src_q = src_q.permute(1, 0, 2)
        src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src_q = src_q + self.dropout12(src2)
        src_q = self.norm1(src_q)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
        src_q = src_q + self.dropout2(src2)
        src_q = self.norm2(src_q)
        return src_q.permute(1, 0, 2)


class GlobalSelfAttn(nn.Module):

    def __init__(self, args, encoder_layer, num_layers, norm=None):
        super(GlobalSelfAttn, self).__init__()

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm2 = nn.LayerNorm(512)
        self.norm = norm

    def forward(self, src_v, mask=None, src_key_padding_mask=None):

        visual_output = src_v

        for i in range(self.num_layers):
            visual_output = self.layers[i](src_v, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            visual_output = self.norm2(visual_output)

        return visual_output


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
        self.TempSegsSelect_Module = TemporalSegmentSelection(args)
        self.SpatRegsSelect_Module = SpatioRegionSelection(args)

        self.GlobalLocal_Module = GlobalLocalPrecption(args,
                                                       AVHanLayer(d_model=512, nhead=1, dim_feedforward=512),
                                                       num_layers=self.num_layers)

        self.GlobalSelf_Module = GlobalSelfAttn(args,
                                                GlobalHanLayer(d_model=512, nhead=1, dim_feedforward=512),
                                                num_layers=self.num_layers)

        self.EmbeddingSimilarityOutput = EmbSimHead()

        self.tanh = nn.Tanh()
        self.fc_answer_pred = nn.Linear(512, 42)

    def forward(self, audio, visual, patch, video, ocr, question, qst_word, options, answer=None):
        ### 1. features input
        # audio: [B, T, C]
        # visual: [B, T, C]
        # patch: [B, T, N, C]
        # video: [B,T,C]
        # ocr: [B,T,C]
        # question: [B, C]
        # qst_word:  [B, 77, C]
        # options: [B,3,C]
        # answer: [B,1,C]

        audio_feat = self.fc_a(audio).unsqueeze(1).repeat(1, 60, 1)  # TODO: REMOVE THIS
        visual_feat = self.fc_v(visual)  # [B, T, C]
        patch_feat = self.fc_p(patch)  # [B, T, N, C]
        # video_feat = self.fc_vid(video) # [B,T,C]
        # ocr_feat = self.fc_ocr(ocr) # [B,T,C]
        qst_feat = self.fc_q(question).squeeze(-2)  # [B, C]
        word_feat = self.fc_word(qst_word).squeeze(-3)  # [B, 77, C]
        options_feat = self.fc_o(options)

        ### 2. Temporal segment selection module *******************************************************
        # audio and visual output: [B, top_k * T/segs, C], eg: [B, 2*5, C]
        # top_k_index_sort: selected top_k index
        audio_feat_tm, visual_feat_tm, top_k_index_sort = self.TempSegsSelect_Module(audio_feat, visual_feat,
                                                                                     qst_feat)

        ### 3. Spatial regions selection module ********************************************************
        # audio_top_k: [B, top_k * T/segs, C]
        # visual_patch_feat: [B, top_k * T/segs, C]
        # visual_patch_feat_sm: [B, top_k * T/segs * select_patch_numbers, C]
        audio_top_k, visual_patch_feat_sm, visual_patch_feat = self.SpatRegsSelect_Module(audio_feat,
                                                                                          patch_feat,
                                                                                          qst_feat,
                                                                                          top_k_index_sort)
        ### 4. Global-local perception module **********************************************************
        audio_feat_gl, visual_feat_gl = self.GlobalLocal_Module(audio_feat, visual_feat)

        ### 5. Global-local perception module 2 **********************************************************

        ### 6. Fusion module **************************************************************************
        visual_feat_fusion = torch.cat((visual_patch_feat, visual_feat_gl), dim=1)
        audio_feat_fusion = torch.cat((audio_feat_tm, audio_feat_gl), dim=1)

        fusion_feat = torch.cat((audio_feat_fusion, visual_feat_fusion, word_feat), dim=1)

        av_fusion_feat = self.GlobalSelf_Module(fusion_feat)
        av_fusion_feat = av_fusion_feat.mean(dim=-2)

        avq_feat = torch.mul(av_fusion_feat, qst_feat)  # [batch_size, embed_size]
        avq_feat = self.tanh(avq_feat)

        ### 7. Answer prediction moudule *************************************************************
        # answer_pred = self.fc_answer_pred(avq_feat)  # [batch_size, ans_vocab_size=42]
        logits, loss, metric = self.EmbeddingSimilarityOutput(avq_feat, options_feat, answer)
        return logits, loss, metric
