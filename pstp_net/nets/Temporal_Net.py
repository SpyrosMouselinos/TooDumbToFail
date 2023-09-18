import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torchmetrics.classification import MulticlassAccuracy
import sys
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, Blip2Config
from transformers.models.blip_2.modeling_blip_2 import Blip2ForConditionalGenerationModelOutput, Blip2QFormerModel, \
    Blip2VisionModel, Blip2PreTrainedModel

sys.path.append('/home/spyros/Desktop/TooDumbToFail')


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
        score_0 = torch.nn.functional.cosine_similarity(suggested_answer, options[:, 0, :], dim=1).unsqueeze(1)
        score_1 = torch.nn.functional.cosine_similarity(suggested_answer, options[:, 1, :], dim=1).unsqueeze(1)
        score_2 = torch.nn.functional.cosine_similarity(suggested_answer, options[:, 2, :], dim=1).unsqueeze(1)
        scores = torch.cat([score_0, score_1, score_2], dim=1)
        if gt_answer is not None:
            loss = self.calc_loss(scores, gt_answer)
            metric = self.calc_acc(scores[:, :3], gt_answer)
            return scores, loss, metric
        return scores, None, None


class TBLIP(Blip2PreTrainedModel):
    config_class = Blip2Config
    main_input_name = "pixel_values"

    def __init__(self, config: Blip2Config):
        super().__init__(config)
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = Blip2QFormerModel(config.qformer_config)

        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        if config.use_decoder_only_language_model:
            language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)
        self.language_model = language_model
        self.post_init()
        self.freeze_lm()

    def freeze_lm(self):
        self.language_model.eval()
        for param in self.language_model.parameters():
            param.requires_grad = False

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _tie_weights(self):
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared

    def _preprocess_accelerate(self):
        if hasattr(self.language_model, "_hf_hook"):
            self.language_model._hf_hook.io_same_device = True

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            question_and_options_tokenized: torch.FloatTensor,
            answer_tokenized=None,
    ):

        return_dict = None
        # step 1: forward the images through the vision encoder,
        image_embeds = pixel_values

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        inputs_embeds = self.language_model.get_input_embeddings()(question_and_options_tokenized)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)


        attention_mask = torch.ones_like(question_and_options_tokenized)
        expected_device = language_model_attention_mask.device
        attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(expected_device)], dim=1)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=return_dict,
        )
        logits = outputs.logits if return_dict else outputs[0]
        loss = None
        # we compute the loss here since we need to take into account the sequence length of the query embeds
        if answer_tokenized is not None:
            answer_tokenized = answer_tokenized.to(logits.device)
            logits = logits[:, -answer_tokenized.size(1):, :]
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = answer_tokenized[..., 1:].contiguous().to(logits.device)

            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="mean")
            loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
        return logits, loss, outputs

    @torch.no_grad()
    def generate(
            self,
            pixel_values: torch.FloatTensor,
            input_ids=None,
            attention_mask=None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()

        batch_size = pixel_values.shape[0]
        image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state

        language_model_inputs = self.language_projection(query_output)
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

        # concatenate query embeddings with prompt embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return outputs
