import os.path

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, Blip2Config
from transformers.models.blip_2.modeling_blip_2 import Blip2QFormerModel, \
    Blip2PreTrainedModel


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

    def save_model(self, path):
        important_keys = ['query_tokens', 'qformer', 'language_projection']
        new_state_dict = {k: v for k, v in self.state_dict().items() if
                          important_keys[0] in k or important_keys[1] in k or important_keys[2] in k}
        pre_path = '/'.join(path.split('/')[:-1])
        if os.path.exists(pre_path):
            torch.save(new_state_dict, path)
        else:
            os.mkdir(pre_path)
            torch.save(new_state_dict, path)
        return

    def load_model(self, path):
        data = torch.load(path)
        self.load_state_dict(data, strict=False)
        return

    def freeze_lm(self):
        self.language_model.eval()
        for param in self.language_model.parameters():
            param.requires_grad = False

    def symbolic_acc(self, logits, labels, dim=1):
        argmaxed_logits = torch.argmax(logits, dim=dim)  # BS, 1
        acc = torch.sum(1.0 * (argmaxed_logits == labels)) / labels.size()[0]
        return acc

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
            image_feats: torch.FloatTensor,
            question_and_options_tokenized: torch.FloatTensor,
            answer_tokenized=None,
    ):

        return_dict = None
        flag = False
        if len(image_feats.size()) == 4:
            # images are of size BS, 60, 677, 1408
            B, L, X, Y = image_feats.size()
            image_feats = image_feats.view(B * L, X, Y).contiguous()
            flag = True

        image_attention_mask = torch.ones(image_feats.size()[:-1], dtype=torch.long, device=image_feats.device)
        query_tokens = self.query_tokens.expand(image_feats.shape[0], -1, -1)

        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_feats,
            encoder_attention_mask=image_attention_mask,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]
        if flag:
            # Queries are of size B * L, Z, V
            _, Z, V = query_output.size()
            query_output = query_output.view(B, L, Z, V).contiguous()
            query_output = query_output.mean(dim=1)

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
            # Last Logit is the answer
            logits = logits[:, -1, :]
            labels = answer_tokenized.contiguous().to(logits.device)

            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="mean")
            metric_fct = self.symbolic_acc
            loss = loss_fct(logits.view(-1, self.config.text_config.vocab_size), labels.view(-1))
            metric = metric_fct(logits.view(-1, self.config.text_config.vocab_size), labels.view(-1))
        return logits, loss, metric

    @torch.no_grad()
    def answer(
            self,
            image_feats: torch.FloatTensor,
            question_and_options_tokenized: torch.FloatTensor):

        return_dict = None
        flag = False
        if len(image_feats.size()) == 4:
            # images are of size BS, 60, 677, 1408
            B, L, X, Y = image_feats.size()
            image_feats = image_feats.view(B * L, X, Y).contiguous()
            flag = True

        image_attention_mask = torch.ones(image_feats.size()[:-1], dtype=torch.long, device=image_feats.device)
        query_tokens = self.query_tokens.expand(image_feats.shape[0], -1, -1)

        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_feats,
            encoder_attention_mask=image_attention_mask,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]
        if flag:
            # Queries are of size B * L, Z, V
            _, Z, V = query_output.size()
            query_output = query_output.view(B, L, Z, V).contiguous()
            query_output = query_output.mean(dim=1)

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
        logits = logits[:, -1, :]
        # A prob - location 250
        prob_a = logits[:, 250]
        # B prob - location 387
        prob_b = logits[:, 387]
        # C prob - location 347
        prob_c = logits[:, 347]

        probs = torch.stack([prob_a, prob_b, prob_c], dim=1)
        return torch.argmax(probs, dim=1).detach().cpu().numpy()
