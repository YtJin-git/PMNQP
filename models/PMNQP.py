import os.path

import torch
import torch.nn as nn
from clip_modules.clip_model import load_clip, QuickGELU
from clip_modules.tokenization_clip import SimpleTokenizer
from models.common import *
from torch.nn.modules.loss import CrossEntropyLoss
import json


class PMNQP(nn.Module):
    def __init__(self, config, dset):
        super(PMNQP, self).__init__()
        self.device = f"cuda:{config.cuda_device}" if torch.cuda.is_available() else "cpu"
        self.clip = load_clip(name=config.clip_arch, context_length=config.context_length, device=self.device)
        self.tokenizer = SimpleTokenizer()
        self.config = config

        allattrs = dset.attrs
        allobj = dset.objs
        classes = [cla.replace(".", " ").lower() for cla in allobj]
        attributes = [attr.replace(".", " ").lower() for attr in allattrs]
        offset = len(attributes)
        self.attributes = attributes
        self.classes = classes
        self.attr_dropout = nn.Dropout(config.attr_dropout)
        self.num_attrs, self.num_objs, self.num_pairs = len(dset.attrs), len(dset.objs), len(dset.pairs)
        self.pairs = dset.pairs
        self.attr_idx = dset.attr2idx
        self.obj_idx = dset.obj2idx
        self.pair_idx = dset.pair2idx

        all_element_words = list(dset.attrs) + list(dset.objs)
        self.attr_obj_displacement = len(dset.attrs)
        self.element_pair_displacement = len(all_element_words)

        self.dict_Obj2IDX = {word: idx for idx, word in enumerate(dset.objs)}
        self.dict_Attr2IDX = {word: idx for idx, word in enumerate(dset.attrs)}

        self.token_ids, self.soft_att_obj, comp_ctx_vectors, attr_ctx_vectors, obj_ctx_vectors = self.construct_soft_prompt()
        self.offset = offset
        self.enable_pos_emb = True
        dtype = self.clip.dtype
        if dtype is None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        self.text_encoder = CustomTextEncoder(self.clip, self.tokenizer, self.dtype)
        # freeze CLIP's parameters
        for p in self.parameters():
            p.requires_grad = False

        # only consider ViT as visual encoder
        assert 'ViT' in config.clip_model

        self.soft_att_obj = nn.Parameter(self.soft_att_obj)
        self.comp_ctx_vectors = nn.Parameter(comp_ctx_vectors).to(self.device)
        self.attr_ctx_vectors = nn.Parameter(attr_ctx_vectors).to(self.device)
        self.obj_ctx_vectors = nn.Parameter(obj_ctx_vectors).to(self.device)
        # for p in self.parameters():
        #     p.requires_grad = False

        self.additional_visual_params = self.add_visual_tunable_params()
        output_dim = self.clip.visual.output_dim

        self.attr_disentangler = Disentangler(output_dim).to(self.device)
        self.obj_disentangler = Disentangler(output_dim).to(self.device)

        # load qualified words
        with open(os.path.join('qualified_words', config.dataset + '.json')) as qualified_file:
            self.qualified_words_dict = json.load(qualified_file)
        self.qualified_attr_list, self.qualified_obj_list = [], []
        for category_name, category in self.qualified_words_dict.items():
            if category_name == "attributes":
                for value in category.values():
                    self.qualified_attr_list.qualified(value)
            elif category_name == "objects":
                for value in category.values():
                    self.qualified_obj_list.qualified(value)
        self.qualified_attr_obj = self.construct_soft_prompt_qualified()

        # primitive memory networks
        if self.config.memorize_text:
            self.attr_memory = torch.zeros(self.num_attrs, self.config.memory_length + 1, self.clip.visual.output_dim,
                                           requires_grad=False).to(self.device)
            self.obj_memory = torch.zeros(self.num_objs, self.config.memory_length + 1, self.clip.visual.output_dim,
                                          requires_grad=False).to(self.device)
        else:
            self.attr_memory = torch.zeros(self.num_attrs, self.config.memory_length, self.clip.visual.output_dim,
                                           requires_grad=False).to(self.device)
            self.obj_memory = torch.zeros(self.num_objs, self.config.memory_length, self.clip.visual.output_dim,
                                          requires_grad=False).to(self.device)

        self.attr_entropy_bank = torch.ones(self.num_attrs, self.config.memory_length, requires_grad=False).to(
            self.device) * 1e9
        self.obj_entropy_bank = torch.ones(self.num_objs, self.config.memory_length, requires_grad=False).to(
            self.device) * 1e9

        self.attr_ptr = torch.zeros(self.num_attrs, dtype=torch.long, requires_grad=False).to(self.device)
        self.obj_ptr = torch.zeros(self.num_objs, dtype=torch.long, requires_grad=False).to(self.device)

        self.mapping_que_a = nn.Linear(self.clip.visual.output_dim, self.clip.visual.output_dim).to(self.device)
        self.mapping_key_a = nn.Linear(self.clip.visual.output_dim, self.clip.visual.output_dim).to(self.device)
        self.mapping_val_a = nn.Linear(self.clip.visual.output_dim, self.clip.visual.output_dim).to(self.device)
        self.mapping_que_o = nn.Linear(self.clip.visual.output_dim, self.clip.visual.output_dim).to(self.device)
        self.mapping_key_o = nn.Linear(self.clip.visual.output_dim, self.clip.visual.output_dim).to(self.device)
        self.mapping_val_o = nn.Linear(self.clip.visual.output_dim, self.clip.visual.output_dim).to(self.device)
        self.mapping_output_a = nn.Linear(self.clip.visual.output_dim, self.clip.visual.output_dim).to(self.device)
        self.mapping_output_o = nn.Linear(self.clip.visual.output_dim, self.clip.visual.output_dim).to(self.device)

        for m in [self.mapping_que_a, self.mapping_key_a, self.mapping_val_a, self.mapping_que_o, self.mapping_key_o,
                  self.mapping_val_o, self.mapping_output_a, self.mapping_output_o]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)


    def construct_soft_prompt_qualified(self):
        tokenized = torch.cat(
            [
                self.tokenizer(tok, context_length=self.config.context_length)
                for tok in self.qualified_attr_list + self.qualified_obj_list
            ]
        )
        orig_token_embedding = self.clip.token_embedding(tokenized.to(self.device))
        soft_qualified_att_obj = torch.zeros(
            (len(self.qualified_attr_list) + len(self.qualified_obj_list), orig_token_embedding.size(-1)),
        )
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            soft_qualified_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        return soft_qualified_att_obj

    def add_visual_tunable_params(self):
        adapter_num = 2 * self.clip.visual.transformer.layers
        params = nn.ModuleList([Adapter(d_model=self.clip.visual.transformer.width,
                                    bottleneck=self.config.adapter_dim,
                                    dropout=self.config.adapter_dropout
                                ) for _ in range(adapter_num)])
        
        return params

    def encode_image(self, x: torch.Tensor):
        return self.encode_image_with_adapter(x)

    def encode_image_with_adapter(self, x: torch.Tensor):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # img_feature = self.clip.visual.transformer(x)
        for i_block in range(self.clip.visual.transformer.layers):
            # MHA
            adapt_x = self.additional_visual_params[i_block](x, add_residual=False, residual=None)
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].attention(
                self.clip.visual.transformer.resblocks[i_block].ln_1(x)
            )
            x = x + adapt_x + residual
            # x = x + residual

            # FFN
            i_adapter = i_block + self.clip.visual.transformer.layers
            adapt_x = self.additional_visual_params[i_adapter](x, add_residual=False, residual=None)
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].mlp(
                self.clip.visual.transformer.resblocks[i_block].ln_2(x)
            )
            x = x + adapt_x + residual
            # x = x + residual

        img_feature = x.permute(1, 0, 2)  # LND -> NLD

        img_feature = self.clip.visual.ln_post(img_feature)
        if self.clip.visual.proj is not None:
            img_feature = img_feature @ self.clip.visual.proj
        return img_feature[:, 0, :], img_feature

    def construct_soft_prompt(self):
        # token_ids indicates the position of [EOS]
        token_ids = self.tokenizer(self.config.prompt_template,
                                   context_length=self.config.context_length).to(self.device)

        tokenized = torch.cat(
            [
                self.tokenizer(tok, context_length=self.config.context_length)
                for tok in self.attributes + self.classes
            ]
        )
        orig_token_embedding = self.clip.token_embedding(tokenized.to(self.device))
        soft_att_obj = torch.zeros(
            (len(self.attributes) + len(self.classes), orig_token_embedding.size(-1)),
        )
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            soft_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        ctx_init = self.config.ctx_init
        assert isinstance(ctx_init, list)
        n_ctx = [len(ctx.split()) for ctx in ctx_init]
        prompt = self.tokenizer(ctx_init,
                                context_length=self.config.context_length).to(self.device)
        with torch.no_grad():
            embedding = self.clip.token_embedding(prompt)

        comp_ctx_vectors = embedding[0, 1: 1 + n_ctx[0], :].to(self.clip.dtype)
        attr_ctx_vectors = embedding[1, 1: 1 + n_ctx[1], :].to(self.clip.dtype)
        obj_ctx_vectors = embedding[2, 1: 1 + n_ctx[2], :].to(self.clip.dtype)

        return token_ids, soft_att_obj, comp_ctx_vectors, attr_ctx_vectors, obj_ctx_vectors

    def encode_text_for_open(self, idx):
        token_tensors = self.construct_token_tensors(idx)
        text_features = []
        for i_element in range(self.token_ids.shape[0]):
            _text_features, _ = self.encode_text(
                self.token_ids[i_element],
                token_tensors[i_element],
                enable_pos_emb=self.enable_pos_emb,
            )

            idx_text_features = _text_features / _text_features.norm(
                dim=-1, keepdim=True
            )
            text_features.append(idx_text_features)
        return text_features

    # add qualified words
    def construct_token_tensors(self, pair_idx, qualified_words=None):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        token_tensor, num_elements = list(), [len(pair_idx), self.offset, len(self.classes), len(self.qualified_attr_list),
                                              len(self.qualified_obj_list)]
        for i_element in range(self.token_ids.shape[0]):
            class_token_ids = self.token_ids[i_element].repeat(num_elements[i_element], 1)
            token_tensor.append(self.clip.token_embedding(
                class_token_ids.to(self.device)
            ).type(self.clip.dtype))
        # 添加qualified token tensor
        token_tensor.append(self.clip.token_embedding(
            self.token_ids[1].repeat(num_elements[3], 1).to(self.device)
        ).type(self.clip.dtype))
        token_tensor.append(self.clip.token_embedding(
            self.token_ids[2].repeat(num_elements[4], 1).to(self.device)
        ).type(self.clip.dtype))

        eos_idx = [int(self.token_ids[i_element].argmax()) for i_element in range(self.token_ids.shape[0])]
        soft_att_obj = self.attr_dropout(self.soft_att_obj).to(self.device)
        # comp
        token_tensor[0][:, eos_idx[0] - 2, :] = soft_att_obj[
            attr_idx
        ].type(self.clip.dtype)
        token_tensor[0][:, eos_idx[0] - 1, :] = soft_att_obj[
            obj_idx + self.offset
        ].type(self.clip.dtype)
        token_tensor[0][
            :, 1 : len(self.comp_ctx_vectors) + 1, :
        ] = self.comp_ctx_vectors.type(self.clip.dtype)

        # attr
        token_tensor[1][:, eos_idx[1] - 1, :] = soft_att_obj[
            :self.offset
        ].type(self.clip.dtype)
        token_tensor[1][
            :, 1 : len(self.attr_ctx_vectors) + 1, :
        ] = self.attr_ctx_vectors.type(self.clip.dtype)

        # obj
        token_tensor[2][:, eos_idx[2] - 1, :] = soft_att_obj[
            self.offset:
        ].type(self.clip.dtype)
        token_tensor[2][
            :, 1 : len(self.obj_ctx_vectors) + 1, :
        ] = self.obj_ctx_vectors.type(self.clip.dtype)

        # qualified soft word tokens ========start
        soft_qualified_att_obj = self.attr_dropout(self.qualified_attr_obj).to(self.device)
        # attr
        token_tensor[3][:, eos_idx[1] - 1, :] = soft_qualified_att_obj[
            :self.offset*5
        ].type(self.clip.dtype)
        # obj
        token_tensor[4][:, eos_idx[2] - 1, :] = soft_qualified_att_obj[
            self.offset*5:
        ].type(self.clip.dtype)
        if hasattr(self.config, 'qualified_soft_prompt_train') and self.config.qualified_soft_prompt_train:
            token_tensor[3][
                :, 1: len(self.attr_ctx_vectors) + 1, :
            ] = self.attr_ctx_vectors.type(self.clip.dtype)
            token_tensor[4][
                :, 1: len(self.obj_ctx_vectors) + 1, :
            ] = self.obj_ctx_vectors.type(self.clip.dtype)
        else:
            token_tensor[3][
                :, 1: len(self.attr_ctx_vectors) + 1, :
            ] = self.attr_ctx_vectors.type(self.clip.dtype).detach()
            token_tensor[4][
                :, 1: len(self.obj_ctx_vectors) + 1, :
            ] = self.obj_ctx_vectors.type(self.clip.dtype).detach()

        # qualified soft word tokens ========end

        return token_tensor

    def encode_text(self, token_ids, token_tensors=None, enable_pos_emb=False):
        return self.text_encoder(token_ids, token_tensors, enable_pos_emb)

    def batch_read_memory(self, query_feats, memory_bank, ptr_bank, mapping_q, mapping_k, mapping_v, mapping_out):
        B, D = query_feats.shape
        C, M_total, D_ = memory_bank.shape
        device = query_feats.device

        M_img = M_total - 1 if self.config.memorize_text else M_total

        with torch.no_grad():
            valid_mask = torch.arange(M_img, device=device).unsqueeze(0) < ptr_bank.view(-1, 1)
            if self.config.memorize_text:
                text_mask = torch.ones(C, 1, dtype=torch.bool, device=device)
                valid_mask = torch.cat([valid_mask, text_mask], dim=1) 
        valid_mask = valid_mask.float()

        q_proj = mapping_q(query_feats)  # [B, D]
        k_proj = mapping_k(memory_bank)  # [C, M_total, D]
        v_proj = mapping_v(memory_bank)  # [C, M_total, D]

        sim = torch.einsum('bd,cmd->bcm', q_proj, k_proj)  # [B, C, M_total]

        weight = torch.exp(-self.config.beta * (1 - sim)) * valid_mask.unsqueeze(0)  # [B, C, M_total]

        weight_sum = weight.sum(dim=2, keepdim=True).clamp(min=1e-6)  # [B, C, 1]
        weight = weight / weight_sum  # softmax over valid memory

        out = torch.einsum('bcm,cmd->bcd', weight, v_proj)  # [B, C, D]

        out = mapping_out(out)
        return out

    def update_memory_batch_diversity(self, feat, text_feat, label, similarity, memory_bank, score_bank, ptr_bank):
        B, D = feat.shape
        M_total = memory_bank.size(1)
        M_img = M_total - 1 if self.config.memorize_text else M_total
        C = memory_bank.size(0)
        device = feat.device

        if self.config.memorize_text:
            memory_bank[:, M_img, :] = text_feat

        mem_full = ptr_bank[label] >= M_img  # [B]

        idx_not_full = (~mem_full).nonzero(as_tuple=False).squeeze(-1)
        if idx_not_full.numel() > 0:
            label_nf = label[idx_not_full]  # [b1]
            ptr_nf = ptr_bank[label_nf]  # [b1]
            feat_nf = feat[idx_not_full]  # [b1, D]
            sim_nf = similarity[idx_not_full]  # [b1]

            score_nf = sim_nf  # [b1]

            memory_bank[label_nf, ptr_nf] = feat_nf
            score_bank[label_nf, ptr_nf] = score_nf
            ptr_bank[label_nf] = ptr_nf + 1

        idx_full = mem_full.nonzero(as_tuple=False).squeeze(-1)
        if idx_full.numel() > 0:
            label_f = label[idx_full]  # [b2]
            feat_f = feat[idx_full]  # [b2, D]
            sim_f = similarity[idx_full]  # [b2]

            mem_entries = memory_bank[label_f, :M_img, :]  # [b2, M, D]
            score_entries = score_bank[label_f, :M_img]  # [b2, M]

            feat_f_norm = F.normalize(feat_f, dim=-1).unsqueeze(1)  # [b2, 1, D]
            mem_entries_norm = F.normalize(mem_entries, dim=-1)  # [b2, M, D]
            sim_matrix = torch.einsum('bnd,bmd->bnm', feat_f_norm, mem_entries_norm).squeeze(1)  # [b2, M]
            mean_sim = sim_matrix.mean(dim=1)  # [b2]

            current_score = sim_f + mean_sim  # [b2]

            replace_idx = torch.argmin(score_entries, dim=1)  # [b2]

            memory_bank[label_f, replace_idx] = feat_f
            score_bank[label_f, replace_idx] = current_score

    def loss_calu(self, predict, target):
        loss_fn = CrossEntropyLoss()
        _, batch_attr, batch_obj, batch_target = target
        batch_attr = batch_attr.to(self.device)
        batch_obj = batch_obj.to(self.device)
        batch_target = batch_target.to(self.device)
        loss_comp = loss_fn(predict['comp_logits'], batch_target)
        loss_attr = loss_fn(predict['attr_logits'], batch_attr)
        loss_obj = loss_fn(predict['obj_logits'], batch_obj)
        loss_attr_qualified = loss_fn(predict['attr_qualified_logits'], batch_attr)
        loss_obj_qualified = loss_fn(predict['obj_qualified_logits'], batch_obj)
        loss_attr_readout = loss_fn(predict['attr_readout_logits'], batch_attr)
        loss_obj_readout = loss_fn(predict['obj_readout_logits'], batch_obj)

        loss = loss_comp * self.config.pair_loss_weight +\
               loss_attr * self.config.attr_loss_weight +\
               loss_obj * self.config.obj_loss_weight +\
               loss_attr_qualified * self.config.attr_qualified_loss_weight +\
               loss_obj_qualified * self.config.obj_qualified_loss_weight + \
               loss_attr_readout * self.config.attr_readout_loss_weight + \
               loss_obj_readout * self.config.obj_readout_loss_weight + \
               predict['loss_attr_memory_contrastive'] * self.config.attr_memory_contrastive_loss_weight + \
               predict['loss_obj_memory_contrastive'] * self.config.obj_memory_contrastive_loss_weight + \
               predict['loss_attr_memory_distill'] * self.config.attr_memory_distill_loss_weight + \
               predict['loss_obj_memory_distill'] * self.config.obj_memory_distill_loss_weight

        return loss

    def logit_infer(self, predict, pairs):
        attr_logits = predict['attr_logits'] * self.config.attr_inference_weight + predict[
            'attr_qualified_logits'] * self.config.attr_qualified_inference_weight + predict[
                          'attr_readout_logits'] * self.config.attr_readout_inference_weight
        obj_logits = predict['obj_logits'] * self.config.obj_inference_weight + predict[
            'obj_qualified_logits'] * self.config.obj_qualified_inference_weight + predict[
                         'obj_readout_logits'] * self.config.obj_readout_inference_weight
        attr_pred = F.softmax(attr_logits, dim=-1)
        obj_pred = F.softmax(obj_logits, dim=-1)
        for i_comp in range(predict['comp_logits'].shape[-1]):
            weighted_attr_pred = 1 if self.config.attr_inference_weight == 0 or self.config.attr_qualified_inference_weight == 0 else attr_pred[:, pairs[i_comp][0]]
            weighted_obj_pred = 1 if self.config.obj_inference_weight == 0 or self.config.obj_qualified_inference_weight == 0 else obj_pred[:, pairs[i_comp][1]]
            predict['comp_logits'][:, i_comp] = predict['comp_logits'][:, i_comp] * self.config.pair_inference_weight + weighted_attr_pred * weighted_obj_pred

        return predict['comp_logits']

    def forward_for_open(self, batch, text_feats):
        batch_img = batch[0].to(self.device)
        b = batch_img.shape[0]
        # l, _ = idx.shape
        batch_img, batch_patch = self.encode_image(batch_img.type(self.clip.dtype))
        batch_img_features = [batch_img, self.attr_disentangler(batch_img), self.obj_disentangler(batch_img)]
        normalized_img_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in batch_img_features]

        logits = list()
        for i_element in range(self.token_ids.shape[0]):
            idx_text_features = text_feats[i_element]

            logits.append(
                torch.einsum(
                    "bd, kd->bk",
                    normalized_img_features[i_element],
                    idx_text_features * self.clip.logit_scale.exp()
            ))
        return logits

    def forward_function(self, batch, pairs_idx, update_ao_memory=False):
        batch_img = batch[0].to(self.device)
        batch_img, batch_patch = self.encode_image(batch_img.type(self.clip.dtype))
        batch_img_features = [batch_img, self.attr_disentangler(batch_img), self.obj_disentangler(batch_img)]
        normalized_img_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in batch_img_features]

        logits = {}
        key = ["comp_logits", "attr_logits", "obj_logits"]
        text_features_list = []
        token_tensors = self.construct_token_tensors(pairs_idx) # c, a, o, qualified_a, qualified_o
        for i_element in range(self.token_ids.shape[0]):
            _text_features, _ = self.encode_text(
                self.token_ids[i_element],
                token_tensors[i_element],
                enable_pos_emb=self.enable_pos_emb,
            )

            idx_text_features = _text_features / _text_features.norm(
                dim=-1, keepdim=True
            )
            text_features_list.append(idx_text_features)
            logits[key[i_element]] = torch.einsum("bd, kd->bk", normalized_img_features[i_element],
                                                  idx_text_features * self.clip.logit_scale.exp())

        # qualified attr logits
        _qualified_text_feat_attr, _ = self.encode_text(
            self.token_ids[1],
            token_tensors[3],
            enable_pos_emb=self.enable_pos_emb,
        )
        qualified_text_feat_attr = _qualified_text_feat_attr / _qualified_text_feat_attr.norm(dim=-1, keepdim=True)
        logits["attr_qualified_logits"] = torch.einsum(
                "bd, ckd->bck",
                normalized_img_features[1],
                qualified_text_feat_attr.view(self.num_attrs,5,-1) * self.clip.logit_scale.exp()
            ).mean(dim=-1)

        # qualified obj logits
        _qualified_text_feat_obj, _ = self.encode_text(
            self.token_ids[2],
            token_tensors[4],
            enable_pos_emb=self.enable_pos_emb,
        )
        qualified_text_feat_obj = _qualified_text_feat_obj / _qualified_text_feat_obj.norm(dim=-1, keepdim=True)
        logits["obj_qualified_logits"] = torch.einsum(
                "bd, ckd->bck",
                normalized_img_features[2],
                qualified_text_feat_obj.view(self.num_objs,5,-1) * self.clip.logit_scale.exp()
            ).mean(dim=-1)

        # primitive memory networks
        logits["loss_attr_memory_contrastive"], logits["loss_attr_memory_distill"], logits[
            "loss_obj_memory_contrastive"], logits["loss_obj_memory_distill"] = 0.0, 0.0, 0.0, 0.0
        if update_ao_memory:
            with torch.no_grad():
                sim_attr = logits["attr_logits"][torch.arange(batch[0].shape[0]),batch[1]]
                sim_obj = logits["obj_logits"][torch.arange(batch[0].shape[0]),batch[2]]
                self.update_memory_batch_diversity(normalized_img_features[1], text_features_list[1], batch[1].to(self.device),
                                         sim_attr, self.attr_memory, self.attr_entropy_bank, self.attr_ptr)
                self.update_memory_batch_diversity(normalized_img_features[2], text_features_list[2], batch[2].to(self.device),
                                         sim_obj, self.obj_memory, self.obj_entropy_bank, self.obj_ptr)

        attr_img_feat_read = self.batch_read_memory(normalized_img_features[1], self.attr_memory, self.attr_ptr,
                                                    self.mapping_que_a, self.mapping_key_a, self.mapping_val_a,
                                                    self.mapping_output_a)  # [B, num_attrs, D]
        obj_img_feat_read = self.batch_read_memory(normalized_img_features[2], self.obj_memory, self.obj_ptr,
                                                   self.mapping_que_o, self.mapping_key_o, self.mapping_val_o,
                                                   self.mapping_output_o)  # [B, num_objs, D]
        
        logits["attr_readout_logits"] = torch.einsum(
                "bd, bkd->bk",
                normalized_img_features[1],
                attr_img_feat_read * self.clip.logit_scale.exp()
            )
        logits["obj_readout_logits"] = torch.einsum(
            "bd, bkd->bk",
            normalized_img_features[2],
            obj_img_feat_read * self.clip.logit_scale.exp()
        )
        return logits

    def forward(self, batch, pairs_idx):
        if self.training:
            output = self.forward_function(batch, pairs_idx, update_ao_memory=True)
        else:
            with torch.no_grad():
                output = self.forward_function(batch, pairs_idx, update_ao_memory=False)
        return output