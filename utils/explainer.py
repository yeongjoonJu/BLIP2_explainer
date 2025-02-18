import math
import torch
import torch.nn.functional as F
from utils.hook import Hook
from utils.ops import get_image_understanding
from utils.visualization import normalize_last_dim, normalize, show_cam_on_image
from utils.base_explainer import *


class AttributionExplainer(BaseExplainerBLIP2):
    def explain(self, samples, decoder_input_ids, target_index=None):
        attn_hooks = [Hook(self.model.visual_encoder.blocks[l].attn.attn_drop) for l in range(len(self.model.visual_encoder.blocks))]
        
        logits, image_embeds, query_output, t5_output = self.predict(samples, decoder_input_ids)
        
        if target_index==None:
            target_index = torch.argmax(logits, dim=1)[0].item()
        
        one_hot = torch.zeros_like(logits).to(logits.device)
        one_hot[0, target_index] = 1.
        one_hot_vector = one_hot.clone().detach()
        one_hot.requires_grad_(True)
        one_hot = torch.sum(one_hot.to(logits.device)*logits)       
        
        self.model.zero_grad()
        
        # get attentions and gradients of ViT-G through hook 
        vit_attentions = []
        vit_gradients = []
        for hook in attn_hooks:
            attention = hook.output
            gradients = torch.autograd.grad(one_hot, [attention], retain_graph=True)[0]
            vit_attentions.append(attention)
            vit_gradients.append(gradients)
            
        q_self_grads  = get_gradients_across_layers(query_output.attentions, one_hot)
        q_cross_grads = get_gradients_across_layers(query_output.cross_attentions, one_hot)
                
        t5_enc_attns = t5_output.encoder_attentions
        t5_dec_attns = t5_output.decoder_attentions
        t5_cross_attns = t5_output.cross_attentions
        t5_enc_grads  = get_gradients_across_layers(t5_enc_attns, one_hot)
        t5_dec_grads  = get_gradients_across_layers(t5_dec_attns, one_hot)
        t5_cross_grads= get_gradients_across_layers(t5_cross_attns, one_hot)
        
        # Get relevance maps
        R_ii = get_relevance_map_for_self_attention(vit_attentions, vit_gradients, \
                                                    device=attention.device)
        R_qq, R_qi = get_relevance_map_for_cross_attention(
            query_output.attentions, q_self_grads,
            query_output.cross_attentions, q_cross_grads,
            R_ii
        )
        R_ee = get_relevance_map_for_self_attention(t5_enc_attns, t5_enc_grads, \
                                                    device=attention.device, R_ctx=R_qq)
        R_dd, R_de = get_relevance_map_for_cross_attention(
            t5_dec_attns, t5_dec_grads,
            t5_cross_attns, t5_cross_grads,
            R_ee
        )
        
        relevance_maps = {"ii":R_ii, "qq":R_qq, "qi":R_qi, "ee":R_ee, "dd":R_dd, "de":R_de}
        # simmaps = get_image_understanding(image_embeds)
        
        return relevance_maps#, simmaps
    

class LocatingExplainer(BaseExplainerBLIP2):
    def perturbation(self, state, sigma=3.0):
        std = state.std(dim=-1,keepdim=True)
        noise = torch.normal(torch.zeros_like(state), std*sigma)
        
        return noise
    
    def get_text_embedding(self):
        if self.model_type == "t5":
            return self.model.t5_model.encoder.embed_tokens
        else:
            return self.model.opt_model.decoder.embed_tokens
        
    def deletion(self, visual_tokens, visual_atts,\
                text_embeds, text_atts, decoder_embeds, target_index):
        
        B, S, D = visual_tokens.shape
        batch_vt = []
        batch_at = []
        for p in range(S):
            perturbed_visual_tokens = visual_tokens.detach().clone()
            perturbed_visual_tokens = torch.cat([perturbed_visual_tokens[:,:p], perturbed_visual_tokens[:,p+1:]], dim=1)
            perturbed_visual_atts = torch.cat([visual_atts[:,:p], visual_atts[:,p+1:]], dim=1)
            encoder_atts = torch.cat([perturbed_visual_atts, text_atts], dim=1)
            batch_vt.append(perturbed_visual_tokens)
            batch_at.append(encoder_atts)
            
        batch_vt = torch.cat(batch_vt, dim=0)
        batch_at = torch.cat(batch_at, dim=0)
        
        device_type = "cuda" if "cuda" in str(self.model.device) else "cpu"
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            text_embeds = text_embeds.expand(batch_vt.shape[0],-1,-1)
            decoder_embeds = decoder_embeds.expand(batch_vt.shape[0],-1,-1)
            
            inputs_embeds = torch.cat([batch_vt, text_embeds], dim=1)
            outputs = self.model.t5_model(inputs_embeds=inputs_embeds, attention_mask=batch_at,\
                                        decoder_inputs_embeds=decoder_embeds, output_attentions=True)
            next_token_logits = outputs[0][:,-1,:]
            
        return torch.softmax(next_token_logits, dim=-1)[:,target_index]
    
    
    def get_effects(self, visual_tokens, visual_atts, ori_probs,
                    enc_input_ids, enc_atts, dec_input_ids, target_index):
        device_type = "cuda" if "cuda" in str(self.model.device) else "cpu"
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            text_embeds = self.model.t5_model.encoder.embed_tokens(enc_input_ids)
            decoder_embeds = self.model.t5_model.decoder.embed_tokens(dec_input_ids)
        
            # w/o visual queries
            outputs = self.model.t5_model(inputs_embeds=text_embeds, attention_mask=enc_atts,\
                                            decoder_inputs_embeds=decoder_embeds, output_attentions=False)
            next_logits_wo_visual = outputs[0][:,-1,:]
        
        # probability without visual queries
        probs_wo_visual = torch.softmax(next_logits_wo_visual, dim=-1)[:,target_index]
                
        perturbed_probs = self.deletion(visual_tokens, visual_atts,\
                                        text_embeds, enc_atts, \
                                        decoder_embeds, target_index)
        
        query_effect = (ori_probs - perturbed_probs).clamp(min=0.0)
        
        return query_effect, ori_probs, probs_wo_visual
    
    
    def interpret_queries(self,
                          image,
                          visualization="", # repr|attn : representation or attention
                          return_dict=False,
                          iteration=1):
        
        # encoding visual queries from image
        attn_hooks = None
        if visualization!="repr":
            attn_hooks = [Hook(self.model.visual_encoder.blocks[l].attn.attn_drop) for l in range(len(self.model.visual_encoder.blocks))]
            
        image_embeds, image_atts = self.embed_image(image)
        visual_tokens, visual_atts, query_output = self.project_visual_tokens(image_embeds, image_atts)
        
        # Initial text generation
        seq_total_effects = None
        seq_token_ids = None
        seq_text = None
        seq_modal_segments = None
        seq_generated = []
        sample = {"image":None, "prompt":""}
        
        for g in range(iteration):
            print(g, sample["prompt"])
            outputs, generated_text = self.generate(sample, 
                                                    visual_queries=visual_tokens,
                                                    visual_atts=visual_atts)
            
            words = []
            for w in generated_text[0].split():
                if len(w)>3:
                    words.append(w)
            words = set(words)
            
            if seq_text is not None:
                print(set(seq_text.split()).intersection(words))
            if seq_text is not None and \
                (generated_text[0] in seq_text or \
                    len(set(seq_text.split()).intersection(words)) >= 3):
                print("seq_text", seq_text)
                print("gen", generated_text[0])
                break
            
            target_indices = outputs.sequences[0]
            
            prompts = self.prepare_text_prompt_input(sample["prompt"], image.size(0), image.device)
            enc_input_ids = prompts.input_ids
            enc_atts = prompts.attention_mask
            decoder_input_ids = torch.LongTensor([target_indices[0]]).to(image.device)
            decoder_input_ids = decoder_input_ids.unsqueeze(0)
            
            total_effects = []
            modal_segments = []
            for it in range(1,len(target_indices)):
                # get original probability
                logits = self.predict(sample, decoder_input_ids, \
                                    visual_tokens=visual_tokens, visual_atts=visual_atts, \
                                    requires_grad=False)
                
                ori_probs = torch.softmax(logits, dim=-1)[:,target_indices[it]]
            
                effects, ori_p, lm_p = self.get_effects(visual_tokens, visual_atts, ori_probs,
                                                enc_input_ids=enc_input_ids,
                                                enc_atts=enc_atts,
                                                dec_input_ids=decoder_input_ids,
                                                target_index=target_indices[it])
                
                total_effects.append(effects / ori_p)
                modal_segments.append((effects.sum() > lm_p).item())
                
                add_id = torch.LongTensor([target_indices[it]]).to(image.device)
                add_id = add_id.unsqueeze(0)
                decoder_input_ids = torch.cat([decoder_input_ids, add_id], dim=-1)
            
            if seq_total_effects is None:
                seq_total_effects = total_effects.copy()
                seq_token_ids = target_indices[1:]
                seq_text = generated_text[0]
                seq_modal_segments = modal_segments.copy()
            else:
                seq_total_effects = seq_total_effects[:-1]
                seq_total_effects.extend(total_effects)
                seq_token_ids = torch.cat([seq_token_ids[:-1], target_indices[1:]],dim=0)
                seq_text = " ".join([seq_text, generated_text[0]])
                seq_modal_segments = seq_modal_segments[:-1]
                seq_modal_segments.extend(modal_segments)
                
            seq_generated.append(generated_text[0])
            sample["prompt"] = seq_text #f"a photo of {seq_text}"
            # used_effects = torch.stack(total_effects, dim=0).max(dim=0)[0]
            # selected = torch.arange(0,used_effects.shape[0]).to(used_effects.device)
            # selected = selected[used_effects > 0.7]
            
            # visual_tokens = torch.index_select(visual_tokens.squeeze(0), dim=0, index=selected)
            # visual_atts = torch.index_select(visual_atts.squeeze(0), dim=0, index=selected)
            # visual_tokens = visual_tokens.unsqueeze(0)
            # visual_atts = visual_atts.unsqueeze(0)
            
            # print(used_effects)

        if return_dict:
            ret = {
                "total_effects": seq_total_effects,
                "token_ids": seq_token_ids,
                "text": seq_text,
                "modal_segment": seq_modal_segments,
                "seq_generated": seq_generated,
                "query_output": query_output,
                "image_embeds": image_embeds
            }
        else:
            ret = (seq_total_effects, seq_token_ids, seq_text, seq_modal_segments,)
        
        if visualization in ["repr", "attn", "reat"]:
            if visualization=="attn":
                vis = self.visualize_attention(image, query_output, attn_hooks)
            else:
                vis = self.visualize_visual_queries(image, query_output, \
                                        image_embeds, attn_hooks=attn_hooks)
            if return_dict:
                ret["query_viz"] = vis
            else:
                ret += (vis,)
                    
        return ret
        
    
    def explain(self, samples, decoder_input_ids, target_index=None):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - text_generated (torch.LongTensor): A tensor of shape (batch_size, S)
            target (int|torch.Tensor): Target token id
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        # encoding visual queries from image
        image = samples["image"]
        image_embeds, image_atts = self.embed_image(image)
        visual_tokens, visual_atts, _ = self.project_visual_tokens(image_embeds, image_atts)
        
        # Prepare text prompt
        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.model.prompt
        text_tokens = self.prepare_text_prompt_input(prompt, image.size(0), image.device)
        
        # get original probability
        logits = self.predict(samples, decoder_input_ids, \
                                visual_tokens=visual_tokens, visual_atts=visual_atts, requires_grad=False)
        
        if target_index==None:
            target_index = torch.argmax(logits, dim=1)[0].item()
            
        ori_probs = torch.softmax(logits, dim=-1)[:,target_index]
        
        effects, ori_prob, lm_prob = self.get_effects(visual_tokens, visual_atts, ori_probs,
                                                    enc_input_ids=text_tokens.input_ids,
                                                    enc_atts=text_tokens.attention_mask,
                                                    dec_input_ids=decoder_input_ids,
                                                    target_index=target_index)

        return effects, ori_prob, lm_prob
        
        
    def visualize_visual_queries(self, ori_img, query_out, img_embeds, attn_hooks=None, \
                                img_size=128, add_cls=False, interpolation="bilinear"):
        # Cross-attention map
        if attn_hooks is None:
            R_qi = rollout_for_cross_attention(
                query_out.attentions,
                query_out.cross_attentions,
            ) # [Q, I^2+1]
        else:
            R_qi = get_relevance_map_for_queries(query_out, attn_hooks)
            
        # Representation simiarities
        simmaps = get_image_understanding(img_embeds) # 256, 256
        simmaps = (simmaps-simmaps.mean(dim=-1,keepdim=True)) / simmaps.std(dim=-1,keepdim=True)
        simmaps = simmaps.clamp(min=0.0)
        simmaps = normalize_last_dim(simmaps)
        
        R_v = R_qi[:,1:]
        
        if add_cls:
            # Classification token
            cls_map = F.cosine_similarity(img_embeds[0,:1], img_embeds[0,1:]) # 256
            cls_map = (cls_map-cls_map.mean()) / cls_map.std()
            cls_map = cls_map.clamp(min=0.0)
            cls_map = normalize_last_dim(cls_map)
            R_v = R_v + torch.matmul(R_qi[:,0:1], cls_map.unsqueeze(0))
        
        R_v = torch.matmul(R_v, simmaps.t())
        R_v = normalize_last_dim(R_v)
        num_q, seq_len = R_v.shape
        rc = int(math.sqrt(seq_len))
        R_v = R_v.view(num_q,1,rc,rc)
        
        ori_img = F.interpolate(ori_img, (img_size, img_size), mode="bilinear")
        R_v = F.interpolate(R_v, (img_size, img_size), mode=interpolation)
        
        ori_img = normalize(ori_img).squeeze(0)
        R_v = normalize(R_v)
        
        vis = []
        for i in range(num_q):
            vis.append(show_cam_on_image(ori_img, R_v[i]))
            
        return vis
    
    def cot_for_image(self, sample, num_beams=5, max_length=50, repetition_penalty=1.2):
        image_embeds, image_atts = self.embed_image(sample["image"])
        vis_inputs, vis_atts, _ = self.project_visual_tokens(image_embeds, image_atts)

        new_sample = {"image":None, "prompt":sample["prompt"]}

        while True:
            _, generated_text = self.generate(new_sample, 
                                            visual_queries=vis_inputs,
                                            visual_atts=vis_atts,
                                            num_beams=num_beams,
                                            repetition_penalty=repetition_penalty,
                                            max_length=max_length)
            text = generated_text[0]

            words = []
            for w in text.split():
                if len(w)>3:
                    words.append(w)
            words = set(words)

            # print(set(new_sample["prompt"].split()).intersection(words))
            # print(words)

            pre_prompt = new_sample["prompt"]
            insec = set(pre_prompt.split()).intersection(words)
            if pre_prompt and (text in pre_prompt or len(insec)/len(words) > 0.5):
                break
            new_sample["prompt"] = " ".join([pre_prompt, text])

        return new_sample["prompt"]