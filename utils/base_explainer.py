import math
import torch
import torch.nn.functional as F
from transformers import GenerationConfig
from utils.visualization import normalize_last_dim, normalize, show_cam_on_image

# rule 5 from paper
def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam

# normalization- eq. 8+9
def handle_residual(orig_self_attention):
    self_attention = orig_self_attention.clone()
    diag_idx = range(self_attention.shape[-1])
    self_attention -= torch.eye(self_attention.shape[-1]).to(self_attention.device)
    assert self_attention[diag_idx, diag_idx].min() >= 0
    self_attention = self_attention / self_attention.sum(dim=-1, keepdim=True)
    self_attention += torch.eye(self_attention.shape[-1]).to(self_attention.device)
    return self_attention


# rule 10 from paper
def apply_mm_attention_rules(R_ss, R_qq, cam_sq, apply_normalization=True, apply_self_in_rule_10=True):
    R_ss_normalized = R_ss
    R_qq_normalized = R_qq
    if apply_normalization:
        R_ss_normalized = handle_residual(R_ss)
        R_qq_normalized = handle_residual(R_qq)
    R_sq_addition = torch.matmul(R_ss_normalized.t(), torch.matmul(cam_sq, R_qq_normalized))
    if not apply_self_in_rule_10:
        R_sq_addition = cam_sq
    R_sq_addition[torch.isnan(R_sq_addition)] = 0
    return R_sq_addition


def get_relevance_map_for_self_attention(attention, gradients, device, R_ctx=None):
    # attention: L [H S S]
    # R_ctx: previous relevance map of tokens given an input
    
    seq_len = attention[0].shape[2]
    R_ii = torch.eye(seq_len, seq_len).to(device)
    
    if R_ctx is not None:
        pre_seq_len = R_ctx.shape[0]
        R_ii[:pre_seq_len,:pre_seq_len] = R_ctx.clone().type(torch.bfloat16)
    
    num_layers = len(attention)
    for l in range(num_layers):
        grad = gradients[l]
        attn = attention[l].detach()
        attn = avg_heads(attn, grad)
        R_ii = R_ii + torch.matmul(attn.float(), R_ii)
        
    return R_ii

def get_relevance_map_for_cross_attention(
            attention, gradients, \
            cross_attention, cross_gradients,
            R_ctx):
    # attn: L [B, H, R_q, R_q]
    # cross_attn: L [B, H, R_q, R_i]
    # grads: L [B, H, R_q, R_q]
    
    num_queries = attention[0].shape[-1]
    num_img_tokens = cross_attention[0].shape[-1]
    device = attention[0].device
    
    # queries self attention matrix
    R_qq = torch.eye(num_queries, num_queries).to(device)
    # impact of image boxes on queries
    R_qi = torch.zeros(num_queries, num_img_tokens).to(device)
    
    for l in range(len(attention)):
        grad = gradients[l]
        attn = attention[l].detach()
        attn = avg_heads(attn, grad)
        R_qq = R_qq + torch.matmul(attn.float(), R_qq)
        R_qi = R_qi + torch.matmul(attn.float(), R_qi)
        
        c_grad = cross_gradients[l]
        c_attn = cross_attention[l]
        
        if c_grad is not None:
            c_attn = avg_heads(c_attn.detach(), c_grad)
            R_qi = R_qi + apply_mm_attention_rules(R_qq, R_ctx, c_attn.float())
    
    return R_qq, R_qi


def rollout_for_self_attention(attention, device, R_ctx=None):
    # attention: L [H S S]
    # R_ctx: previous relevance map of tokens given an input
    
    seq_len = attention[0].shape[2]
    R_ii = torch.eye(seq_len, seq_len).to(device)
    
    if R_ctx is not None:
        pre_seq_len = R_ctx.shape[0]
        R_ii[:pre_seq_len,:pre_seq_len] = R_ctx.clone().type(torch.bfloat16)
    
    num_layers = len(attention)
    for l in range(num_layers):
        attn = attention[l].detach()
        attn = attn.mean(dim=1).squeeze(0) # head sum
        R_ii = R_ii + torch.matmul(attn.float(), R_ii)
        
    return R_ii

def rollout_for_cross_attention(attention, cross_attention, R_ctx=None, dim=0):
    num_queries = attention[0].shape[-1]
    num_img_tokens = cross_attention[0].shape[-1]
    device = attention[0].device
    
    # queries self attention matrix
    R_qq = torch.eye(num_queries, num_queries).to(device)
    # impact of image boxes on queries
    R_qi = torch.zeros(num_queries, num_img_tokens).to(device)

    for l in range(len(attention)):
        attn = attention[l].detach()
        attn = attn.mean(dim=1).squeeze(0) # head sum
        R_qq = R_qq + torch.matmul(attn, R_qq)
        R_qi = R_qi + torch.matmul(attn, R_qi)

        c_attn = cross_attention[l]
        if type(c_attn) is torch.Tensor:
            c_attn = c_attn.mean(dim=1).squeeze(0)
            
            R_qq_norm = handle_residual(R_qq)
            if R_ctx is not None:
                R_ctx_norm = handle_residual(R_ctx)
                R_qi_addition = torch.matmul(R_qq_norm.t(), torch.matmul(c_attn, R_ctx_norm))
            else:
                R_qi_addition = torch.matmul(R_qq_norm.t(), c_attn)#+ c_attn
            
            R_qi = R_qi + R_qi_addition
    
    min_v = R_qi.min(dim=dim,keepdim=True)[0]
    max_v = R_qi.max(dim=dim,keepdim=True)[0]
    R_qi = (R_qi - min_v) / (max_v-min_v)
    
    return R_qi
    

def get_gradients_across_layers(attentions, target):
    grads = []
    for attn in attentions:
        if type(attn) is tuple:
            grads.append(None)
        else:
            grad = torch.autograd.grad(target, [attn], retain_graph=True)[0]
            grads.append(grad)
    
    return grads

def get_relevance_map_for_queries(query_out, attn_hooks):
    # get attentions and gradients of ViT-G through hook 
    vit_attentions = [layer.output for layer in attn_hooks]
    
    # Get relevance maps
    R_ii = rollout_for_self_attention(vit_attentions, device=vit_attentions[0].device)
    R_qi = rollout_for_cross_attention(query_out.attentions, query_out.cross_attentions, R_ctx=R_ii)  # [Q, I^2+1]
        
    return R_qi

    

class BaseExplainerBLIP2():
    """
    BLIP2-flanT5 and BLIP2-OPT
    """
    def __init__(self, model, model_type="t5"):
        self.model = model
        self.model.eval()
        self.gen_cfg = GenerationConfig(return_dict_in_generate=True, \
                    output_attentions=True, output_scores=True, output_hidden_states=True)
        if model_type=="t5":
            self.proj = self.model.t5_proj
            self.tokenizer = self.model.t5_tokenizer
        elif model_type=="opt":
            self.proj = self.model.opt_proj
            self.tokenizer = self.model.opt_tokenizer
        else:
            raise ValueError("type should be either t5 or opt")

        self.model_type = model_type
        
        
    def embed_image(self, image):
        with torch.cuda.amp.autocast(enabled=(self.model.device != torch.device("cpu"))):
            image_embeds = self.model.ln_vision(self.model.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        
        return image_embeds, image_atts
    
    def project_visual_tokens(self, image_embeds, image_atts):
        query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            output_attentions=True,
            return_dict=True,
        )
        
        inputs = self.proj(query_output.last_hidden_state)
        atts = torch.ones(inputs.size()[:-1], dtype=torch.long).to(image_embeds.device)
        
        return inputs, atts, query_output
    
    def prepare_text_prompt_input(self, prompt, batch_size, device):
        if isinstance(prompt, str):
            prompt = [prompt] * batch_size
        else:
            assert len(prompt) == batch_size, "The number of prompts must be equal to the batch size."

        input_tokens = self.tokenizer(prompt, padding="longest", return_tensors="pt").to(device)
        
        return input_tokens
        
        
    def generate(
        self,
        samples,
        visual_queries=None,
        visual_atts=None,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        
        image = samples["image"]
        
        if image is not None:
            image_embeds, image_atts = self.embed_image(image)
            inputs, atts, _ = self.project_visual_tokens(image_embeds, image_atts)
        elif visual_queries is not None and visual_atts is not None:
            inputs, atts = visual_queries, visual_atts
        else:
            inputs = None
        
        # Prepare text prompt
        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.model.prompt
            
        batch_size = 1
        device = self.model.query_tokens.device
        if inputs is not None:
            batch_size = inputs.size(0)
            device = inputs.device
            
        input_tokens = self.prepare_text_prompt_input(prompt, batch_size, device)
        input_ids = input_tokens.input_ids
        attention_mask = input_tokens.attention_mask
        if inputs is not None:
            attention_mask = torch.cat([atts, attention_mask], dim=1)
        
        if self.model_type=="t5":
            device_type = "cuda" if "cuda" in str(self.model.device) else "cpu"
            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
                inputs_embeds = self.model.t5_model.encoder.embed_tokens(input_ids)
                
                if inputs is not None:
                    inputs_embeds = torch.cat([inputs, inputs_embeds], dim=1)

                outputs = self.model.t5_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    do_sample=use_nucleus_sampling,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_new_tokens=max_length,
                    min_length=min_length,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                    generation_config=self.gen_cfg
                )
                
                output_text = self.tokenizer.batch_decode(
                    outputs.sequences, skip_special_tokens=True
                )
        else:
            if use_nucleus_sampling:
                query_embeds = inputs.repeat_interleave(num_captions, dim=0)
                num_beams = 1
            else:
                query_embeds = inputs.repeat_interleave(num_beams, dim=0)

            with torch.cuda.amp.autocast(enabled=(self.model.device != torch.device("cpu"))):
                outputs = self.model.opt_model.generate(
                    input_ids=input_ids,
                    query_embeds=query_embeds,
                    attention_mask=attention_mask,
                    do_sample=use_nucleus_sampling,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_new_tokens=max_length,
                    min_length=min_length,
                    eos_token_id=self.model.eos_token_id,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                )

            output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return outputs, output_text
    

    def predict(self, samples, decoder_input_ids=None,  \
                visual_tokens=None, visual_atts=None, requires_grad=False):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - text_generated (torch.LongTensor): A tensor of shape (batch_size, S)
            target (int|torch.Tensor): Target token id
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if visual_tokens is None:
            image = samples["image"]
            image.requires_grad = requires_grad
            image_embeds, image_atts = self.embed_image(image)
            inputs, atts, query_output = self.project_visual_tokens(image_embeds, image_atts)
        else:
            inputs = visual_tokens
            image_embeds = None
            if visual_atts is None:
                atts = torch.ones(inputs.size()[:-1], dtype=torch.long).to(inputs.device)
            else:
                atts = visual_atts
        
        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.model.prompt

        input_tokens = self.prepare_text_prompt_input(prompt, inputs.size(0), inputs.device)
        input_ids = input_tokens.input_ids
        attention_mask = input_tokens.attention_mask
        attention_mask = torch.cat([atts, attention_mask], dim=1)
        
        if self.model_type=="t5":
            device_type = "cuda" if "cuda" in str(self.model.device) else "cpu"
            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
                inputs_embeds = self.model.t5_model.encoder.embed_tokens(input_tokens.input_ids)
                inputs_embeds = torch.cat([inputs, inputs_embeds], dim=1)
                decoder_embeds = self.model.t5_model.decoder.embed_tokens(decoder_input_ids)
                # decoder_embeds.requires_grad = requires_grad
                
                outputs = self.model.t5_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,\
                                            decoder_inputs_embeds=decoder_embeds, output_attentions=True)
                next_token_logits = outputs[0][:,-1,:]
        else:
            with torch.cuda.amp.autocast(enabled=(self.model.device != torch.device("cpu"))):
                outputs = self.model.opt_model(
                    input_ids=input_ids,
                    query_embeds=inputs,
                    attention_mask=attention_mask,
                    output_attentions=True,
                )
                next_token_logits = outputs[0][:,-1,:]

        if visual_tokens is None:
            return next_token_logits, image_embeds, query_output, outputs
        else:
            return next_token_logits
        
        
    def visualize_attention(self, ori_img, query_out, attn_hooks, \
                            img_size=128, interpolation="bilinear",):
        
        R_qi = get_relevance_map_for_queries(query_out, attn_hooks)
        
        R_qi = R_qi[:,1:]
        R_qi = normalize_last_dim(R_qi)
        
        num_q, seq_len = R_qi.shape
        rc = int(math.sqrt(seq_len))
        R_qi = R_qi.view(num_q,1,rc,rc)
        
        ori_img = F.interpolate(ori_img, (img_size, img_size), mode="bilinear")
        R_qi = F.interpolate(R_qi, (img_size, img_size), mode=interpolation)
        
        ori_img = normalize(ori_img).squeeze(0)
        R_qi = normalize(R_qi)
        
        vis = []
        for i in range(R_qi.shape[0]):
            vis.append(show_cam_on_image(ori_img, R_qi[i]))
            
        return vis