import random 
import torch 
import math 

from attendome import llama_utils
from attendome import gpt_utils

def pile_chunk(random_len, pile, tok, shuf_pile=True):
    sample = []
    while len(sample) < random_len:
        doc = pile.shuffle()[0]['text'] # sample from huggingface
        sample = tok(doc, bos=False)[: random_len]
        if shuf_pile:
            random.shuffle(sample)
    return sample 

def get_l2_attn_weights(model, tokenized, layer, value_weighting):
    with torch.no_grad():
        with model.trace(tokenized):
            if value_weighting:
                value_states = model.model.layers[layer].self_attn.source.past_key_value_update_0.output[1].save()
            attn_weights = model.model.layers[layer].self_attn.source.attention_interface_0.output[1].save()

    if not value_weighting:
        return attn_weights.detach().cpu()
    else:
        # get l2 norm of each head value vector [bsz, n_heads, seq_len, n_dims] -> [bsz, n_heads, seq_len]
        value_norms = torch.linalg.vector_norm(value_states, dim=-1).detach().cpu()

        # attn_weights [bsz, n_heads, seq_len, seq_len]
        attn_weights = attn_weights.detach().cpu()

        # then multiply by softmax values and normalize 
        effective = attn_weights * value_norms.unsqueeze(2).expand(attn_weights.shape)
        effective /= torch.sum(effective, dim=-1, keepdim=True)
        return effective

def get_gptj_attn_weights(model, tokenized, layer, value_weighting):
    with torch.no_grad():
        with model.trace(tokenized):
            if value_weighting:
                value_states = model.transformer.h[layer].attn.source.layer_past_update_0.output[1].save()
            attn_weights = model.transformer.h[layer].attn.source.self__attn_0.output[1].save()

    if not value_weighting:
        return attn_weights.detach().cpu()
    else:
        # get l2 norm of each head value vector [bsz, n_heads, seq_len, n_dims] -> [bsz, n_heads, seq_len]
        value_norms = torch.linalg.vector_norm(value_states, dim=-1).detach().cpu()

        # attn_weights [bsz, n_heads, seq_len, seq_len]
        attn_weights = attn_weights.detach().cpu()

        # then multiply by softmax values and normalize 
        effective = attn_weights * value_norms.unsqueeze(2).expand(attn_weights.shape)
        effective /= torch.sum(effective, dim=-1, keepdim=True)
        return effective

def get_qwen3_attn_weights(model, tokenized, layer, value_weighting):
    with torch.no_grad():
        with model.trace(tokenized):
            if value_weighting:
                value_states = model.model.layers[layer].self_attn.source.attention_interface_0.source.repeat_kv_1.output.save()
            attn_weights = model.model.layers[layer].self_attn.source.attention_interface_0.output[1].save()

    if not value_weighting:
        return attn_weights.detach().cpu()
    else:
        # get l2 norm of each head value vector [bsz, n_heads, seq_len, n_dims] -> [bsz, n_heads, seq_len]
        value_norms = torch.linalg.vector_norm(value_states, dim=-1).detach().cpu()

        # attn_weights [bsz, n_heads, seq_len, seq_len]
        attn_weights = attn_weights.detach().cpu()

        # then multiply by softmax values and normalize 
        effective = attn_weights * value_norms.unsqueeze(2).expand(attn_weights.shape)
        effective /= torch.sum(effective, dim=-1, keepdim=True)
        return effective

def get_gpt_neox_attn_weights(model, tokenized, layer, value_weighting):
    with torch.no_grad():
        with model.trace(tokenized):
            if value_weighting:
                value_states = model.gpt_neox.layers[layer].attention.source.layer_past_update_0.output[1].save()
            attn_weights = model.gpt_neox.layers[layer].attention.source.attention_interface_0.output[1].save()

    if not value_weighting:
        return attn_weights.detach().cpu()
    else:
        # get l2 norm of each head value vector [bsz, n_heads, seq_len, n_dims] -> [bsz, n_heads, seq_len]
        value_norms = torch.linalg.vector_norm(value_states, dim=-1).detach().cpu()

        # attn_weights [bsz, n_heads, seq_len, seq_len]
        attn_weights = attn_weights.detach().cpu()

        # then multiply by softmax values and normalize 
        effective = attn_weights * value_norms.unsqueeze(2).expand(attn_weights.shape)
        effective /= torch.sum(effective, dim=-1, keepdim=True)
        return effective

# def get_l2_attn_weights(model, tokenized, layer, value_weighting):
    # n_heads = model.config.num_attention_heads 
    # head_dim = model.config.hidden_size // n_heads
    
    # with torch.no_grad():
    #     with model.trace(tokenized):
    #         # positional_embeddings (cos, sin) each shape [bsz, seq_len, head_dim]
    #         position = model.model.layers[layer].self_attn.inputs[1]['position_embeddings']
    #         attention_mask = model.model.layers[layer].self_attn.inputs[1]['attention_mask']

    #         # [bsz, seq_len, model_size]
    #         query_states = model.model.layers[layer].self_attn.q_proj.output
    #         key_states = model.model.layers[layer].self_attn.k_proj.output 

    #         bsz = query_states.shape[0]; seq_len = query_states.shape[1] 
    #         if value_weighting:
    #             # [bsz, seq_len, model_size] -> [bsz, seq_len, n_heads, head_dim]
    #             value_states = model.model.layers[layer].self_attn.v_proj.output
    #             value_states = value_states.view(bsz, seq_len, n_heads, head_dim).save()

    #         # from modeling_llama, convert to [bsz, n_heads, seq_len, head_dim] and rotate 
    #         query_states = query_states.view(bsz, seq_len, -1, head_dim).transpose(1, 2)
    #         key_states = key_states.view(bsz, seq_len, -1, head_dim).transpose(1, 2)
    #         query_states, key_states = llama_utils.apply_rotary_pos_emb(query_states, key_states, position[0], position[1])

    #         # not needed because num_key_value_heads == num_attention_heads 
    #         # key_states = repeat_kv(key_states, self.num_key_value_groups)
    #         attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
            
    #         # has to be eager implementation 
    #         causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    #         attn_weights = attn_weights + causal_mask
    #         attn_weights = attn_weights.save()
    
    # if not value_weighting:
    #     return attn_weights.softmax(dim=-1).detach().cpu()
    # else: 
    #     # get l2 norm of each head value vector [bsz, seq_len, n_heads] -> [bsz, n_heads, seq_len]
    #     value_norms = torch.linalg.vector_norm(value_states, dim=-1).detach().cpu().transpose(1, 2)

    #     # attn_weights [bsz, n_heads, seq_len, seq_len]
    #     attn_weights = attn_weights.softmax(dim=-1).detach().cpu()

    #     # then multiply by softmax values and normalize 
    #     effective = attn_weights * value_norms.unsqueeze(2).expand(attn_weights.shape)
    #     effective /= torch.sum(effective, dim=-1, keepdim=True)
    #     return effective 

def get_l3_attn_weights(model, tokenized, layer, value_weighting):
    n_heads = model.config.num_attention_heads 
    head_dim = model.config.hidden_size // n_heads
    
    with torch.no_grad():
        with model.trace(tokenized):
            # self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
            n_kv_groups = model.model.layers[layer].self_attn.num_key_value_groups

            # positional_embeddings (cos, sin) each shape [bsz, seq_len, head_dim]
            position = model.model.layers[layer].self_attn.inputs[1]['position_embeddings']
            attention_mask = model.model.layers[layer].self_attn.inputs[1]['attention_mask']

            # grouped query means that we have more queries than we do keys/values
            query_states = model.model.layers[layer].self_attn.q_proj.output # [bsz, seq_len, model_size=4096]
            key_states = model.model.layers[layer].self_attn.k_proj.output  # [bsz, seq_len, 1024]
            bsz = query_states.shape[0]; seq_len = query_states.shape[1] 

            if value_weighting:
                value_states = model.model.layers[layer].self_attn.v_proj.output # [bsz, seq_len, 1024]
                value_states = value_states.view(bsz, seq_len, -1, head_dim).transpose(1, 2) # [bsz, seq_len, 8, head_dim] -> [bsz, 8, seq_len, head_dim]
                value_states = llama_utils.repeat_kv(value_states, n_kv_groups).save()

            # from modeling_llama, convert to [bsz, n_heads, seq_len, head_dim] and rotate 
            query_states = query_states.view(bsz, seq_len, -1, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, seq_len, -1, head_dim).transpose(1, 2)
            query_states, key_states = llama_utils.apply_rotary_pos_emb(query_states, key_states, position[0], position[1])

            # not needed because num_key_value_heads == num_attention_heads 
            key_states = llama_utils.repeat_kv(key_states, n_kv_groups)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
            
            # has to be eager implementation 
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
            attn_weights = attn_weights.save()
    
    if not value_weighting:
        return attn_weights.softmax(dim=-1).detach().cpu()
    else: 
        # get l2 norm of each head value vector [bsz, n_heads, seq_len]
        value_norms = torch.linalg.vector_norm(value_states, dim=-1).detach().cpu()

        # attn_weights [bsz, n_heads, seq_len, seq_len]
        attn_weights = attn_weights.softmax(dim=-1).detach().cpu()

        # then multiply by softmax values and normalize 
        effective = attn_weights * value_norms.unsqueeze(2).expand(attn_weights.shape)
        effective /= torch.sum(effective, dim=-1, keepdim=True)
        return effective   


def get_qwen3_attn_weights(model, tokenized, layer, value_weighting):
    n_heads = model.config.num_attention_heads 
    head_dim = getattr(model.config, "head_dim", model.config.hidden_size // n_heads)
    
    with torch.no_grad():
        with model.trace(tokenized):
            # Get attention module
            attn_module = model.model.layers[layer].self_attn
            
            # num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
            n_kv_groups = attn_module.num_key_value_groups
            
            # position_embeddings (cos, sin) from inputs
            position_embeddings = attn_module.inputs[1]['position_embeddings']
            attention_mask = attn_module.inputs[1]['attention_mask']
            
            # Get projected states before normalization
            query_states = attn_module.q_proj.output  # [bsz, seq_len, n_heads * head_dim]
            key_states = attn_module.k_proj.output    # [bsz, seq_len, n_kv_heads * head_dim]
            
            bsz = query_states.shape[0]
            seq_len = query_states.shape[1]
            
            if value_weighting:
                value_states = attn_module.v_proj.output  # [bsz, seq_len, n_kv_heads * head_dim]
                # Reshape and transpose: [bsz, seq_len, n_kv_heads, head_dim] -> [bsz, n_kv_heads, seq_len, head_dim]
                value_states = value_states.view(bsz, seq_len, -1, head_dim).transpose(1, 2)
                # Repeat KV if using grouped query attention
                value_states = llama_utils.repeat_kv(value_states, n_kv_groups).save()
            
            # Reshape to [bsz, seq_len, n_heads/n_kv_heads, head_dim]
            query_states = query_states.view(bsz, seq_len, -1, head_dim)
            key_states = key_states.view(bsz, seq_len, -1, head_dim)
            
            # Apply RMSNorm (Qwen3 specific) - norm is applied per head_dim
            query_states = attn_module.q_norm(query_states)
            key_states = attn_module.k_norm(key_states)
            
            # Transpose to [bsz, n_heads/n_kv_heads, seq_len, head_dim]
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            
            # Apply rotary position embeddings
            cos = position_embeddings[0]
            sin = position_embeddings[1]
            query_states, key_states = llama_utils.apply_rotary_pos_emb(query_states, key_states, cos, sin)
            
            # Repeat KV if using grouped query attention
            key_states = llama_utils.repeat_kv(key_states, n_kv_groups)
            
            # Compute attention weights with Qwen3's scaling
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * attn_module.scaling
            
            # Apply attention mask
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask
            
            attn_weights = attn_weights.save()
    
    if not value_weighting:
        return attn_weights.softmax(dim=-1).detach().cpu()
    else:
        # Get L2 norm of each head value vector [bsz, n_heads, seq_len]
        value_norms = torch.linalg.vector_norm(value_states, dim=-1).detach().cpu()
        # attn_weights [bsz, n_heads, seq_len, seq_len]
        attn_weights = attn_weights.softmax(dim=-1).detach().cpu()
        # Multiply by value norms and normalize
        effective = attn_weights * value_norms.unsqueeze(2).expand(attn_weights.shape)
        effective /= torch.sum(effective, dim=-1, keepdim=True)
        return effective

# def get_gptj_attn_weights(model, tokenized, layer, value_weighting):
#     """
#     Extract attention weights from a specific layer of GPT-J model.
    
#     Args:
#         model: The GPT-J model
#         tokenized: Tokenized input
#         layer: Layer index to extract attention from
#         value_weighting: Whether to apply value weighting to attention scores
    
#     Returns:
#         Attention weights tensor
#     """
#     n_heads = model.config.num_attention_heads
#     head_dim = model.config.hidden_size // n_heads
    
#     with torch.no_grad():
#         with model.trace(tokenized):
#             # Get the attention layer
#             attn_layer = model.transformer.h[layer].attn
            
#             # Get inputs - position_ids and attention_mask
#             position_ids = attn_layer.inputs[1]['position_ids']
#             attention_mask = attn_layer.inputs[1].get('attention_mask', None)
            
#             # Get Q, K, V projections
#             query_states = attn_layer.q_proj.output  # [bsz, seq_len, hidden_size]
#             key_states = attn_layer.k_proj.output    # [bsz, seq_len, hidden_size]
            
#             bsz = query_states.shape[0]
#             seq_len = query_states.shape[1]
            
#             if value_weighting:
#                 value_states = attn_layer.v_proj.output  # [bsz, seq_len, hidden_size]
#                 # Split heads for value (rotary=False for values in GPT-J)
#                 value_states = value_states.view(bsz, seq_len, n_heads, head_dim)
#                 value_states = value_states.permute(0, 2, 1, 3)  # [bsz, n_heads, seq_len, head_dim]
#                 value_states = value_states.save()
            
#             # Split heads with rotary=True for query and key
#             # From GPT-J's _split_heads method with rotary=True
#             query_states = query_states.view(bsz, seq_len, n_heads, head_dim)  # Keep original shape for rotary
#             key_states = key_states.view(bsz, seq_len, n_heads, head_dim)
            
#             # Get sinusoidal position embeddings
#             embed_positions = attn_layer._get_embed_positions(position_ids)
#             repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
#             sincos = torch.gather(embed_positions, 1, repeated_position_ids)
#             split_size = sincos.shape[-1] // 2
#             sin = sincos[..., :split_size]
#             cos = sincos[..., split_size:]
#             # sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
            
#             # Apply rotary position embeddings based on rotary_dim
#             if attn_layer.rotary_dim is not None:
#                 # Only rotate first rotary_dim dimensions
#                 k_rot = key_states[:, :, :, :attn_layer.rotary_dim]
#                 k_pass = key_states[:, :, :, attn_layer.rotary_dim:]
                
#                 q_rot = query_states[:, :, :, :attn_layer.rotary_dim]
#                 q_pass = query_states[:, :, :, attn_layer.rotary_dim:]
                
#                 # Apply rotary embeddings to the rotary portion
#                 # Note: apply_rotary_pos_emb is defined in the GPT-J module
#                 k_rot = gpt_utils.apply_rotary_pos_emb(k_rot, sin, cos)
#                 q_rot = gpt_utils.apply_rotary_pos_emb(q_rot, sin, cos)
                
#                 key_states = torch.cat([k_rot, k_pass], dim=-1)
#                 query_states = torch.cat([q_rot, q_pass], dim=-1)
#             else:
#                 # Rotate all dimensions
#                 key_states = gpt_utils.apply_rotary_pos_emb(key_states, sin, cos)
#                 query_states = gpt_utils.apply_rotary_pos_emb(query_states, sin, cos)
            
#             # Permute to [bsz, n_heads, seq_len, head_dim]
#             key_states = key_states.permute(0, 2, 1, 3)
#             query_states = query_states.permute(0, 2, 1, 3)
            
#             # Compute attention scores
#             # Convert to float32 to avoid overflow (as done in GPT-J)
#             query_states = query_states.to(torch.float32)
#             key_states = key_states.to(torch.float32)
            
#             attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))
#             attn_weights = attn_weights / attn_layer.scale_attn
            
#             # Apply causal mask if available
#             if attention_mask is not None:
#                 causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
#                 attn_weights = attn_weights + causal_mask
            
#             attn_weights = attn_weights.save()
    
#     if not value_weighting:
#         return attn_weights.softmax(dim=-1).detach().cpu()
#     else:
#         # Get l2 norm of each head value vector [bsz, n_heads, seq_len]
#         value_norms = torch.linalg.vector_norm(value_states, dim=-1).detach().cpu()
#         # attn_weights [bsz, n_heads, seq_len, seq_len]
#         attn_weights = attn_weights.softmax(dim=-1).detach().cpu()
#         # Multiply by value norms and normalize
#         effective = attn_weights * value_norms.unsqueeze(2).expand(attn_weights.shape)
#         effective /= torch.sum(effective, dim=-1, keepdim=True)
#         return effective