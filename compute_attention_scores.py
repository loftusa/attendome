import json
import os 
import argparse 
import torch 
import random 
import numpy as np
import pandas as pd 
from tqdm import tqdm 
from collections import defaultdict
import math

from nnsight import LanguageModel
from datasets import load_dataset


def pile_chunk(random_len, pile, tok, shuf_pile=True):
    sample = []
    while len(sample) < random_len:
        doc = pile.shuffle()[0]['text'] # sample from huggingface
        sample = tok(doc, bos=False)[: random_len]
        if shuf_pile:
            random.shuffle(sample)
    return sample 

def get_l2_attn_weights(model, tokenized, layer, value_weighting):
    n_heads = model.config.num_attention_heads 
    head_dim = model.config.hidden_size // n_heads
    
    with torch.no_grad():
        with model.trace(tokenized):
            # positional_embeddings (cos, sin) each shape [bsz, seq_len, head_dim]
            position = model.model.layers[layer].self_attn.inputs[1]['position_embeddings']
            attention_mask = model.model.layers[layer].self_attn.inputs[1]['attention_mask']

            # [bsz, seq_len, model_size]
            query_states = model.model.layers[layer].self_attn.q_proj.output
            key_states = model.model.layers[layer].self_attn.k_proj.output 

            bsz = query_states.shape[0]; seq_len = query_states.shape[1] 
            if value_weighting:
                # [bsz, seq_len, model_size] -> [bsz, seq_len, n_heads, head_dim]
                value_states = model.model.layers[layer].self_attn.v_proj.output
                value_states = value_states.view(bsz, seq_len, n_heads, head_dim).save()

            # from modeling_llama, convert to [bsz, n_heads, seq_len, head_dim] and rotate 
            query_states = query_states.view(bsz, seq_len, -1, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, seq_len, -1, head_dim).transpose(1, 2)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, position[0], position[1])

            # not needed because num_key_value_heads == num_attention_heads 
            # key_states = repeat_kv(key_states, self.num_key_value_groups)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
            
            # has to be eager implementation 
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
            attn_weights = attn_weights.save()
    
    if not value_weighting:
        return attn_weights.softmax(dim=-1).detach().cpu()
    else: 
        # get l2 norm of each head value vector [bsz, seq_len, n_heads] -> [bsz, n_heads, seq_len]
        value_norms = torch.linalg.vector_norm(value_states, dim=-1).detach().cpu().transpose(1, 2)

        # attn_weights [bsz, n_heads, seq_len, seq_len]
        attn_weights = attn_weights.softmax(dim=-1).detach().cpu()

        # then multiply by softmax values and normalize 
        effective = attn_weights * value_norms.unsqueeze(2).expand(attn_weights.shape)
        effective /= torch.sum(effective, dim=-1, keepdim=True)
        return effective 

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
                value_states = repeat_kv(value_states, n_kv_groups).save()

            # from modeling_llama, convert to [bsz, n_heads, seq_len, head_dim] and rotate 
            query_states = query_states.view(bsz, seq_len, -1, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, seq_len, -1, head_dim).transpose(1, 2)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, position[0], position[1])

            # not needed because num_key_value_heads == num_attention_heads 
            key_states = repeat_kv(key_states, n_kv_groups)
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

import torch 

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L178
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# can use for llama2, llama3, and olmo2 
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    # print(hidden_states)
    # batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    batch = hidden_states.shape[0]
    num_key_value_heads = hidden_states.shape[1] 
    slen = hidden_states.shape[2]
    head_dim = hidden_states.shape[3]
    
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def get_olmo2_attn_weights(model, tokenized, layer, value_weighting):
    n_heads = model.config.num_attention_heads 
    head_dim = model.config.hidden_size // n_heads
    
    with torch.no_grad():
        with model.trace(tokenized):
            # positional_embeddings (cos, sin) each shape [bsz, seq_len, head_dim]
            position = model.model.layers[layer].self_attn.inputs[1]['position_embeddings']
            attention_mask = model.model.layers[layer].self_attn.inputs[1]['attention_mask']

            # [bsz, seq_len, model_size]
            query_states = model.model.layers[layer].self_attn.q_norm.output
            key_states = model.model.layers[layer].self_attn.k_norm.output 
            bsz = query_states.shape[0]; seq_len = query_states.shape[1] 

            if value_weighting:
                # [bsz, seq_len, model_size] -> [bsz, seq_len, n_heads, head_dim]
                value_states = model.model.layers[layer].self_attn.v_proj.output
                value_states = value_states.view(bsz, seq_len, n_heads, head_dim).save()

            # from modeling_llama, convert to [bsz, n_heads, seq_len, head_dim] and rotate 
            bsz = query_states.shape[0]; seq_len = query_states.shape[1]
            query_states = query_states.view(bsz, seq_len, -1, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, seq_len, -1, head_dim).transpose(1, 2)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, position[0], position[1])

            scaling = model.model.layers[layer].self_attn.scaling # it's just 1/math.sqrt(head_dim)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
            attn_weights = attn_weights.save()
    
    if not value_weighting:
        return attn_weights.softmax(dim=-1).detach().cpu()
    else: 
        # get l2 norm of each head value vector [bsz, seq_len, n_heads] -> [bsz, n_heads, seq_len]
        value_norms = torch.linalg.vector_norm(value_states, dim=-1).detach().cpu().transpose(1, 2)

        # attn_weights [bsz, n_heads, seq_len, seq_len]
        attn_weights = attn_weights.softmax(dim=-1).detach().cpu()

        # then multiply by softmax values and normalize 
        effective = attn_weights * value_norms.unsqueeze(2).expand(attn_weights.shape)
        effective /= torch.sum(effective, dim=-1, keepdim=True)
        return effective 

def get_pythia_attn_weights(model, tokenized, layer, value_weighting):
    n_heads = model.config.num_attention_heads 
    head_dim = model.config.hidden_size // n_heads
    
    with torch.no_grad():
        with model.trace(tokenized):
            attention_mask = model.gpt_neox.layers[layer].attention.inputs[1]['attention_mask']
            pos = model.gpt_neox.layers[layer].attention.inputs[1]['position_embeddings']

            qkv = model.gpt_neox.layers[layer].attention.query_key_value.output
            bsz = qkv.shape[0]; seq_len = qkv.shape[1]
            qkv = qkv.view((bsz, seq_len, n_heads, 3 * head_dim)).transpose(1, 2).chunk(3, dim=-1)

            query_states = qkv[0]; key_states = qkv[1]; value_states = qkv[2]
            query_states, key_states = pythia_apply_rotary_pos_emb(query_states, key_states, pos[0], pos[1])

            if value_weighting:
                # [bsz, seq_len, model_size] -> [bsz, seq_len, n_heads, head_dim]
                value_states = value_states.reshape(bsz, seq_len, n_heads, head_dim).save()
            
            scaling = model.gpt_neox.layers[layer].attention.scaling # it's just 1/math.sqrt(head_dim)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
            attn_weights = attn_weights.save()
    
    if not value_weighting:
        return attn_weights.softmax(dim=-1).detach().cpu()
    else: 
        # get l2 norm of each head value vector [bsz, seq_len, n_heads] -> [bsz, n_heads, seq_len]
        value_norms = torch.linalg.vector_norm(value_states, dim=-1).detach().cpu().transpose(1, 2)

        # attn_weights [bsz, n_heads, seq_len, seq_len]
        attn_weights = attn_weights.softmax(dim=-1).detach().cpu()

        # then multiply by softmax values and normalize 
        effective = attn_weights * value_norms.unsqueeze(2).expand(attn_weights.shape)
        effective /= torch.sum(effective, dim=-1, keepdim=True)
        return effective 


def pythia_rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def pythia_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed

# import torch
# import math

# def get_qwen3_attn_weights(model, tokenized, layer, value_weighting=False):
#     """
#     Extract attention weights from a specific layer of a Qwen3 model.
    
#     Args:
#         model: Qwen3 model with tracing capability
#         tokenized: Tokenized input
#         layer: Layer index to extract attention from
#         value_weighting: If True, weight attention by value vector magnitudes
    
#     Returns:
#         Attention weights tensor of shape [bsz, n_heads, seq_len, seq_len]
#     """
#     # Get model configuration
#     config = model.config
#     n_heads = config.num_attention_heads
#     head_dim = config.hidden_size // n_heads
    
#     # Qwen3 might use different config names, adjust as needed
#     n_kv_heads = getattr(config, 'num_key_value_heads', n_heads)
#     n_kv_groups = n_heads // n_kv_heads
    
#     with torch.no_grad():
#         with model.trace(tokenized):
#             # Access the transformer layer - Qwen3 typically uses 'model.layers'
#             layer_module = model.model.layers[layer]
            
#             # Access self-attention module
#             attn_module = layer_module.self_attn
            
#             # Get query, key, value projections - save them immediately to avoid proxy issues
#             query_states = attn_module.q_proj.output.save()
#             key_states = attn_module.k_proj.output.save()
            
#             if value_weighting:
#                 value_states = attn_module.v_proj.output.save()
    
#     # # Now work with actual tensors outside the tracing context
#     # print(f"Debug - query_states shape: {query_states.shape}")
#     # print(f"Debug - key_states shape: {key_states.shape}")
#     # if value_weighting:
#     #     print(f"Debug - value_states shape: {value_states.shape}")
#     # print(f"Debug - n_heads: {n_heads}, n_kv_heads: {n_kv_heads}, head_dim: {head_dim}")
    
#     # Get actual dimensions
#     bsz  = query_states.shape[0]
#     seq_len = query_states.shape[1]
    
#     # Calculate actual head dimensions from tensor shapes (more reliable than config)
#     actual_hidden_size = query_states.shape[2]
#     actual_kv_size = key_states.shape[2]
#     actual_head_dim = actual_hidden_size // n_heads
#     actual_kv_head_dim = actual_kv_size // n_kv_heads
    
#     # print(f"Debug - bsz: {bsz}, seq_len: {seq_len}")
#     # print(f"Debug - actual_hidden_size: {actual_hidden_size}, actual_kv_size: {actual_kv_size}")
#     # print(f"Debug - calculated head_dim: {actual_head_dim}, kv_head_dim: {actual_kv_head_dim}")
    
#     # Verify dimensions match
#     if actual_head_dim != actual_kv_head_dim:
#         print(f"Warning: Query head_dim ({actual_head_dim}) != KV head_dim ({actual_kv_head_dim})")
#         print("Using query head_dim for calculations")
    
#     # Use the actual head dimension
#     head_dim = actual_head_dim
    
#     # Reshape for multi-head attention
#     # Handle different possible shapes
#     if len(query_states.shape) == 3:
#         # Standard shape: [bsz, seq_len, hidden_size]
#         query_states = query_states.view(bsz, seq_len, n_heads, head_dim).transpose(1, 2)
#         key_states = key_states.view(bsz, seq_len, n_kv_heads, head_dim).transpose(1, 2)
        
#         if value_weighting:
#             value_states = value_states.view(bsz, seq_len, n_kv_heads, head_dim).transpose(1, 2)
            
#     elif len(query_states.shape) == 4:
#         # Already reshaped: might be [bsz, seq_len, n_heads, head_dim] or [bsz, n_heads, seq_len, head_dim]
#         if query_states.shape[1] == n_heads:
#             # Already in [bsz, n_heads, seq_len, head_dim] format
#             pass
#         elif query_states.shape[2] == n_heads:
#             # In [bsz, seq_len, n_heads, head_dim] format, need to transpose
#             query_states = query_states.transpose(1, 2)
#             key_states = key_states.transpose(1, 2)
#             if value_weighting:
#                 value_states = value_states.transpose(1, 2)
#         else:
#             raise ValueError(f"Cannot understand 4D query tensor shape: {query_states.shape}")
#     else:
#         raise ValueError(f"Unexpected query tensor shape: {query_states.shape}")
    
#     # print(f"Debug - after reshape, query_states shape: {query_states.shape}")
#     # print(f"Debug - after reshape, key_states shape: {key_states.shape}")
    
#     # Apply rotary positional embeddings if available
#     # For Qwen3, we'll try to reconstruct them or skip if not available
#     cos, sin = create_rotary_emb(head_dim, seq_len, query_states.device)
#     query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
#     # Handle grouped query attention - repeat key/value states if needed
#     if n_kv_groups > 1:
#         key_states = repeat_kv(key_states, n_kv_groups)
#         if value_weighting:
#             value_states = repeat_kv(value_states, n_kv_groups)
    
#     # Compute attention scores
#     attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
    
#     # Apply causal mask
#     causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=attn_weights.device), diagonal=1)
#     attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
    
#     # Apply softmax to get attention probabilities
#     attn_probs = attn_weights.softmax(dim=-1)
    
#     if not value_weighting:
#         return attn_probs.detach().cpu()
#     else:
#         # Compute value weighting
#         # Get L2 norm of each head value vector [bsz, n_heads, seq_len]
#         value_norms = torch.linalg.vector_norm(value_states, dim=-1)
        
#         # Weight attention by value magnitudes
#         # attn_probs: [bsz, n_heads, seq_len, seq_len]
#         # value_norms: [bsz, n_heads, seq_len]
#         effective = attn_probs * value_norms.unsqueeze(2).expand(attn_probs.shape)
        
#         # Renormalize
#         effective = effective / (torch.sum(effective, dim=-1, keepdim=True) + 1e-8)
        
#         return effective.detach().cpu()


# def create_rotary_emb(head_dim, seq_len, device):
#     """
#     Create rotary positional embeddings.
    
#     Args:
#         head_dim: Dimension of attention head
#         seq_len: Sequence length
#         device: Device to create tensors on
    
#     Returns:
#         cos, sin tensors for rotary embedding
#     """
#     # Create position indices
#     position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    
#     # Create frequency tensor
#     inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    
#     # Compute frequencies
#     freqs = torch.outer(position_ids.squeeze(), inv_freq)
#     emb = torch.cat((freqs, freqs), dim=-1)
    
#     cos = emb.cos()[None, None, :, :]  # [1, 1, seq_len, head_dim]
#     sin = emb.sin()[None, None, :, :]  # [1, 1, seq_len, head_dim]
    
#     return cos, sin


# def apply_rotary_pos_emb(q, k, cos, sin):
#     """
#     Apply rotary positional embeddings to query and key tensors.
    
#     Args:
#         q: Query tensor [batch, n_heads, seq_len, head_dim]
#         k: Key tensor [batch, n_heads, seq_len, head_dim]
#         cos: Cosine values for rotary embedding
#         sin: Sine values for rotary embedding
    
#     Returns:
#         Rotated query and key tensors
#     """
#     def rotate_half(x):
#         """Rotates half the hidden dims of the input."""
#         x1 = x[..., : x.shape[-1] // 2]
#         x2 = x[..., x.shape[-1] // 2 :]
#         return torch.cat((-x2, x1), dim=-1)
    
#     # Apply rotation
#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
    
#     return q_embed, k_embed

torch.set_grad_enabled(False)

def generate_ragged_batch(batch_ents, pile, tok, seq_len):
    assert len({len(e) for e in batch_ents}) == 1

    newline = tok('\n', bos=False)[-1]
    if 'qwen' in args.model.lower():
        bos = None
    else:
        bos = tok('', bos=True)[0]

    sequences = []
    start_idxs, end_idxs = [], []
    for ent in batch_ents:
        position = random.choice(range(seq_len // 2, seq_len - len(ent) + 1))
        rand1 = pile_chunk(position, pile, tok)
        rand2 = pile_chunk(seq_len - position - len(ent), pile, tok)

        start_idxs.append(position + 1)
        end_idxs.append(position + len(ent))
        if 'qwen' in args.model.lower():
            sequences.append(
                rand1 + ent + rand2 + [newline] + rand1 
            )
        else:
            sequences.append(
                [bos] + rand1 + ent + rand2 + [newline] + rand1 
            )

    # since batches have ragged ends by design, save padding offsets 
    flipped_masks = [m - 1 for m in tok(sequences, pad_mask=True)]
    pad_offsets = [-sum(f).item() for f in flipped_masks]

    return sequences, torch.tensor(start_idxs), torch.tensor(end_idxs), torch.tensor(pad_offsets)

def retrieve_attention(model, tokenized, layer, value_weighting=True):
    func = {
        # 'Qwen/Qwen3-4B' : get_qwen3_attn_weights,
        # 'Qwen/Qwen3-8B' : get_qwen3_attn_weights,
        'meta-llama/Llama-3.2-3B-Instruct' : get_l3_attn_weights,
        'meta-llama/Llama-3.1-8B-Instruct' : get_l3_attn_weights,
        # 'meta-llama/Meta-Llama-3-8B' : get_l3_attn_weights,
        'meta-llama/Llama-2-7b-hf': get_l2_attn_weights,
        'allenai/OLMo-2-1124-7B' : get_olmo2_attn_weights,
        'EleutherAI/pythia-6.9b' : get_pythia_attn_weights
    }[model.config._name_or_path]

    return func(model, tokenized, layer, value_weighting)

def normalize(d, total):
    for k in d.keys():
        d[k] /= total 
    return d 

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='meta-llama/Llama-2-7b-hf')
parser.add_argument('--ckpt', default=None, type=str)
parser.add_argument('--n', default=2048, type=int)
parser.add_argument('--bsz', default=128, type=int, help='may have bugs with bsz=1.')
parser.add_argument('--sequence_len', default=30)
parser.add_argument('--random_tok_entities', action='store_true')
parser.set_defaults(random_tok_entities=False)
args = parser.parse_args()
print(args)

random.seed(8)
torch.manual_seed(8)
np.random.seed(8)

model = LanguageModel(args.model, device_map='auto')

model_name = args.model.split('/')[-1]
d = model.tokenizer.decode

assert args.bsz <= args.n // 4 

def tok(s, bos=False, model=model, pad_mask=False):
    if pad_mask:
        assert type(s) == list and type(s[0]) == list and type(s[0][0]) == int, s
        return model.tokenizer.pad({'input_ids' : s}, return_tensors='pt')['attention_mask']

    # otherwise get actual tokens 
    if 'llama' in model.config._name_or_path:
        if not bos: 
            return model.tokenizer(s)['input_ids'][1:]
        else:
            return model.tokenizer(s)['input_ids']
    elif 'qwen' in model.config._name_or_path.lower():
        # qwen models don't have a BOS token, so just return the tokens as-is
        return model.tokenizer(s)['input_ids']
    elif model.config._name_or_path in ['allenai/OLMo-2-1124-7B', 'EleutherAI/pythia-6.9b']:
        if not bos:
            return model.tokenizer(s)['input_ids']
        else:
            return [model.tokenizer.bos_token_id] + model.tokenizer(s)['input_ids']
        

# load in pile sample to use as basic material that we shuffle around 
# pile = load_dataset('NeelNanda/pile-10k')['train']
pile = load_dataset('JeanKaddour/minipile')['test']

# dummy entities for comparison 
sorted_entities = defaultdict(list)
if args.random_tok_entities:
    for i in range(args.n):
        doc_toks = []
        while len(doc_toks) < 5:
            doc = pile.shuffle()[0]['text']
            doc_toks = tok(doc)

        random.shuffle(doc_toks)
        if i % 4 == 0: 
            sorted_entities['bigram'].append(doc_toks[:2])
        elif i % 4 == 1: 
            sorted_entities['trigram'].append(doc_toks[:3])
        elif i % 4 == 2:
            sorted_entities['fourgram'].append(doc_toks[:4])
        elif i % 4 == 3: 
            sorted_entities['fivegram'].append(doc_toks[:5])
# load and sort entities of different token lengths
else:
    str_entities = list(pd.read_csv('./dataset_files/counterfact_expanded.csv')['subject'])
    for ent in str_entities:
        toks = tok(ent)
        if len(toks) == 2:
            sorted_entities['bigram'].append(toks)
        elif len(toks) == 3: 
            sorted_entities['trigram'].append(toks)
        elif len(toks) == 4: 
            sorted_entities['fourgram'].append(toks)
        elif len(toks) == 5: 
            sorted_entities['fivegram'].append(toks) 

# For each head, save the stuff 
total_examples = 0 
next_tok_attn = defaultdict(int)
end_tok_attn = defaultdict(int)

# I guess we're doing each batch is the same length entity 
for l, ents in sorted_entities.items():
    selected_ents = ents[ : args.n // 4]
    n_batches = len(selected_ents) // args.bsz
    print('attention for', l, model.tokenizer.decode(selected_ents[0]))

    for batch_idx in tqdm(range(n_batches)):
        batch_ents = selected_ents[batch_idx * args.bsz : (batch_idx + 1) * args.bsz]
        batch_seqs, start_idxs, end_idxs, pad_offsets = generate_ragged_batch(batch_ents, pile, tok, args.sequence_len)
        
        print(repr(model.tokenizer.decode(batch_seqs[0])))
        print(start_idxs[0].item(), end_idxs[0].item(), model.tokenizer.decode(batch_seqs[0][start_idxs[0]]), model.tokenizer.decode(batch_seqs[0][end_idxs[0]]))

        # get attention patterns for each head and example 
        for layer in range(model.config.num_hidden_layers):
            # [bsz, n_heads, seq_from, seq_to]
            attns = retrieve_attention(model, batch_seqs, layer)

            # index in and save beginnings, ends 
            for head in range(model.config.num_attention_heads):
                next_tok_attn[(layer, head)] += attns[torch.arange(len(attns)), head, -1, start_idxs + pad_offsets].sum().item()
                end_tok_attn[(layer, head)] += attns[torch.arange(len(attns)), head, -1, end_idxs + pad_offsets].sum().item()
        
        total_examples += len(batch_ents)

def json_tuple_keys(mapping):
    return [{'layer':k[0], 'head_idx': k[1], 'score' : v} for k, v in mapping.items()]

results = {
    'next_tok_attn' : json_tuple_keys(normalize(next_tok_attn, total_examples)),
    'end_tok_attn' : json_tuple_keys(normalize(end_tok_attn, total_examples))
}

path = f'./results/attention_scores/{model_name}/'
path += f'{args.ckpt}/' if args.ckpt is not None else ''
os.makedirs(path, exist_ok=True)

fname = f'n{args.n}_seqlen{args.sequence_len}'
fname += f'_randomtokents' if args.random_tok_entities else ''
fname += '.json'
print(path + fname)

with open(path + fname, 'w') as f:
    json.dump(results, f)