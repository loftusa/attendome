#%%
"""
Frozen encoder experiment (phase 1 baseline): kv-cloning

====
To begin with, don't bother with learning an embedding.  Pick a smart model [e.g., a SOTA sentence embedding transformer, call it S] and then see if we can train KV using its last-layer token embedding to duplicate the A pattern of an arbitrary different LM head in a target transformer T.  I.e., can we clone T's heads in the space of S?  We expect it to be "some yes, some no" and force other issues, like: what to do with different tokenization schemes.

====
* **Raw signal** Fit head-specific key/value tensors `(Kₕ, Vₕ)` so that, for a *frozen* sentence-encoder `S` with hidden states `z_p`, the reconstructed attention
  \( \hat A_h(p)=\operatorname{Softmax}(z_p K_h^\top/\sqrt{d_s}) V_h )\ matches the true pattern `A_h(p)` over many prompts `p`.
* **Alignment** All heads are projected into `S`'s token space, eliminating tokenizer mismatch.
* **Embedding** Concatenate and flatten `[Kₕ, Vₕ]`, then compress via PCA/random projection to size `d`.
* **Pros** Directly captures the *causal* mechanism that generates attention.
* **Cons** High-dimensional before compression; requires an optimisation per head.

====
(1) identify some well-understood attention heads in some open-source models, e.g., known induction heads, copying heads
(2a) build a classification dataset of known attn heads. Could be binary ("is it an induction head") or multiclass ("is it an induction head, is it a copying head, is it x, ..., is it unclear"). Probably should be binary to start? we can use the method+code @Sheridan @Kerem Sahin  mentioned to build this dataset. Then split it into a train/test set and put it somewhere public.
(2b) for a few of these well-understood target heads, train a shallow MLP or something else simple that takes sentence embeddings from qwen3-embedding-8b as input and tries to predict the target head's (num_tokens, num_tokens) attn pattern (this is pretty much @David Bau 's step 1 I think?). The idea is to de-risk the project and attack obvious technical hurdles early
"""
#%%
import sentence_transformers

sentence_embedder = "Qwen/Qwen3-Embedding-8B" 

# Requires transformers>=4.51.0
# Requires sentence-transformers>=2.7.0

from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")

# We recommend enabling flash_attention_2 for better acceleration and memory saving,
# together with setting `padding_side` to "left":
# model = SentenceTransformer(
#     "Qwen/Qwen3-Embedding-8B",
#     model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
#     tokenizer_kwargs={"padding_side": "left"},
# )

# The queries and documents to embed
queries = [
    "What is the capital of China?",
    "Explain gravity",
]
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

# Encode the queries and documents. Note that queries benefit from using a prompt
# Here we use the prompt called "query" stored under `model.prompts`, but you can
# also pass your own prompt via the `prompt` argument
query_embeddings = model.encode(queries, prompt_name="query")
document_embeddings = model.encode(documents)

# Compute the (cosine) similarity between the query and document embeddings
similarity = model.similarity(query_embeddings, document_embeddings)
print(similarity)
# tensor([[0.7493, 0.0751],
#         [0.0880, 0.6318]])
