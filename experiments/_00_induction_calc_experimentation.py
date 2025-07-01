#%%
from attendome.dataset.attention_head_classifier import InductionHeadClassifier
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

classifier = InductionHeadClassifier(device="cuda")

model_name = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

#%%
scores = classifier.compute_induction_score(model, tokenizer, save_random_repetitive_sequence=True)

#%%
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

scores_array = np.array(scores)
num_layers, num_heads = scores_array.shape

#%%
import transformer_lens
from transformer_lens.head_detector import detect_head


seq = tokenizer.decode(classifier.random_repetitive_sequence_[0], skip_special_tokens=False)
model_tl = transformer_lens.HookedTransformer.from_pretrained(model_name)
scores_tl = detect_head(model_tl, detection_pattern="induction_head", seq=seq).numpy()
#%%
scores_tl_array = np.array(scores_tl)
num_layers, num_heads = scores_tl_array.shape

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 12))

# Plot our scores
sns.heatmap(
    scores_array, 
    xticklabels=[f"Head {i}" for i in range(num_heads)],
    yticklabels=[f"Layer {i}" for i in range(num_layers)],
    cmap="viridis",
    cbar_kws={'label': 'Induction Score'},
    annot=True,
    fmt='.3f',
    ax=ax1
)
ax1.set_title("Our Induction Scores by Layer and Head")
ax1.set_xlabel("Attention Head")
ax1.set_ylabel("Layer")

# Plot transformer_lens scores
sns.heatmap(
    scores_tl_array, 
    xticklabels=[f"Head {i}" for i in range(num_heads)],
    yticklabels=[f"Layer {i}" for i in range(num_layers)],
    cmap="viridis",
    cbar_kws={'label': 'Induction Score'},
    annot=True,
    fmt='.3f',
    ax=ax2
)
ax2.set_title("TransformerLens Induction Scores by Layer and Head")
ax2.set_xlabel("Attention Head")
ax2.set_ylabel("Layer")

plt.tight_layout()
plt.show()
# %%
