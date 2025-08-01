import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple
import math
from tqdm import tqdm

class AttentionMapDataset(Dataset):
    """Dataset for attention maps and sentence embeddings."""
    
    def __init__(self, attention_maps: Dict, selected_samples_dict: Dict, sentence_embeddings: np.ndarray):
        """
        Args:
            attention_maps: Dict[layer_idx][head_idx] -> array of shape [10000, 128, 128]
            sentence_embeddings: array of shape [10000, 4096]
        """
        self.sentence_embeddings = torch.tensor(sentence_embeddings, dtype=torch.float32)
        self.samples = []
        
        # Flatten the attention maps structure and create head indices
        head_idx = 0
        for layer_idx in attention_maps:
            for head_idx_in_layer in attention_maps[layer_idx]:
                maps = attention_maps[layer_idx][head_idx_in_layer]  # [10000, 128, 128]
                selected_samples = selected_samples_dict[layer_idx][head_idx_in_layer]
                for sample_idx in selected_samples:
                    self.samples.append({
                        'sample_idx': sample_idx,
                        'head_idx': head_idx,
                        'attention_map': torch.tensor(maps[sample_idx], dtype=torch.float32)
                    })
                head_idx += 1
        
        self.num_heads = head_idx
        print(f"Created dataset with {len(self.samples)} samples and {self.num_heads} heads")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        sentence_emb = self.sentence_embeddings[sample['sample_idx']]
        head_idx = sample['head_idx']
        attention_map = sample['attention_map']
        
        return sentence_emb, head_idx, attention_map

class AttentionMapPredictor(nn.Module):
    """
    Generative model that predicts attention maps from sentence embeddings 
    and learnable attention head embeddings.
    """
    
    def __init__(self, 
                 sentence_dim: int = 4096,
                 head_embed_dim: int = 256,
                 num_heads: int = 50,
                 hidden_dim: int = 512,
                 map_size: int = 128):
        super().__init__()
        
        self.sentence_dim = sentence_dim
        self.head_embed_dim = head_embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.map_size = map_size
        
        # Learnable attention head embeddings
        self.head_embeddings = nn.Embedding(num_heads, head_embed_dim)
        
        # Sentence encoder (reduce dimensionality)
        self.sentence_encoder = nn.Sequential(
            nn.Linear(sentence_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )
        
        # Head embedding encoder
        self.head_encoder = nn.Sequential(
            nn.Linear(head_embed_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU()
        )
        
        # Fusion layer
        fusion_input_dim = hidden_dim // 2 + hidden_dim // 4
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Attention map decoder
        # Generate the attention map in a structured way
        self.map_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, map_size * map_size)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0, 0.1)
    
    def forward(self, sentence_embeddings: torch.Tensor, head_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            sentence_embeddings: [batch_size, sentence_dim]
            head_indices: [batch_size] - indices of attention heads
            
        Returns:
            attention_logits: [batch_size, map_size, map_size] - logits before softmax
        """
        batch_size = sentence_embeddings.size(0)
        
        # Get head embeddings
        head_embeds = self.head_embeddings(head_indices)  # [batch_size, head_embed_dim]
        
        # Encode inputs
        sentence_encoded = self.sentence_encoder(sentence_embeddings)  # [batch_size, hidden_dim//2]
        head_encoded = self.head_encoder(head_embeds)  # [batch_size, hidden_dim//4]
        
        # Fuse representations
        fused = torch.cat([sentence_encoded, head_encoded], dim=-1)  # [batch_size, fusion_input_dim]
        fused = self.fusion(fused)  # [batch_size, hidden_dim]
        
        # Generate attention map logits
        map_logits = self.map_decoder(fused)  # [batch_size, map_size^2]
        map_logits = map_logits.view(batch_size, self.map_size, self.map_size)  # [batch_size, map_size, map_size]
        
        return map_logits
    
    def predict_attention_map(self, sentence_embeddings: torch.Tensor, head_indices: torch.Tensor) -> torch.Tensor:
        """
        Predict attention maps with softmax applied.
        
        Returns:
            attention_maps: [batch_size, map_size, map_size] - softmax applied on last dim
        """
        with torch.no_grad():
            logits = self.forward(sentence_embeddings, head_indices)
            attention_maps = F.softmax(logits, dim=-1)
            return attention_maps


class AttentionMapPredictorCNN(nn.Module):
    """
    CNN-based generative model that predicts attention maps from sentence embeddings 
    and learnable attention head embeddings. Uses convolutional layers to better 
    capture spatial patterns in attention maps.
    """
    
    def __init__(self, 
                 sentence_dim: int = 4096,
                 head_embed_dim: int = 256,
                 num_heads: int = 50,
                 hidden_dim: int = 512,
                 map_size: int = 128):
        super().__init__()
        
        self.sentence_dim = sentence_dim
        self.head_embed_dim = head_embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.map_size = map_size
        
        # Calculate initial feature map size for CNN decoder
        # We'll start from 8x8 and upsample to 128x128
        self.initial_size = 8
        self.initial_channels = 64
        
        # Learnable attention head embeddings
        self.head_embeddings = nn.Embedding(num_heads, head_embed_dim)
        
        # Sentence encoder (reduce dimensionality)
        self.sentence_encoder = nn.Sequential(
            nn.Linear(sentence_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )
        
        # Head embedding encoder
        self.head_encoder = nn.Sequential(
            nn.Linear(head_embed_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU()
        )
        
        # Fusion layer
        fusion_input_dim = hidden_dim // 2 + hidden_dim // 4
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Project to initial feature map
        self.to_feature_map = nn.Sequential(
            nn.Linear(hidden_dim, self.initial_channels * self.initial_size * self.initial_size),
            nn.ReLU()
        )
        
        # CNN decoder using transposed convolutions
        # 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
        self.cnn_decoder = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(self.initial_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            
            # Final convolution to single channel
            nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0, 0.1)
            elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, sentence_embeddings: torch.Tensor, head_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            sentence_embeddings: [batch_size, sentence_dim]
            head_indices: [batch_size] - indices of attention heads
            
        Returns:
            attention_logits: [batch_size, map_size, map_size] - logits before softmax
        """
        batch_size = sentence_embeddings.size(0)
        
        # Get head embeddings
        head_embeds = self.head_embeddings(head_indices)  # [batch_size, head_embed_dim]
        
        # Encode inputs
        sentence_encoded = self.sentence_encoder(sentence_embeddings)  # [batch_size, hidden_dim//2]
        head_encoded = self.head_encoder(head_embeds)  # [batch_size, hidden_dim//4]
        
        # Fuse representations
        fused = torch.cat([sentence_encoded, head_encoded], dim=-1)  # [batch_size, fusion_input_dim]
        fused = self.fusion(fused)  # [batch_size, hidden_dim]
        
        # Project to feature map
        feature_map = self.to_feature_map(fused)  # [batch_size, initial_channels * initial_size^2]
        feature_map = feature_map.view(batch_size, self.initial_channels, self.initial_size, self.initial_size)
        
        # Generate attention map using CNN decoder
        attention_logits = self.cnn_decoder(feature_map)  # [batch_size, 1, map_size, map_size]
        attention_logits = attention_logits.squeeze(1)  # [batch_size, map_size, map_size]
        
        return attention_logits
    
    def predict_attention_map(self, sentence_embeddings: torch.Tensor, head_indices: torch.Tensor) -> torch.Tensor:
        """
        Predict attention maps with softmax applied.
        
        Returns:
            attention_maps: [batch_size, map_size, map_size] - softmax applied on last dim
        """
        with torch.no_grad():
            logits = self.forward(sentence_embeddings, head_indices)
            attention_maps = F.softmax(logits, dim=-1)
            return attention_maps

def train_model(model: AttentionMapPredictor, 
                train_dataloader: DataLoader,
                test_dataloader: DataLoader,
                num_epochs: int = 100,
                learning_rate: float = 1e-3,
                model_prefix: str = '',
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict[str, List[float]]:
    """
    Train the attention map prediction model with test set evaluation.
    
    Args:
        model: The attention map predictor model
        train_dataloader: DataLoader for training data
        test_dataloader: DataLoader for test data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to run training on
        eval_every: Evaluate on test set every N epochs
        
    Returns:
        Dictionary containing 'train_losses' and 'test_losses' lists
    """
    model = model.to(device)
    
    # Loss function - KL divergence between predicted and target attention maps
    def attention_loss(pred_logits, target_maps):
        # pred_maps = F.softmax(pred_logits, dim=-1)
        # Add small epsilon to avoid log(0)
        pred_log_maps = F.log_softmax(pred_logits, dim=-1)
        target_maps = target_maps + 1e-8
        target_maps = target_maps / target_maps.sum(dim=-1, keepdim=True)
        
        # KL divergence
        kl_loss = F.kl_div(pred_log_maps, target_maps, reduction='batchmean')
        return kl_loss
    
    def evaluate_model(model, dataloader, device):
        """Evaluate model on given dataloader and return average loss."""
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for sentence_embs, head_indices, target_maps in tqdm(dataloader, desc="Eval"):
                sentence_embs = sentence_embs.to(device)
                head_indices = head_indices.to(device)
                target_maps = target_maps.to(device)
                
                # Forward pass
                pred_logits = model(sentence_embs, head_indices)
                
                # Compute loss
                loss = attention_loss(pred_logits, target_maps)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)
    
    train_losses = []
    test_losses = []
    best_test_loss = 10000
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for sentence_embs, head_indices, target_maps in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            sentence_embs = sentence_embs.to(device)
            head_indices = head_indices.to(device)
            target_maps = target_maps.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_logits = model(sentence_embs, head_indices)
            
            # Compute loss
            loss = attention_loss(pred_logits, target_maps)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Evaluation phase
        test_loss = None
        test_loss = evaluate_model(model, test_dataloader, device)
        test_losses.append(test_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, "
              f"Test Loss: {test_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # scheduler.step(test_loss)
        scheduler.step()
            
        # save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), 
                       f'./results/models/{model_prefix}multihead_generator_best.pth')
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses
    }

def compare_models(attention_maps_sample: Dict, sentence_embeddings_sample: np.ndarray):
    """
    Compare the two model architectures and provide recommendations.
    
    Args:
        attention_maps_sample: Small sample of attention maps for testing
        sentence_embeddings_sample: Corresponding sentence embeddings
    """
    print("=== MODEL COMPARISON ===\n")
    
    # Create small dataset for comparison
    dataset = AttentionMapDataset(attention_maps_sample, sentence_embeddings_sample)
    
    # Create both models
    model_fc = AttentionMapPredictor(
        sentence_dim=4096,
        head_embed_dim=256,
        num_heads=dataset.num_heads,
        hidden_dim=512,
        map_size=128
    )
    
    model_cnn = AttentionMapPredictorCNN(
        sentence_dim=4096,
        head_embed_dim=256,
        num_heads=dataset.num_heads,
        hidden_dim=512,
        map_size=128
    )
    
    # Count parameters
    fc_params = sum(p.numel() for p in model_fc.parameters())
    cnn_params = sum(p.numel() for p in model_cnn.parameters())
    
    print(f"Parameter Count:")
    print(f"  FC Model:  {fc_params:,} parameters")
    print(f"  CNN Model: {cnn_params:,} parameters")
    print(f"  Ratio:     {cnn_params/fc_params:.2f}x\n")
    
    # Test inference speed
    import time
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_fc.to(device)
    model_cnn.to(device)
    
    # Sample batch
    sample_sentence = torch.randn(16, 4096).to(device)
    sample_heads = torch.randint(0, dataset.num_heads, (16,)).to(device)
    
    # Warmup
    for _ in range(10):
        _ = model_fc(sample_sentence, sample_heads)
        _ = model_cnn(sample_sentence, sample_heads)
    
    # Time FC model
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    for _ in range(100):
        _ = model_fc(sample_sentence, sample_heads)
    torch.cuda.synchronize() if device == 'cuda' else None
    fc_time = time.time() - start
    
    # Time CNN model
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    for _ in range(100):
        _ = model_cnn(sample_sentence, sample_heads)
    torch.cuda.synchronize() if device == 'cuda' else None
    cnn_time = time.time() - start
    
    print(f"Inference Speed (100 batches of 16 samples):")
    print(f"  FC Model:  {fc_time:.3f}s ({fc_time/100*1000:.1f}ms per batch)")
    print(f"  CNN Model: {cnn_time:.3f}s ({cnn_time/100*1000:.1f}ms per batch)")
    print(f"  Speedup:   {fc_time/cnn_time:.2f}x {'(FC faster)' if fc_time < cnn_time else '(CNN faster)'}\n")
    
    print("RECOMMENDATIONS:")
    print("================")
    print("Use CNN Model if:")
    print("  • Attention patterns have strong spatial structure")
    print("  • You want to capture local spatial dependencies")
    print("  • You have sufficient compute resources")
    print("  • Attention maps show clear spatial clustering\n")
    
    print("Use FC Model if:")
    print("  • Attention patterns are more global/distributed")
    print("  • You need faster inference")
    print("  • You have limited compute resources") 
    print("  • Dataset is very small (<5k samples)")
    print("  • Attention maps don't show strong spatial structure\n")
    
    print("GENERAL ADVICE:")
    print("• Start with CNN model for most cases (better inductive bias)")
    print("• Use FC model if CNN overfits on your specific dataset")
    print("• CNN model may need more regularization (dropout, weight decay)")
    print("• Both models support the same training interface")