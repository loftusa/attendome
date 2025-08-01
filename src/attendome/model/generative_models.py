import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class AttentionMapGenerator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=256):
        """
        CNN-based generator for attention maps
        
        Args:
            embedding_dim: Dimension of input sentence embeddings
            hidden_dim: Hidden dimension size
        """
        super(AttentionMapGenerator, self).__init__()
        
        # Fully connected layers to process sentence embedding
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 8 * 8 * 64)  # Output for 8x8x64 feature map
        )
        
        # Transposed convolutions to upsample to 128 x 128
        self.deconv_layers = nn.Sequential(
            # 8x8x64 -> 16x16x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            
            # 16x16x32 -> 32x32x16
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            
            # 32x32x16 -> 64x64x8
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            
            # 64x64x8 -> 128x128x1
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
        )
        
    def forward(self, sentence_embedding):
        # Process sentence embedding through FC layers
        x = self.fc_layers(sentence_embedding)
        
        # Reshape to 2D feature map (batch_size, channels, height, width)
        x = x.view(-1, 64, 8, 8)
        
        # Generate attention map through deconvolutions
        attention_map = self.deconv_layers(x)
        attention_map = attention_map.squeeze(1)  # Remove channel dimension
        
        # Apply softmax along the last dimension to ensure valid probability distribution
        batch_size = attention_map.size(0)
        attention_map = attention_map.view(batch_size, 128, 128)
        attention_map = torch.softmax(attention_map, dim=-1)
        
        return attention_map

def kl_divergence_loss(predicted, target, epsilon=1e-8):
    """
    KL divergence loss for probability distributions
    More suitable for attention maps than MSE
    """
    # Add small epsilon to avoid log(0)
    predicted = predicted + epsilon
    target = target + epsilon
    predicted = predicted.view((-1, 128))
    target = target.view((-1, 128))
    
    # Normalize to ensure they are valid probability distributions
    predicted = predicted / predicted.sum(dim=-1, keepdim=True)
    target = target / target.sum(dim=-1, keepdim=True)
    
    # KL divergence: sum(target * log(target / predicted))
    kl_div = torch.sum(target * torch.log(target / predicted), dim=1)
    return torch.mean(kl_div)

def train_attention_generator(modelsentence_embeddings, attention_maps, embedding_dim, 
                            num_epochs=150, batch_size=32, learning_rate=1e-3,
                            use_kl_loss=True, device='cpu', model_prefix=''):
    """
    Train the attention map generator
    
    Args:
        sentence_embeddings: numpy array of shape [10000, embedding_dim]
        attention_maps: numpy array of shape [10000, 128, 128]
        embedding_dim: dimension of sentence embeddings
        use_kl_loss: whether to use KL divergence loss instead of MSE
    """
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Convert to tensors
    sentence_embeddings = torch.FloatTensor(sentence_embeddings).to(device)
    attention_maps = torch.FloatTensor(attention_maps).to(device)
    
    # Create train/validation split
    indices = np.arange(len(sentence_embeddings))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    train_embeddings = sentence_embeddings[train_idx]
    train_maps = attention_maps[train_idx]
    val_embeddings = sentence_embeddings[val_idx]
    val_maps = attention_maps[val_idx]
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_embeddings, train_maps)
    val_dataset = TensorDataset(val_embeddings, val_maps)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = AttentionMapGenerator(embedding_dim).to(device)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    if use_kl_loss:
        criterion = kl_divergence_loss
        print("Using KL divergence loss")
    else:
        criterion = nn.MSELoss()
        print("Using MSE loss")
        
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.5, verbose=True)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for embeddings, targets in train_loader:
            optimizer.zero_grad()
            
            outputs = model(embeddings) # after softmax: [batch_size, 128, 128]
            loss = criterion(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for embeddings, targets in val_loader:
                outputs = model(embeddings)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), './results/models/generator_best.pth')

        
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    # Load best model
    model.load_state_dict(torch.load('./results/models/generator_best.pth'))
    
    return model, train_losses, val_losses

def evaluate_model(model, sentence_embeddings, attention_maps, device='cpu', num_samples=5):
    """
    Evaluate model and visualize some predictions
    """
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        # Get random samples
        indices = np.random.choice(len(sentence_embeddings), num_samples, replace=False)
        sample_embeddings = torch.FloatTensor(sentence_embeddings[indices]).to(device)
        sample_targets = attention_maps[indices]
        
        # Generate predictions
        predictions = model(sample_embeddings).cpu().numpy()
        
        # Plot comparisons
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 2*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
            
        for i in range(num_samples):
            # Original attention map
            axes[i, 0].imshow(sample_targets[i], cmap='viridis', aspect='auto')
            axes[i, 0].set_title(f'Ground Truth {i+1}')
            axes[i, 0].axis('off')
            
            # Predicted attention map
            im = axes[i, 1].imshow(predictions[i], cmap='viridis', aspect='auto')
            axes[i, 1].set_title(f'Predicted {i+1}')
            axes[i, 1].axis('off')
            
            # Add colorbar to the last subplot
            if i == num_samples - 1:
                plt.colorbar(im, ax=axes[i, 1])
        
        plt.tight_layout()
        plt.show()
        
        # Calculate some metrics
        mse = np.mean((predictions - sample_targets)**2)
        print(f"MSE on sample: {mse:.6f}")