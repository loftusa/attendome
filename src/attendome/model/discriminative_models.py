import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class AttentionHeadDataset(Dataset):
    """Dataset for attention head embeddings"""
    
    def __init__(self, sent_emb, attn_maps, sentence_ids, layer_ids, head_ids, labels=None):
        self.sent_emb = sent_emb
        self.attn_maps = attn_maps
        self.sentence_ids = sentence_ids
        self.layer_ids = layer_ids
        self.head_ids = head_ids
        self.labels = labels
        
    def __len__(self):
        return len(self.sentence_ids)
    
    def __getitem__(self, idx):
        sent_id = self.sentence_ids[idx]
        layer_id = self.layer_ids[idx]
        head_id = self.head_ids[idx]
        
        # Get sentence embedding and attention map
        sent_embedding = self.sent_emb[sent_id]
        attn_map = self.attn_maps[layer_id][head_id][sent_id]
        
        # Flatten attention map if it's 2D
        if len(attn_map.shape) > 1:
            attn_map = attn_map.flatten()
        
        item = {
            'sent_embedding': torch.FloatTensor(sent_embedding),
            'attn_map': torch.FloatTensor(attn_map),
            'layer_id': layer_id,
            'head_id': head_id,
            'sent_id': sent_id
        }
        
        if self.labels is not None:
            item['label'] = torch.LongTensor([self.labels[idx]])[0]
            
        return item

class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning"""
    
    def __init__(self, sent_emb, attn_maps, sentence_ids, layer_ids, head_ids, num_negatives=5):
        self.sent_emb = sent_emb
        self.attn_maps = attn_maps
        self.sentence_ids = sentence_ids
        self.layer_ids = layer_ids
        self.head_ids = head_ids
        self.num_negatives = num_negatives
        
        # Create mapping from (layer, head) to sentence indices
        self.head_to_sentences = {}
        for i, (layer_id, head_id) in enumerate(zip(layer_ids, head_ids)):
            key = (layer_id, head_id)
            if key not in self.head_to_sentences:
                self.head_to_sentences[key] = []
            self.head_to_sentences[key].append(i)
        
        self.all_heads = list(self.head_to_sentences.keys())
        
    def __len__(self):
        return len(self.sentence_ids)
    
    def __getitem__(self, idx):
        anchor_sent_id = self.sentence_ids[idx]
        anchor_layer_id = self.layer_ids[idx]
        anchor_head_id = self.head_ids[idx]
        anchor_head_key = (anchor_layer_id, anchor_head_id)
        
        # Get anchor
        anchor_sent_emb = torch.FloatTensor(self.sent_emb[anchor_sent_id])
        anchor_attn_map = torch.FloatTensor(self.attn_maps[anchor_layer_id][anchor_head_id][anchor_sent_id])
        if len(anchor_attn_map.shape) > 1:
            anchor_attn_map = anchor_attn_map.flatten()
        
        # Get positive (same head, different sentence)
        positive_indices = self.head_to_sentences[anchor_head_key]
        positive_idx = random.choice([i for i in positive_indices if i != idx])
        pos_sent_id = self.sentence_ids[positive_idx]
        
        pos_sent_emb = torch.FloatTensor(self.sent_emb[pos_sent_id])
        pos_attn_map = torch.FloatTensor(self.attn_maps[anchor_layer_id][anchor_head_id][pos_sent_id])
        if len(pos_attn_map.shape) > 1:
            pos_attn_map = pos_attn_map.flatten()
        
        # Get negatives (different heads)
        negative_heads = [h for h in self.all_heads if h != anchor_head_key]
        selected_neg_heads = random.sample(negative_heads, min(self.num_negatives, len(negative_heads)))
        
        negatives = []
        for neg_layer, neg_head in selected_neg_heads:
            neg_indices = self.head_to_sentences[(neg_layer, neg_head)]
            neg_idx = random.choice(neg_indices)
            neg_sent_id = self.sentence_ids[neg_idx]
            
            neg_sent_emb = torch.FloatTensor(self.sent_emb[neg_sent_id])
            neg_attn_map = torch.FloatTensor(self.attn_maps[neg_layer][neg_head][neg_sent_id])
            if len(neg_attn_map.shape) > 1:
                neg_attn_map = neg_attn_map.flatten()
            
            negatives.append({
                'sent_embedding': neg_sent_emb,
                'attn_map': neg_attn_map
            })
        
        return {
            'anchor': {
                'sent_embedding': anchor_sent_emb,
                'attn_map': anchor_attn_map
            },
            'positive': {
                'sent_embedding': pos_sent_emb,
                'attn_map': pos_attn_map
            },
            'negatives': negatives
        }

class AttentionHeadEncoder(nn.Module):
    """Neural network to encode attention heads"""
    
    def __init__(self, sent_emb_dim, attn_map_dim, hidden_dim=512, output_dim=256):
        super().__init__()
        
        # Separate encoders for sentence embeddings and attention maps
        self.sent_encoder = nn.Sequential(
            nn.Linear(sent_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.attn_encoder = nn.Sequential(
            nn.Linear(attn_map_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, sent_embedding, attn_map):
        sent_encoded = self.sent_encoder(sent_embedding)
        attn_encoded = self.attn_encoder(attn_map)
        
        # Concatenate and fuse
        combined = torch.cat([sent_encoded, attn_encoded], dim=-1)
        output = self.fusion(combined)
        
        # L2 normalize for contrastive learning
        output = F.normalize(output, p=2, dim=-1)
        
        return output

class SupervisedClassifier(nn.Module):
    """Classifier for supervised learning"""
    
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.fusion[-1].out_features, num_classes)
        
    def forward(self, sent_embedding, attn_map):
        features = self.encoder(sent_embedding, attn_map)
        logits = self.classifier(features)
        return features, logits

class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning"""
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, anchor, positive, negatives):
        # anchor, positive: [batch_size, emb_dim]
        # negatives: [batch_size, num_negatives, emb_dim]
        
        batch_size = anchor.size(0)
        
        # Compute positive similarities
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature  # [batch_size]
        
        # Compute negative similarities
        neg_sim = torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1) / self.temperature  # [batch_size, num_negatives]
        
        # Combine positive and negative similarities
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [batch_size, 1 + num_negatives]
        
        # InfoNCE loss (positive should be first)
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
        loss = F.cross_entropy(logits, labels)
        
        return loss

def train_supervised_model(model, train_loader, val_loader, device, epochs=50):
    """Train supervised classification model"""
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print("Training supervised model...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            sent_emb = batch['sent_embedding'].to(device)
            attn_map = batch['attn_map'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            features, logits = model(sent_emb, attn_map)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # Validation
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                sent_emb = batch['sent_embedding'].to(device)
                attn_map = batch['attn_map'].to(device)
                labels = batch['label'].to(device)
                
                features, logits = model(sent_emb, attn_map)
                loss = criterion(logits, labels)
                
                total_val_loss += loss.item()
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_preds)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    
    return train_losses, val_losses, val_accuracies

def train_contrastive_model(model, train_loader, val_loader, device, epochs=50):
    """Train contrastive learning model"""
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = InfoNCELoss(temperature=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    train_losses = []
    val_losses = []
    
    print("Training contrastive model...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Get anchor, positive, and negatives
            anchor_sent = batch['anchor']['sent_embedding'].to(device)
            anchor_attn = batch['anchor']['attn_map'].to(device)
            
            pos_sent = batch['positive']['sent_embedding'].to(device)
            pos_attn = batch['positive']['attn_map'].to(device)
            
            # Process negatives
            batch_size = anchor_sent.size(0)
            num_negatives = len(batch['negatives'])
            
            neg_sent = torch.stack([batch['negatives'][i]['sent_embedding'] for i in range(num_negatives)], dim=1).to(device)
            neg_attn = torch.stack([batch['negatives'][i]['attn_map'] for i in range(num_negatives)], dim=1).to(device)
            
            optimizer.zero_grad()
            
            # Encode anchor and positive
            anchor_emb = model(anchor_sent, anchor_attn)
            pos_emb = model(pos_sent, pos_attn)
            
            # Encode negatives
            neg_emb = []
            for i in range(num_negatives):
                neg_emb_i = model(neg_sent[:, i], neg_attn[:, i])
                neg_emb.append(neg_emb_i)
            neg_emb = torch.stack(neg_emb, dim=1)
            
            # Compute loss
            loss = criterion(anchor_emb, pos_emb, neg_emb)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                anchor_sent = batch['anchor']['sent_embedding'].to(device)
                anchor_attn = batch['anchor']['attn_map'].to(device)
                
                pos_sent = batch['positive']['sent_embedding'].to(device)
                pos_attn = batch['positive']['attn_map'].to(device)
                
                num_negatives = len(batch['negatives'])
                neg_sent = torch.stack([batch['negatives'][i]['sent_embedding'] for i in range(num_negatives)], dim=1).to(device)
                neg_attn = torch.stack([batch['negatives'][i]['attn_map'] for i in range(num_negatives)], dim=1).to(device)
                
                anchor_emb = model(anchor_sent, anchor_attn)
                pos_emb = model(pos_sent, pos_attn)
                
                neg_emb = []
                for i in range(num_negatives):
                    neg_emb_i = model(neg_sent[:, i], neg_attn[:, i])
                    neg_emb.append(neg_emb_i)
                neg_emb = torch.stack(neg_emb, dim=1)
                
                loss = criterion(anchor_emb, pos_emb, neg_emb)
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return train_losses, val_losses

def evaluate_embeddings(model, test_loader, device, true_labels=None):
    """Evaluate learned embeddings using clustering metrics"""
    
    model.eval()
    all_embeddings = []
    all_labels = []
    all_head_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            sent_emb = batch['sent_embedding'].to(device)
            attn_map = batch['attn_map'].to(device)
            
            if hasattr(model, 'encoder'):
                embeddings = model.encoder(sent_emb, attn_map)
            else:
                embeddings = model(sent_emb, attn_map)
            
            all_embeddings.append(embeddings.cpu().numpy())
            
            if 'label' in batch:
                all_labels.extend(batch['label'].cpu().numpy())
            
            # Store head IDs for analysis
            head_ids = [(l.item(), h.item()) for l, h in zip(batch['layer_id'], batch['head_id'])]
            all_head_ids.extend(head_ids)
    
    all_embeddings = np.vstack(all_embeddings)
    
    # Perform clustering
    n_clusters = len(set(all_head_ids)) if true_labels is None else len(set(all_labels))
    n_clusters = min(n_clusters, 20)  # Limit for computational efficiency
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(all_embeddings)
    
    # Compute clustering metrics
    silhouette = silhouette_score(all_embeddings, cluster_labels)
    
    metrics = {'silhouette_score': silhouette}
    
    if all_labels:
        ari = adjusted_rand_score(all_labels, cluster_labels)
        metrics['adjusted_rand_index'] = ari
    
    return all_embeddings, cluster_labels, metrics