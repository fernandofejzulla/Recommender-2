import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

from evaluation import evaluate

#Train SASRec for one epoch
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        input_sequence = batch["input_seq"].to(device)
        target_sequence = batch["target_seq"].to(device)
        negatives = batch["negatives"].to(device)

        #get scores for positive (actual next time)
        positive_scores = model.predict(input_sequence , target_sequence)

        #get scores for negative (random items)
        negative_scores = model.predict(input_sequence, negatives)

        #masking
        #computing loss where target is not padding (0)
        mask = (target_sequence != 0).float()

        #loss function
        #binary cross-entropy
        positive_loss = -torch.nn.functional.logsigmoid(positive_scores) * mask
        negative_loss = -torch.nn.functional.logsigmoid(- negative_scores) * mask
        #avg
        loss = (positive_loss + negative_loss).sum() / mask.sum()

        #backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def train_model(model, train_dataset, valid_data, user_sequences, num_items, maxlen, device, epochs=200, batch_size=128, lr=0.001, patience=5, save_prefix=""):

    #Implement early stopping based on validation
    #Learning rate and batch size follow the paper's recommendations

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    best_ndcg = 0.0
    patience_counter = 0
    train_losses = []
    valuation_ndcgs = []

    for epoch in range(epochs):
        #Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)

        #evaluate on validation set
        metrics = evaluate(model, valid_data, user_sequences, num_items, maxlen, device, ks=[10, 20])
        val_ndcg10 = metrics["NDCG@10"]
        valuation_ndcgs.append(val_ndcg10)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Recall@10: {metrics['Recall@10']:.4f}  Recall@20: {metrics['Recall@20']:.4f}")
        print(f"  Val NDCG@10:   {metrics['NDCG@10']:.4f}  NDCG@20:   {metrics['NDCG@20']:.4f}")

        #early stopping 
        if val_ndcg10 > best_ndcg:
            best_ndcg = val_ndcg10
            patience_counter = 0
            torch.save(model.state_dict(), f"best_sasrec_model_{save_prefix}.pth")
            print(f"  → New best NDCG@10! Model saved.")
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    #Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.set_title("Training Loss")

    ax2.plot(valuation_ndcgs)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("NDCG@10")
    ax2.set_title("Validation NDCG@10")
    
    plt.tight_layout()
    plt.savefig(f"training_curves_{save_prefix}.png")
    print(f"\nTraining curves saved to training_curves_{save_prefix}.png")

    return train_losses, valuation_ndcgs