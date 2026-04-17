import torch 
import os

from data_prepro import (load_movielens, build_user_sequence, filter_users, remap_items, leave_one_out_split, SASRecTrainDataset)
from SASRec_model import SASRec
from train import train_model
from evaluation import evaluate

#configuration
#hyperparameters from SASRec paper
DATA_PATH = "./ml-1m/ratings.dat"

MAX_LEN = 200
HIDDEN_UNITS = 50
NUM_BLOCKS = 2
NUM_HEADS = 1
DROPOUT_RATE = 0.2

BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 200
PATIENCE = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#1) Data Processing
print("\n" + "=" * 5)
print("1) Data Processing")
print("=" * 50)

# Load and filter to implicit feedback
print("\nLoading MovieLens 1M...")
df = load_movielens(DATA_PATH)
print(f"  Positive interactions: {len(df)}")

# Build chronological sequences
print("\nBuilding user sequences...")
user_sequences = build_user_sequence(df)

# Filter short sequences
print("\nFiltering short sequences...")
user_sequences = filter_users(user_sequences, min_length=5)

# Remap item IDs (0 = padding, 1..N = items)
print("\nRemapping item IDs...")
user_sequences, num_items = remap_items(user_sequences)
num_users = len(user_sequences)
print(f"  Number of users: {num_users}")

# Leave-one-out split
print("\nSplitting data (leave-one-out)...")
train_seqs, valid_data, test_data = leave_one_out_split(user_sequences)
print(f"  Train: {len(train_seqs)} users")
print(f"  Valid: {len(valid_data)} users (1 target item each)")
print(f"  Test:  {len(test_data)} users (1 target item each)")
 
# Create training dataset
train_dataset = SASRecTrainDataset(train_seqs, num_items, maxlen=MAX_LEN)

#2) SASRec Model
print("\n" + "=" * 50)
print("STEP 2: Creating SASRec Model")
print("=" * 50)
 
model = SASRec(
    item_num=num_items,
    maxlen=MAX_LEN,
    hidden_units=HIDDEN_UNITS,
    num_blocks=NUM_BLOCKS,
    num_heads=NUM_HEADS,
    dropout_rate=DROPOUT_RATE,
)
 
total_params = sum(p.numel() for p in model.parameters())
print(f"  Model parameters: {total_params:,}")
print(f"  Config: blocks={NUM_BLOCKS}, hidden={HIDDEN_UNITS}, heads={NUM_HEADS}")

#3) Training & Optimization
print("\n" + "=" * 50)
print("STEP 3: Training")
print("=" * 50)
 
train_losses, val_ndcgs = train_model(
    model=model,
    train_dataset=train_dataset,
    valid_data=valid_data,
    user_sequences=user_sequences,
    num_items=num_items,
    maxlen=MAX_LEN,
    device=DEVICE,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    patience=PATIENCE,
)

#4) Evaluation
print("\n" + "=" * 50)
print("STEP 4: Final Evaluation on Test Set")
print("=" * 50)
 
# Load best model
model.load_state_dict(torch.load("best_sasrec_model.pth", map_location=DEVICE))
model.to(DEVICE)

test_metrics = evaluate(model, test_data, user_sequences, num_items, MAX_LEN, DEVICE, ks=[10, 20])

print("\nFinal Test Metrics:")
print(f"Recall@10:  {test_metrics['Recall@10']:.4f}")
print(f"Recall@20:  {test_metrics['Recall@20']:.4f}")
print(f"NDCG@10:    {test_metrics['NDCG@10']:.4f}")
print(f"NDCG@20:    {test_metrics['NDCG@20']:.4f}")

# Save results
with open("results.txt", "w") as f:
    f.write(f"Configuration:\n")
    f.write(f"  maxlen={MAX_LEN}, hidden={HIDDEN_UNITS}, blocks={NUM_BLOCKS}, heads={NUM_HEADS}\n")
    f.write(f"  dropout={DROPOUT_RATE}, lr={LEARNING_RATE}, batch_size={BATCH_SIZE}\n\n")
    f.write(f"Test Results:\n")
    for k, v in test_metrics.items():
        f.write(f"  {k}: {v:.4f}\n")

print("Done")
