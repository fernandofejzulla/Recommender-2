import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import random

#Load and preprocess the MovieLens dataset
def load_movielens(path):
    df = pd.read_csv(path, sep="::", engine="python", names=["user", "item", "ratings", "timestamp"])

    #Convert explicit ratings(keep >=4)
    df = df[df["ratings"] >= 4]
    df = df[["user", "item", "timestamp"]]

    return df

#Generate chronological interaction sequences for each user
def build_user_sequence(df):
    df = df.sort_values(["user", "timestamp"])
    user_sequences = df.groupby("user")["item"].apply(list).to_dict()

    return user_sequences

#Filter out users with fewer than 5 interactions
def filter_users(user_sequences, min_length=5):
    before = len(user_sequences)
    filtered = {u: seq for u, seq in user_sequences.items() if len(seq) >= min_length}

    print(f"Users before filtering: {before}")
    print(f"Users after filtering (>= {min_length}): {len(filtered)}")

    return filtered

#Remaping item IDs to start from 1 because embedding layer uses padding
def remap_items(user_sequences):
    all_items = set()
    for seq in user_sequences.values():
        all_items.update(seq)
    
    item_map = {old: new for new, old in enumerate(sorted(all_items), start=1)}
    num_items = len(item_map)

    remapped = {u: [item_map[i] for i in seq] for u, seq in user_sequences.items()} 

    return remapped, num_items

#apply a leave-one-out split: Use all but the last two interactions for training [A,B,C], Use the second-to-last interaction for validation [D], Use the last interaction for testing [E]
def leave_one_out_split(user_sequences):
    train_seqs = {}
    valid_data = {}
    test_data ={}

    for user, seq in user_sequences.items():
        train_seqs[user] = seq[:-2]
        valid_data[user] = (seq[:-2], seq[-2])
        test_data[user] = (seq[:-1], seq[-1])

    return train_seqs, valid_data, test_data

#Construct input–target pairs for next-item prediction based on user sequences
#padding/truncation helper
def pad_or_trun(sequence, maxlen):
    if len(sequence) >= maxlen:
        return sequence[-maxlen:]
    else:
        return [0] * (maxlen - len(sequence)) + sequence
    
#pytorch data for training
class SASRecTrainDataset(Dataset):
    def __init__(self, user_train_seqs, num_items, maxlen):
        self.users = list(user_train_seqs.keys())
        self.user_train_seqs = user_train_seqs
        self.num_items = num_items
        self.maxlen = maxlen

    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        user = self.users[idx]
        seq = self.user_train_seqs[user]

        input_ids = pad_or_trun(seq[:-1], self.maxlen)
        target_ids = pad_or_trun(seq[1:], self.maxlen)

        user_items = set(seq)
        negatives = []
        for _ in range(self.maxlen):
            neg = random.randint(1, self.num_items)
            while neg in user_items:
                neg = random.randint(1, self.num_items)
            negatives.append(neg)

        return {"input_seq": torch.tensor(input_ids, dtype=torch.long), "target_seq": torch.tensor(target_ids, dtype=torch.long), "negatives": torch.tensor(negatives, dtype=torch.long)}
    
