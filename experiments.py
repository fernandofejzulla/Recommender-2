import torch
import json
import time

from data_prepro import remap_items, leave_one_out_split, SASRecTrainDataset, load_movielens, build_user_sequence, filter_users
from SASRec_model import SASRec
from train import train_model
from evaluation import evaluate

DATA_PATH = "./ml-1m/ratings.dat"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#load and preprocess the data
print("loading data...")

df = load_movielens(DATA_PATH)
user_sequences = build_user_sequence(df)
user_sequences = filter_users(user_sequences, min_length=5)
user_sequences, num_items = remap_items(user_sequences)

train_sequences, valid_data, test_data = leave_one_out_split(user_sequences)
print(f"Dataset: {len(user_sequences)} users, {num_items} items")

#experiments
experiments =[
    {"name": "baseline", "blocks": 2, "hidden": 50, "heads": 1, "maxlen": 200},
    {"name": "experiment_1", "blocks": 1, "hidden": 50, "heads": 1, "maxlen": 200},
    {"name": "experiment_2", "blocks": 3, "hidden": 50, "heads": 1, "maxlen": 200},
    {"name": "experiment_3", "blocks": 2, "hidden": 32, "heads": 1, "maxlen": 200},
    {"name": "experiment_4", "blocks": 2, "hidden": 50, "heads": 2, "maxlen": 200},
    {"name": "experiment_5", "blocks": 2, "hidden": 50, "heads": 1, "maxlen": 50}
]

all_results = []

for exp in experiments:
    print("=" * 60)
    print(f"Experiment: {exp['name']}")
    print(f"  blocks={exp['blocks']}, hidden={exp['hidden']}, heads={exp['heads']}, maxlen={exp['maxlen']}")
    print("=" * 60)

    start = time.time()

    #create dataset (maxlen)
    train_dataset = SASRecTrainDataset(train_sequences, num_items, maxlen=exp["maxlen"])

    #create model
    model = SASRec(item_num=num_items, maxlen=exp["maxlen"], hidden_units=exp["hidden"], num_blocks=exp["blocks"], num_heads=exp["heads"],dropout_rate=0.2)

    #train model
    train_model(model=model, train_dataset=train_dataset, valid_data=valid_data, user_sequences=user_sequences, num_items=num_items, maxlen=exp["maxlen"], device=DEVICE, epochs=30, batch_size=128, lr=0.001, patience=3, save_prefix=exp["name"])

    #finale test
    model.load_state_dict(torch.load(f"best_sasrec_model_{exp['name']}.pth", map_location=DEVICE))
    model.to(DEVICE)
    test_metrics = evaluate(model, test_data, user_sequences, num_items, exp["maxlen"], DEVICE, ks=[10, 20])

    elapsed_time = time.time() - start

    result = {"name": exp["name"], "blocks": exp["blocks"], "hidden": exp["hidden"], "heads": exp["heads"], "maxlen": exp["maxlen"], "num_params": sum(p.numel() for p in model.parameters()), "time_seconds": round(elapsed_time, 1), **{k: round(v, 4) for k, v in test_metrics.items()}}

    all_results.append(result)

    print(f"\n{exp['name']} DONE in {elapsed_time:.1f}s")
    print(f"  Recall@10: {test_metrics['Recall@10']:.4f}")
    print(f"  NDCG@10:   {test_metrics['NDCG@10']:.4f}\n")

#save results
with open("comparison_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

#comparisson table
print("\n" + "=" * 100)
print("COMPARISON TABLE")
print("=" * 100)
print(f"{'Name':<15} {'Blocks':<8} {'Hidden':<8} {'Heads':<7} {'MaxLen':<8} "
      f"{'Params':<10} {'R@10':<8} {'R@20':<8} {'N@10':<8} {'N@20':<8}")
print("-" * 100)
for r in all_results:
    print(f"{r['name']:<15} {r['blocks']:<8} {r['hidden']:<8} {r['heads']:<7} {r['maxlen']:<8} "
          f"{r['num_params']:<10,} {r['Recall@10']:<8.4f} {r['Recall@20']:<8.4f} "
          f"{r['NDCG@10']:<8.4f} {r['NDCG@20']:<8.4f}")
print("=" * 100)
