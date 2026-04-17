import torch
import numpy as np
from tqdm import tqdm 
from data_prepro import pad_or_trun

def recall_at_k(ranked_items, target_item, k):
    #is the target item in the top-k recommendations?
    # we have 1 target item per use (leave-one-out)
    return 1.0 if target_item in ranked_items[:k] else 0.0

def ncdg_at_k(ranked_items, target_item, k):
    #NDCG@k: Position-aware metric — rewards finding the target higher in the list.
    top_k = ranked_items[:k]
    for i, item in enumerate(top_k):
        if item == target_item:
            return 1.0 / np.log2(i + 2)  # i+2 because i starts at 0 and we want log2(1) for the top item
    return 0.0
    
def evaluate(model, evaluate_data, user_sequences, num_items, maxlen, device, ks=[10, 20]):
    #Evaluate the model on validation or test data.
    model.eval()

    #Initialize metrics
    metrics = {}
    for k in ks:
        metrics[f"Recall@{k}"] = []
        metrics[f"NDCG@{k}"] = []

    #all item embeddings
    all_item_ids = torch.arange(1, num_items + 1).to(device=device)

    with torch.no_grad():
        for user, (input_seq, target_item) in tqdm(evaluate_data.items(), desc="Evaluating"):
            
            #pad/truncate the input sequence to maxlen
            padded_sequence = pad_or_trun(input_seq, maxlen)
            sequence_tensor = torch.tensor([padded_sequence], dtype=torch.long).to(device=device)

            #forward pass
            hidden = model(sequence_tensor)

            last_hidden = hidden[:, -1, :] # last hidden state

            all_item_embeddings = model.item_embedding(all_item_ids) 
            scores = torch.matmul(last_hidden, all_item_embeddings.T).squeeze(0) # scores for all items

            #masking out items that are already watched
            user_history = set(user_sequences[user])
            for item in user_history:
                if item != target_item:
                    scores[item - 1] = -np.inf

            #ranking items by score
            ranked_items = torch.argsort(scores, descending=True).cpu().numpy() + 1

            #compute metrics
            for k in ks:
                metrics[f"Recall@{k}"].append(recall_at_k(ranked_items, target_item, k))
                metrics[f"NDCG@{k}"].append(ncdg_at_k(ranked_items, target_item, k))

    for key in metrics:
        metrics[key] = np.mean(metrics[key])

    return metrics 
    