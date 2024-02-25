import numpy as np
import torch
from torch.nn.functional import log_softmax

def beam_search_decode(model, input_ids, attach_emb, max_length, beam_size=3, temperature=1.0):
    """Performs autoregressive decoding using beam search."""
    device = input_ids.device
    input_ids = input_ids.unsqueeze(0)
    input_length = input_ids.size(1)

    # Initialize beam search
    beam_scores = torch.zeros(1, beam_size, device=device)
    beam_seqs = torch.zeros(1, beam_size, input_length + max_length, dtype=torch.long, device=device)
    beam_seqs[:, :, :input_length] = input_ids
    finished_seqs = []

    for step in range(max_length):
        if len(finished_seqs) == beam_size:
            break

        next_candidates = []
        for i in range(beam_size):
            input_ids = beam_seqs[:, i, :input_length + step]
            logits = model(input_ids).squeeze(0) / temperature
            log_probs = log_softmax(logits, dim=-1)

            # Select top k candidates from current beam
            scores, indices = torch.topk(log_probs[-1, :], beam_size)
            print(scores, 'scores_first')
            scores = scores + beam_scores[:, i]
            print(scores, 'scores')
            for j in range(beam_size):
                score = scores[j].item()
                if score > -float("inf"):
                    next_candidates.append((score, indices[j].item(), i))

        # Select top beam_size candidates among all candidates
        next_candidates.sort(reverse=True)
        next_candidates = next_candidates[:beam_size]

        beam_scores_new = torch.zeros(1, beam_size, device=device)
        beam_seqs_new = torch.zeros(1, beam_size, input_length + max_length, dtype=torch.long, device=device)

        for idx, (score, token, beam_idx) in enumerate(next_candidates):
            beam_scores_new[:, idx] = score
            beam_seqs_new[:, idx, :input_length + step] = beam_seqs[:, beam_idx, :input_length + step]
            beam_seqs_new[:, idx, input_length + step] = token

        beam_scores = beam_scores_new
        beam_seqs = beam_seqs_new

    best_seq = beam_seqs[0, 0, input_length:].tolist()
    return best_seq

# # Example usage
# # Assuming model and inputs are defined
# # model = YourGPTModel(...)
# # inputs = YourInputTensor

# # Generate sequences using beam search
# beam_width = 3
# n_samples = 5
# generated_sequences = beam_search_decode(model, inputs, beam_width, n_samples)
# print(generated_sequences)



def get_top_k_probs(logits, k, temperature=1.0):
    # Apply temperature scaling
    scaled_logits = logits / temperature
    
    # Compute probabilities using softmax
    probabilities = np.exp(scaled_logits - np.max(scaled_logits)) / np.sum(np.exp(scaled_logits - np.max(scaled_logits)))
    
    # Find the top k indices
    top_k_indices = np.argsort(probabilities)[::-1][:k]
    
    # Normalize probabilities for sampling
    top_k_probs = probabilities[top_k_indices]
    # top_k_probs /= np.sum(top_k_probs)
    
    return top_k_probs

def greedy_sampling(logits):
    # Find the token index with the highest probability
    sampled_index = np.argmax(logits)
    
    return sampled_index

def top_k_sampling(logits, k, temperature=1.0):
    # Apply temperature scaling
    scaled_logits = logits / temperature
    
    # Compute probabilities using softmax
    probabilities = np.exp(scaled_logits - np.max(scaled_logits)) / np.sum(np.exp(scaled_logits - np.max(scaled_logits)))
    
    # Find the top k indices
    top_k_indices = np.argsort(probabilities)[::-1][:k]
    
    # Normalize probabilities for sampling
    top_k_probs = probabilities[top_k_indices]
    top_k_probs /= np.sum(top_k_probs)
    
    # Sample token index based on probabilities corresponding to top k indices
    sampled_index = np.random.choice(top_k_indices, p=top_k_probs)
    
    return sampled_index


def top_p_sampling(logits, p, temperature=1.0):
    # Apply temperature scaling
    scaled_logits = logits / temperature
    
    # Compute probabilities using softmax
    probabilities = np.exp(scaled_logits - np.max(scaled_logits)) / np.sum(np.exp(scaled_logits - np.max(scaled_logits)))
    # Sort probabilities and indices in decreasing order
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_probs = probabilities[sorted_indices]
    # Compute cumulative probabilities
    cumulative_probs = np.cumsum(sorted_probs)
    
    # Find the top p% indices
    if np.max(probabilities) > p:
        # If the highest probability is less than p, include it to ensure at least one token is selected
        selected_indices = sorted_indices[:1]
    else:
        selected_indices = sorted_indices[cumulative_probs <= p]
    
    # Normalize probabilities for sampling
    print(selected_indices, 'selected_indices')
    selected_probs = probabilities[selected_indices]
    selected_probs /= np.sum(selected_probs)
    # Sample token index based on probabilities corresponding to selected indices
    sampled_index = np.random.choice(selected_indices, p=selected_probs)
    
    return sampled_index

# Example usage
# logits = np.random.rand(1000)  # Example logits of shape 1000
# p = 0.1  # Example value for p
# temperature = 1.5  # Example temperature parameter
# sampled_index = top_p_sampling(logits, p, temperature)
# print(sampled_index)
