import numpy as np
import torch
from torch.nn.functional import log_softmax

def beam_search(start_token, model, attach_emb, max_len, device, beam_size, temperature):
    # Initialize beam search variables
    beam = [(torch.tensor([start_token], device=device), 0)]  # (sequence, cumulative log probability)

    # Perform beam search
    for _ in range(max_len):
        new_beam = []

        for seq, seq_score in beam:
            with torch.no_grad():
                x = seq.unsqueeze(0).to(device)
                outs = model(x, None, attach_emb, [0, 0])
                logits = outs[0, -1, :] / temperature
                log_probs = torch.log_softmax(logits, dim=-1)

            # Get top candidates using beam search
            top_log_probs, top_indices = torch.topk(log_probs, beam_size, dim=-1)

            for log_prob, index in zip(top_log_probs.squeeze().tolist(), top_indices.squeeze().tolist()):
                new_seq = torch.cat((seq, torch.tensor([index], device=device)), dim=0)
                new_score = seq_score + log_prob
                new_beam.append((new_seq, new_score))

        # Sort and select top sequences from the beam
        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_size]

    # Select the sequence with the highest score
    best_seq, _ = max(beam, key=lambda x: x[1])

    return best_seq.tolist()[1:]  # Convert to list

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

def random_sampling(logits):
    # Sample token index randomly
    sampled_index = np.random.choice(len(logits))
    return sampled_index

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
