import torch
import torch.nn.functional as F


def temperature_sampling(logits, temperature):
    scaled_logits = logits / temperature
    return F.softmax(scaled_logits, dim=-1)

def top_k_filtering(logits, top_k):
    if top_k > 0:
        values, _ = torch.topk(logits, top_k, dim=-1)
        min_value = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_value, torch.tensor(float('-inf')).to(logits.device), logits)
    return logits

def top_p_filtering(logits, top_p):
    if top_p > 0.0 and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    return logits

def apply_ngram_penalty(logits, generated_tokens, window_size=20, penalty_factor=1.2):
    """
    Apply n-gram penalty: P'(x_i) = P(x_i) / penalty_factor if x_i in last w tokens

    Args:
        logits: Model logits before softmax
        generated_tokens: List of previously generated token IDs
        window_size: Look back w tokens
        penalty_factor: Penalty strength (> 1.0 reduces probability of repeated tokens)
    """
    if len(generated_tokens) == 0 or penalty_factor == 1.0:
        return logits

    # Get tokens in the window
    window = generated_tokens[-window_size:] if len(generated_tokens) > window_size else generated_tokens

    # Apply penalty by subtracting log(penalty_factor) from logits
    # This is equivalent to dividing probabilities by penalty_factor
    penalty = torch.log(torch.tensor(penalty_factor, device=logits.device))

    for token_id in set(window):
        logits[0, token_id] = logits[0, token_id] - penalty

    return logits
