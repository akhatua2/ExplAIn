import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def plot_sharpness_across_tokens(scores, output_file="sharpness_across_tokens.png"):
    """
    Plot sharpness across token positions and save to file.
    
    Args:
        scores: torch.Tensor of shape [num_tokens, vocab_size] - the raw logits
        output_file: str - filename to save the plot
    """
    # Calculate sharpness
    step_logprobs = F.log_softmax(scores, dim=-1)
    probs = torch.exp(step_logprobs)
    entropy = -torch.sum(probs * step_logprobs, dim=-1)
    sharpness = -entropy
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(sharpness)), sharpness.cpu().numpy(), 'b-', linewidth=2)
    plt.xlabel('Token Position')
    plt.ylabel('Sharpness (higher = sharper)')
    plt.title('Distribution Sharpness Across Generated Tokens')
    plt.grid(True, alpha=0.3)
    
    # Add some stats to the plot
    avg_sharpness = torch.mean(sharpness).item()
    plt.axhline(y=avg_sharpness, color='r', linestyle='--', alpha=0.7, 
                label=f'Average: {avg_sharpness:.3f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved sharpness plot to {output_file}")

    return sharpness


def plot_topk_distributions(scores, token_positions=None, k=20, output_file="topk_distributions.png"):
    """
    Plot top-k probability distributions for selected tokens.
    
    Args:
        scores: torch.Tensor of shape [num_tokens, vocab_size] - the raw logits
        token_positions: list of int - which token positions to show (default: evenly spaced)
        k: int - number of top probabilities to show
        output_file: str - filename to save the plot
    """
    # Calculate probabilities and sharpness
    step_logprobs = F.log_softmax(scores, dim=-1)
    probs = torch.exp(step_logprobs)
    entropy = -torch.sum(probs * step_logprobs, dim=-1)
    sharpness = -entropy
    
    # Default token positions if not provided
    if token_positions is None:
        num_tokens = len(sharpness)
        token_positions = [0, num_tokens//4, num_tokens//2, 3*num_tokens//4, num_tokens-1]
    
    # Create subplots
    fig, axes = plt.subplots(1, len(token_positions), figsize=(4*len(token_positions), 4))
    if len(token_positions) == 1:
        axes = [axes]
    
    for idx, pos in enumerate(token_positions):
        # Get top-k probabilities
        top_k_probs, top_k_indices = torch.topk(probs[pos], k)
        
        axes[idx].bar(range(k), top_k_probs.cpu().numpy())
        axes[idx].set_title(f'Token {pos}\nSharpness: {sharpness[pos]:.3f}')
        axes[idx].set_xlabel('Top-K Tokens')
        axes[idx].set_ylabel('Probability')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved top-k distributions to {output_file}")


def plot_topk_heatmap(scores, k=10, output_file="topk_heatmap.png"):
    """
    Create a heatmap showing top-k probabilities across all tokens.
    
    Args:
        scores: torch.Tensor of shape [num_tokens, vocab_size] - the raw logits
        k: int - number of top probabilities to show
        output_file: str - filename to save the plot
    """
    # Calculate probabilities and sharpness
    step_logprobs = F.log_softmax(scores, dim=-1)
    probs = torch.exp(step_logprobs)
    entropy = -torch.sum(probs * step_logprobs, dim=-1)
    sharpness = -entropy
    
    # Create top-k matrix
    top_k_matrix = torch.zeros(len(sharpness), k)
    for i in range(len(sharpness)):
        top_k_probs, _ = torch.topk(probs[i], k)
        top_k_matrix[i] = top_k_probs
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    im = plt.imshow(top_k_matrix.cpu().numpy(), aspect='auto', cmap='viridis')
    plt.colorbar(im, label='Probability')
    plt.xlabel('Top-K Rank')
    plt.ylabel('Token Position')
    plt.title(f'Top-{k} Probabilities Across All Tokens')
    
    # Add average sharpness info
    avg_sharpness = torch.mean(sharpness).item()
    plt.figtext(0.02, 0.02, f'Average Sharpness: {avg_sharpness:.3f}', 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved top-k heatmap to {output_file}")