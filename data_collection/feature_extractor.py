"""Feature and label extraction helpers for data collection."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch


def _flatten_tokens(draft_tokens: torch.Tensor) -> List[int]:
    if draft_tokens.dim() == 2 and draft_tokens.shape[0] == 1:
        return draft_tokens[0].detach().cpu().tolist()
    return draft_tokens.detach().cpu().view(-1).tolist()


def extract_node_features(
    draft_tokens: torch.Tensor,
    retrieve_indices: torch.Tensor,
    tree_position_ids: torch.Tensor,
    debug_info: Optional[Dict] = None,
) -> List[Dict]:
    tokens = _flatten_tokens(draft_tokens)
    depths = tree_position_ids.detach().cpu().view(-1).tolist()
    num_nodes = min(len(tokens), len(depths))
    tokens = tokens[:num_nodes]

    # Placeholder sibling ranks until deeper instrumentation is added.
    sibling_ranks = [0] * num_nodes

    # Populate draft probabilities when debug payload is available; fallback to zeros.
    draft_probs = [0.0] * num_nodes
    if debug_info is not None and "scores_list" in debug_info:
        try:
            scores_list = [t.detach().cpu() for t in debug_info["scores_list"]]
            scores_flat = torch.cat(scores_list, dim=0).view(-1)

            total_tokens = max(0, num_nodes - 1)  # exclude root node
            if total_tokens > 0:
                top_scores = torch.topk(scores_flat, total_tokens, dim=-1)
                sorted_idx = torch.sort(top_scores.indices).values
                selected = scores_flat[sorted_idx].exp().tolist()

                # Root token probability is undefined; set to 1.0 for convenience.
                draft_probs = [1.0] + selected
                draft_probs = draft_probs[:num_nodes]
        except Exception:
            # Leave draft_probs as zeros if reconstruction fails
            pass

    features: List[Dict] = []
    for idx in range(num_nodes):
        features.append(
            {
                "draft_token_id": tokens[idx],
                "draft_prob": draft_probs[idx],
                "tree_depth": depths[idx],
                "sibling_rank": sibling_ranks[idx],
            }
        )
    return features


def extract_node_labels(
    draft_tokens: torch.Tensor,
    retrieve_indices: torch.Tensor,
    best_candidate: int,
    accept_length: int,
) -> List[bool]:
    tokens = _flatten_tokens(draft_tokens)
    num_nodes = len(tokens)
    labels = [False] * num_nodes

    if retrieve_indices.numel() == 0:
        return labels

    winning_path = retrieve_indices[best_candidate]
    # Align with update_inference_inputs: accept_length counts accepted tokens (exclude root)
    max_depth = min(accept_length, winning_path.shape[0] - 1)
    for depth in range(1, max_depth + 1):
        node_idx = int(winning_path[depth].item())
        if 0 <= node_idx < num_nodes:
            labels[node_idx] = True
    return labels
