"""
Subtree Pruner for EAGLE-3 speculative decoding.

This module implements pruning of draft tokens before the expensive
tree_decoding forward pass, reducing the number of tokens the target
model must verify.

Two pruner modes:
  - OraclePruner: Uses pre-collected ground-truth labels (is_subtree_wasted)
    from nodes.parquet to prune with perfect knowledge. Used for ceiling
    measurement (Step 0 of the generalization roadmap).
  - ThresholdPruner: Uses a cumulative_logprob threshold to decide which
    subtrees to prune. (Future — Step 1+)

The core tensor surgery (rebuild_tensors) is shared across all modes.
"""
import torch
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Path utilities (shared between build_tables and pruner)
# ---------------------------------------------------------------------------

def build_token_paths_from_tensors(
    draft_tokens: torch.Tensor,
    tree_mask: torch.Tensor,
) -> Dict[int, str]:
    """
    Build root-to-node token path for every node in the live draft tree.

    Parameters
    ----------
    draft_tokens : [1, N]  live draft token IDs
    tree_mask    : [1, 1, N, N]  attention mask (tree_mask[0,0,i,j]==1 means j is ancestor of i)

    Returns
    -------
    {node_idx: token_path_str}  e.g. {4: "791|3752|315|24101"}
    """
    N = draft_tokens.shape[1]
    tokens = draft_tokens[0]  # [N]
    mask_2d = tree_mask[0, 0].bool()  # [N, N]  mask_2d[i, j] = j is ancestor of i

    path_map: Dict[int, str] = {}
    for node_idx in range(N):
        # Ancestors of node_idx = columns where mask_2d[node_idx, :] is True,
        # sorted by depth (depth proxy: number of ancestors each ancestor has)
        ancestor_mask = mask_2d[node_idx]  # [N], True for node_idx's own ancestors
        ancestor_indices = torch.where(ancestor_mask)[0].tolist()
        # Sort ancestors by their own ancestor count (= their depth, since
        # root has 1 ancestor, depth-1 nodes have 2, etc.)
        ancestor_indices.sort(key=lambda a: int(mask_2d[a].sum().item()))
        path_tokens = [int(tokens[a].item()) for a in ancestor_indices]
        path_map[node_idx] = "|".join(str(t) for t in path_tokens)

    return path_map


# ---------------------------------------------------------------------------
# Tensor Surgery
# ---------------------------------------------------------------------------

def rebuild_tensors(
    draft_tokens: torch.Tensor,
    retrieve_indices: torch.Tensor,
    tree_mask: torch.Tensor,
    tree_position_ids: torch.Tensor,
    prune_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Remove pruned nodes from the draft tree and rebuild all 4 tensors.

    Parameters
    ----------
    draft_tokens : [1, N]      Token IDs for the draft tree.
    retrieve_indices : [L, D]  Candidate paths (L leaves × max depth D).
    tree_mask : [1, 1, N, N]   Attention mask encoding tree ancestry.
    tree_position_ids : [N]    Depth of each node.
    prune_mask : [N]           Boolean tensor. True = PRUNE this node.

    Returns
    -------
    (draft_tokens', retrieve_indices', tree_mask', tree_position_ids')
    with all pruned nodes removed and indices remapped.

    Invariants
    ----------
    - Node 0 (root / sample_token) is NEVER pruned.
    - If a parent is pruned, all its descendants are also pruned
      (enforced internally via enforce_subtree_consistency).
    - retrieve_indices rows that pass through a pruned node are dropped.
    - Surviving index values are remapped to the new compact positions.
    """
    draft_device = draft_tokens.device
    N = draft_tokens.shape[1]

    # Safety: never prune node 0 (root)
    prune_mask = prune_mask.clone()
    prune_mask[0] = False

    # Enforce subtree consistency using tree_mask:
    # tree_mask[0,0,i,j] == 1 means node i can attend to node j,
    # i.e., j is an ancestor of i (or i itself).
    # If parent j is pruned, all descendants i (where tree_mask[:,,:,j]==1
    # and i != j) must also be pruned.
    prune_mask = _enforce_subtree_consistency(prune_mask, tree_mask)

    # Build keep mask (True = keep this node)
    keep_mask = ~prune_mask  # [N]
    keep_indices = torch.where(keep_mask)[0]  # sorted indices of kept nodes
    M = keep_indices.shape[0]

    if M == N:
        # Nothing to prune — return originals unchanged
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids

    # Build old→new index remap on retrieve_indices device
    # remap[old_idx] = new_idx, or -1 if pruned
    retrieve_device = retrieve_indices.device
    remap = torch.full((N,), -1, dtype=torch.long, device=retrieve_device)
    remap[keep_indices.to(retrieve_device)] = torch.arange(M, dtype=torch.long, device=retrieve_device)

    # 1. Rebuild draft_tokens: [1, N] → [1, M]
    new_draft_tokens = draft_tokens[:, keep_indices.to(draft_tokens.device)]

    # 2. Rebuild tree_mask: [1, 1, N, N] → [1, 1, M, M]
    keep_indices_tree = keep_indices.to(tree_mask.device)
    new_tree_mask = tree_mask[:, :, keep_indices_tree][:, :, :, keep_indices_tree]

    # 3. Rebuild tree_position_ids: [N] → [M]
    new_tree_position_ids = tree_position_ids[keep_indices.to(tree_position_ids.device)]

    # 4. Rebuild retrieve_indices: [L, D] → [L', D']
    new_retrieve_indices = _rebuild_retrieve_indices(
        retrieve_indices,
        prune_mask.to(retrieve_device),
        remap,
        M,
    )

    return (
        new_draft_tokens.to(draft_device),
        new_retrieve_indices.to(draft_device),
        new_tree_mask.to(draft_device),
        new_tree_position_ids.to(draft_device),
    )


def _enforce_subtree_consistency(
    prune_mask: torch.Tensor,
    tree_mask: torch.Tensor,
) -> torch.Tensor:
    """
    If a node is pruned, ensure ALL its descendants are also pruned.

    Uses tree_mask to identify descendants: node i is a descendant of node j
    iff tree_mask[0, 0, i, j] == 1 (and i != j).
    """
    N = prune_mask.shape[0]
    device = prune_mask.device
    # tree_mask_2d[i, j] = True means j is an ancestor of i
    tree_mask_2d = tree_mask[0, 0].bool().to(device)  # [N, N] - ensure same device

    pruned_indices = torch.where(prune_mask)[0]
    for j in pruned_indices:
        # All nodes that have j as an ancestor (column j is True) → prune them
        descendants = tree_mask_2d[:, j].clone()
        descendants[j] = False  # exclude self (already marked)
        prune_mask = prune_mask | descendants

    # Safety: never prune root
    prune_mask[0] = False
    return prune_mask


def _rebuild_retrieve_indices(
    retrieve_indices: torch.Tensor,
    prune_mask: torch.Tensor,
    remap: torch.Tensor,
    new_size: int,
) -> torch.Tensor:
    """
    Rebuild retrieve_indices after pruning.

    1. Drop any row (candidate path) that passes through a pruned node.
    2. Remap surviving node indices to new compact positions.
    3. Re-pad to the new max depth.
    """
    L, D = retrieve_indices.shape
    device = retrieve_indices.device

    surviving_rows = []
    for row_idx in range(L):
        path = retrieve_indices[row_idx]
        valid = path[path >= 0]  # remove -1 padding

        # Check if ANY node in this path was pruned
        path_pruned = False
        for node_idx in valid:
            if prune_mask[node_idx.item()]:
                path_pruned = True
                break

        if not path_pruned:
            # Remap indices
            remapped = remap[valid]
            surviving_rows.append(remapped)

    if len(surviving_rows) == 0:
        # Edge case: all paths pruned. Keep at least root.
        # This shouldn't happen with oracle pruning, but be safe.
        return torch.zeros((1, 1), dtype=torch.long, device=device)

    # Find new max depth
    new_max_depth = max(len(r) for r in surviving_rows)

    # Pad rows to new_max_depth
    padded_rows = []
    for row in surviving_rows:
        pad_len = new_max_depth - len(row)
        if pad_len > 0:
            padding = torch.full((pad_len,), -1, dtype=torch.long, device=device)
            padded_row = torch.cat([row, padding])
        else:
            padded_row = row
        padded_rows.append(padded_row)

    return torch.stack(padded_rows)


# ---------------------------------------------------------------------------
# Oracle Pruner
# ---------------------------------------------------------------------------

class OraclePruner:
    """
    Oracle pruner that uses pre-collected is_subtree_wasted labels
    from nodes.parquet.

    At each cycle, it looks up the pre-computed prune mask by
    (question_id, turn_id, cycle_idx) and returns it.

    Requirements:
      - The generation must replay EXACTLY the same prompts at the same
        temperature (0.0) to ensure deterministic draft trees.
      - The cycle_idx must match between the live run and the parquet.
    """

    def __init__(self, nodes_parquet_path: str, run_id: str, bench_name: str):
        """
        Load nodes.parquet and build a lookup table.

        Parameters
        ----------
        nodes_parquet_path : str
            Path to the directory containing nodes.parquet.
        run_id : str
            The run_id used when building the parquet (matches cycle_uuid prefix).
        bench_name : str
            The bench_name to filter on (e.g., 'mt_bench').
        """
        import polars as pl
        from pathlib import Path

        parquet_path = Path(nodes_parquet_path) / "nodes.parquet"
        if not parquet_path.exists():
            # Maybe the path IS the parquet file
            parquet_path = Path(nodes_parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"nodes.parquet not found at {nodes_parquet_path}"
            )

        print(f"[OraclePruner] Loading {parquet_path}...")
        df = pl.read_parquet(str(parquet_path))

        # Filter to the relevant bench if there are multiple
        if bench_name and "cycle_uuid" in df.columns:
            # cycle_uuid format: "{run_id}_{bench_name}_{question_id}_{turn_id}_{cycle_idx}"
            df = df.filter(
                pl.col("cycle_uuid").str.contains(f"_{bench_name}_")
            )

        print(f"[OraclePruner] Loaded {len(df)} nodes for bench={bench_name}")

        # Build lookup: (question_id, turn_id, cycle_idx) → {node_idx: is_subtree_wasted}
        # Also store token_ids for determinism verification.
        self._lookup: Dict[Tuple[int, int, int], Dict[str, object]] = {}
        self._build_lookup(df, run_id, bench_name)
        self._stats = {"cycles_matched": 0, "cycles_missed": 0, "nodes_pruned": 0, "nodes_total": 0}

    def _build_lookup(self, df, run_id: str, bench_name: str):
        """Build the (question_id, turn_id, cycle_idx) -> data lookup, keyed by token_path."""
        import polars as pl

        has_token_path = "token_path" in df.columns
        if not has_token_path:
            print(
                "[OraclePruner] WARNING: nodes.parquet has no 'token_path' column. "
                "Re-run build_tables.py to regenerate. Falling back to node_idx matching "
                "(CUDA nondeterministic - may cause incorrect pruning)."
            )

        select_cols = ["cycle_uuid", "node_idx", "is_subtree_wasted", "token_id"]
        if has_token_path:
            select_cols.append("token_path")

        groups = df.select(select_cols).group_by("cycle_uuid").agg([
            pl.col("node_idx"),
            pl.col("is_subtree_wasted"),
            pl.col("token_id"),
            *([] if not has_token_path else [pl.col("token_path")]),
        ])

        for row in groups.iter_rows(named=True):
            cycle_uuid = row["cycle_uuid"]
            bench_marker = f"_{bench_name}_"
            pos = cycle_uuid.find(bench_marker)
            if pos < 0:
                continue
            suffix = cycle_uuid[pos + len(bench_marker):]
            parts = suffix.split("_")
            if len(parts) < 3:
                continue

            try:
                question_id = int(parts[0])
                turn_id = int(parts[1])
                cycle_idx = int(parts[2])
            except (ValueError, IndexError):
                continue

            node_indices = row["node_idx"]
            wasted_flags = row["is_subtree_wasted"]
            token_ids = row["token_id"]
            token_paths = row["token_path"] if has_token_path else None

            wasted_by_path: Dict[str, bool] = {}
            wasted_by_node: Dict[int, bool] = {}
            token_by_node: Dict[int, int] = {}

            for i, (nidx, wasted, tid) in enumerate(zip(node_indices, wasted_flags, token_ids)):
                wasted_by_node[nidx] = wasted
                token_by_node[nidx] = tid
                if token_paths is not None:
                    wasted_by_path[token_paths[i]] = wasted

            key = (question_id, turn_id, cycle_idx)
            self._lookup[key] = {
                "wasted_by_path": wasted_by_path,
                "wasted_by_node": wasted_by_node,
                "token_by_node":  token_by_node,
                "draft_tree_size": len(node_indices),
                "has_token_path": has_token_path,
            }

        print(f"[OraclePruner] Built lookup with {len(self._lookup)} cycles")

    def get_prune_mask(
        self,
        question_id: int,
        turn_id: int,
        cycle_idx: int,
        draft_tree_size: int,
        draft_tokens: torch.Tensor = None,
        tree_mask: torch.Tensor = None,
        device: torch.device = None,
    ) -> Optional[torch.Tensor]:
        """
        Return the oracle prune mask for the given cycle.

        If draft_tokens + tree_mask are provided, uses path-based matching
        (robust to topk reordering). Otherwise falls back to node_idx matching.

        Returns
        -------
        prune_mask : [draft_tree_size] boolean tensor, True = prune
        None if the cycle is not found in the lookup (miss).
        """
        key = (question_id, turn_id, cycle_idx)
        data = self._lookup.get(key)

        if data is None:
            self._stats["cycles_missed"] += 1
            return None

        self._stats["cycles_matched"] += 1

        # --- Path-based matching (preferred) ---
        if draft_tokens is not None and tree_mask is not None and data.get("has_token_path"):
            live_paths = build_token_paths_from_tensors(draft_tokens, tree_mask)
            wasted_by_path = data["wasted_by_path"]
            mask = torch.zeros(draft_tree_size, dtype=torch.bool, device=device)
            missing_paths = []
            for node_idx, path_str in live_paths.items():
                if node_idx >= draft_tree_size:
                    continue
                if path_str in wasted_by_path:
                    mask[node_idx] = wasted_by_path[path_str]
                else:
                    # Path not found in oracle - conservative: don't prune
                    mask[node_idx] = False
                    missing_paths.append(path_str)
            
            # Log missing paths once per cycle (avoid spam)
            if missing_paths:
                # Show which paths are in live tree but not in oracle
                sample_missing = missing_paths[:3]
                sample_oracle = list(wasted_by_path.keys())[:3]
                print(
                    f"[OraclePruner] INFO: {len(missing_paths)}/{len(live_paths)} live paths not in oracle at "
                    f"q={question_id} t={turn_id} c={cycle_idx} (non-determinism). "
                    f"Conservatively not pruning them.\n"
                    f"  Live tree samples (not in oracle): {sample_missing}\n"
                    f"  Oracle samples: {sample_oracle}"
                )
        else:
            # --- Fallback: node_idx matching (CUDA nondeterministic risk) ---
            wasted_by_node = data.get("wasted_by_node") or data.get("wasted_map", {})
            mask = torch.zeros(draft_tree_size, dtype=torch.bool, device=device)
            for node_idx in range(draft_tree_size):
                mask[node_idx] = wasted_by_node.get(node_idx, False)

        # Never prune root
        mask[0] = False

        pruned_count = mask.sum().item()
        self._stats["nodes_pruned"] += pruned_count
        self._stats["nodes_total"] += draft_tree_size

        return mask

    def verify_determinism(
        self,
        draft_tokens: torch.Tensor,
        tree_mask: torch.Tensor,
        question_id: int,
        turn_id: int,
        cycle_idx: int,
    ) -> bool:
        """
        Verify path determinism: build live token paths and check they all exist
        in the oracle's wasted_by_path map.

        This is the path-aware replacement for the old node_idx-based check.
        A mismatch means the live tree has paths not seen in the oracle data,
        so oracle labels won't cover some live nodes.

        Returns True if all live paths are found in oracle data, False otherwise.
        """
        key = (question_id, turn_id, cycle_idx)
        data = self._lookup.get(key)
        if data is None:
            return True  # Can't verify if cycle not found

        if not data.get("has_token_path"):
            # Old parquet without token_path — skip path verification
            return True

        live_paths = build_token_paths_from_tensors(draft_tokens, tree_mask)
        wasted_by_path = data["wasted_by_path"]

        live_path_set = set(live_paths.values())
        oracle_path_set = set(wasted_by_path.keys())
        
        missing_in_oracle = live_path_set - oracle_path_set  # in live, not in oracle
        extra_in_oracle = oracle_path_set - live_path_set   # in oracle, not in live
        
        if missing_in_oracle:
            # Only print detailed mismatch if it's severe (>10% of nodes)
            if len(missing_in_oracle) > len(live_paths) * 0.1:
                sample_missing = list(missing_in_oracle)[:3]
                sample_extra = list(extra_in_oracle)[:3] if extra_in_oracle else []
                print(
                    f"[OraclePruner] PATH MISMATCH at q={question_id} t={turn_id} "
                    f"c={cycle_idx}: {len(missing_in_oracle)}/{len(live_paths)} live paths not in oracle.\n"
                    f"  Live tree paths NOT in oracle (sample): {sample_missing}\n"
                    f"  Oracle paths NOT in live tree (sample): {sample_extra if sample_extra else 'none'}"
                )
            return False
        return True

    def get_stats(self) -> Dict:
        """Return cumulative pruning statistics."""
        stats = dict(self._stats)
        if stats["nodes_total"] > 0:
            stats["prune_rate"] = stats["nodes_pruned"] / stats["nodes_total"]
        else:
            stats["prune_rate"] = 0.0
        # Include dry-run stats if present
        if "nodes_pruned_dry" in stats:
            stats["prune_rate_dry"] = stats["nodes_pruned_dry"] / max(stats["nodes_total"], 1)
        return stats


# ---------------------------------------------------------------------------
# Top-level prune() function — called from _maybe_prune in ea_model.py
# ---------------------------------------------------------------------------

def prune_draft_tree(
    pruner: OraclePruner,
    draft_tokens: torch.Tensor,
    retrieve_indices: torch.Tensor,
    tree_mask: torch.Tensor,
    tree_position_ids: torch.Tensor,
    question_id: int,
    turn_id: int,
    cycle_idx: int,
    verify: bool = True,
    dry_run: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Main entry point: get prune mask from the pruner and apply tensor surgery.

    Returns the original tensors unchanged if pruner returns None (cache miss)
    or if nothing should be pruned.

    If dry_run=True, computes and logs what WOULD be pruned but returns the
    original tensors unchanged. This avoids the cascade divergence caused by
    CUDA non-determinism when the tree_decoding forward pass sees a
    different-sized tensor after pruning.
    """
    device = draft_tokens.device
    N = draft_tokens.shape[1]

    # Path-aware determinism check
    if verify:
        pruner.verify_determinism(draft_tokens, tree_mask, question_id, turn_id, cycle_idx)

    # Get prune mask (path-based when possible)
    prune_mask = pruner.get_prune_mask(
        question_id, turn_id, cycle_idx, N,
        draft_tokens=draft_tokens,
        tree_mask=tree_mask,
        device=device,
    )

    if prune_mask is None:
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids

    if not prune_mask.any():
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids

    if dry_run:
        # Record what WOULD be pruned without actually modifying the tree.
        # This preserves the generation trajectory (identical to unpruned run)
        # while still computing the theoretical savings.
        n_pruned = int(prune_mask.sum().item())
        pruner._stats["nodes_pruned_dry"] = pruner._stats.get("nodes_pruned_dry", 0) + n_pruned
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids

    return rebuild_tensors(
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, prune_mask
    )
