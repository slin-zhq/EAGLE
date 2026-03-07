#!/usr/bin/env python3
"""
Unit tests for the pruner tensor surgery.

Run: python -m pytest test_pruner.py -v
Or:  python test_pruner.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from pruner import rebuild_tensors, _enforce_subtree_consistency


def _make_simple_tree():
    """
    Build a minimal 7-node draft tree for testing:

         0 (root, depth=0)
        / \\
       1   2     (depth=1)
      /|    \\
     3  4    5   (depth=2)
     |
     6           (depth=3)

    draft_tokens: [1, 7] (batch=1, 7 nodes)
    tree_position_ids: [0, 1, 1, 2, 2, 2, 3]
    tree_mask: 7×7 boolean ancestry matrix
    retrieve_indices: paths from root to each leaf
    """
    N = 7
    draft_tokens = torch.tensor([[100, 200, 300, 400, 500, 600, 700]])  # [1, 7]

    # tree_position_ids: depth of each node
    tree_position_ids = torch.tensor([0, 1, 1, 2, 2, 2, 3])

    # Build tree_mask: tree_mask[i, j] = 1 iff j is ancestor of i (or i==j)
    # Ancestry:
    #   0 → root (ancestor of all)
    #   1 → ancestors: {0, 1}
    #   2 → ancestors: {0, 2}
    #   3 → ancestors: {0, 1, 3}
    #   4 → ancestors: {0, 1, 4}
    #   5 → ancestors: {0, 2, 5}
    #   6 → ancestors: {0, 1, 3, 6}
    tm = torch.zeros(N, N)
    # Each node attends to itself
    for i in range(N):
        tm[i, i] = 1.0
    # All attend to root
    tm[:, 0] = 1.0
    # Children attend to parent
    tm[1, 0] = 1.0  # 1's parent is 0
    tm[2, 0] = 1.0  # 2's parent is 0
    tm[3, 1] = 1.0  # 3's parent is 1
    tm[4, 1] = 1.0  # 4's parent is 1
    tm[5, 2] = 1.0  # 5's parent is 2
    tm[6, 3] = 1.0  # 6's parent is 3
    # Transitive: 3 attends to 0 (through 1) — already via tm[:,0]=1
    # 6 attends to 1 (grandparent) and 0
    tm[6, 1] = 1.0
    tree_mask = tm[None, None]  # [1, 1, 7, 7]

    # retrieve_indices: paths from root to each LEAF
    # Leaves are: 4, 5, 6 (nodes with no children)
    # Path to 4: [0, 1, 4, -1]
    # Path to 5: [0, 2, 5, -1]
    # Path to 6: [0, 1, 3, 6]
    retrieve_indices = torch.tensor([
        [0, 1, 4, -1],
        [0, 2, 5, -1],
        [0, 1, 3,  6],
    ])

    return draft_tokens, retrieve_indices, tree_mask, tree_position_ids


def test_no_prune():
    """When nothing is pruned, output equals input."""
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = _make_simple_tree()
    prune_mask = torch.zeros(7, dtype=torch.bool)

    new_dt, new_ri, new_tm, new_tp = rebuild_tensors(
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, prune_mask
    )

    assert torch.equal(new_dt, draft_tokens), "draft_tokens should be unchanged"
    assert torch.equal(new_ri, retrieve_indices), "retrieve_indices should be unchanged"
    assert torch.equal(new_tm, tree_mask), "tree_mask should be unchanged"
    assert torch.equal(new_tp, tree_position_ids), "tree_position_ids should be unchanged"
    print("✅ test_no_prune passed")


def test_prune_single_leaf():
    """Pruning leaf node 4 should remove it and drop its retrieve_indices row."""
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = _make_simple_tree()

    # Prune node 4 (leaf)
    prune_mask = torch.zeros(7, dtype=torch.bool)
    prune_mask[4] = True

    new_dt, new_ri, new_tm, new_tp = rebuild_tensors(
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, prune_mask
    )

    # After pruning node 4:
    # Surviving nodes: [0, 1, 2, 3, 5, 6] (M=6)
    # Remap: 0→0, 1→1, 2→2, 3→3, 5→4, 6→5
    assert new_dt.shape == (1, 6), f"Expected [1,6] but got {new_dt.shape}"
    assert torch.equal(new_dt, torch.tensor([[100, 200, 300, 400, 600, 700]]))
    assert new_tp.shape == (6,)
    assert torch.equal(new_tp, torch.tensor([0, 1, 1, 2, 2, 3]))

    # tree_mask should be 6×6
    assert new_tm.shape == (1, 1, 6, 6)

    # retrieve_indices: path to node 4 dropped
    # Remaining paths:
    #   [0, 2, 5] → remapped [0, 2, 4]
    #   [0, 1, 3, 6] → remapped [0, 1, 3, 5]
    assert new_ri.shape[0] == 2, f"Expected 2 rows but got {new_ri.shape[0]}"
    print(f"  retrieve_indices after prune: {new_ri}")
    print("✅ test_prune_single_leaf passed")


def test_prune_subtree():
    """Pruning internal node 1 should also prune all descendants (3, 4, 6)."""
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = _make_simple_tree()

    # Prune node 1 (internal) — should cascade to descendants 3, 4, 6
    prune_mask = torch.zeros(7, dtype=torch.bool)
    prune_mask[1] = True

    new_dt, new_ri, new_tm, new_tp = rebuild_tensors(
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, prune_mask
    )

    # After pruning nodes 1, 3, 4, 6:
    # Surviving: [0, 2, 5] (M=3)
    # Remap: 0→0, 2→1, 5→2
    assert new_dt.shape == (1, 3), f"Expected [1,3] but got {new_dt.shape}"
    assert torch.equal(new_dt, torch.tensor([[100, 300, 600]]))
    assert new_tp.shape == (3,)
    assert torch.equal(new_tp, torch.tensor([0, 1, 2]))

    # tree_mask should be 3×3
    assert new_tm.shape == (1, 1, 3, 3)

    # Only the path [0, 2, 5] → [0, 1, 2] survives
    assert new_ri.shape[0] == 1, f"Expected 1 row but got {new_ri.shape[0]}"
    # Check the surviving path has correct remapped values
    surviving_path = new_ri[0]
    valid = surviving_path[surviving_path >= 0]
    assert torch.equal(valid, torch.tensor([0, 1, 2])), f"Expected [0,1,2] but got {valid}"
    print("✅ test_prune_subtree passed")


def test_subtree_consistency():
    """If parent pruned, child MUST also be pruned even if child mask is False."""
    _, _, tree_mask, _ = _make_simple_tree()

    # Only mark node 1 for pruning (not its children 3, 4, 6)
    prune_mask = torch.zeros(7, dtype=torch.bool)
    prune_mask[1] = True

    enforced = _enforce_subtree_consistency(prune_mask, tree_mask)

    # Descendants of 1 are: 3, 4, 6 — all should be True
    assert enforced[1] == True, "Node 1 should still be pruned"
    assert enforced[3] == True, "Node 3 (child of 1) should be pruned"
    assert enforced[4] == True, "Node 4 (child of 1) should be pruned"
    assert enforced[6] == True, "Node 6 (grandchild of 1) should be pruned"

    # Non-descendants should be unaffected
    assert enforced[0] == False, "Root should never be pruned"
    assert enforced[2] == False, "Node 2 should not be pruned"
    assert enforced[5] == False, "Node 5 should not be pruned"
    print("✅ test_subtree_consistency passed")


def test_root_never_pruned():
    """Even if prune_mask[0] = True, root should survive."""
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = _make_simple_tree()

    prune_mask = torch.zeros(7, dtype=torch.bool)
    prune_mask[0] = True  # Try to prune root

    new_dt, new_ri, new_tm, new_tp = rebuild_tensors(
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, prune_mask
    )

    # Root should survive. And since subtree consistency would cascade from root
    # to all nodes — but root is NEVER pruned, so nothing cascades.
    assert new_dt.shape == (1, 7), "Nothing should be pruned since root can't be pruned"
    print("✅ test_root_never_pruned passed")


def test_prune_all_wasted():
    """Prune all nodes except root and one path — simulates oracle scenario."""
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = _make_simple_tree()

    # Keep only path [0, 2, 5] — prune 1, 3, 4, 6
    prune_mask = torch.zeros(7, dtype=torch.bool)
    prune_mask[1] = True  # subtree consistency will cascade to 3, 4, 6

    new_dt, new_ri, new_tm, new_tp = rebuild_tensors(
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, prune_mask
    )

    # Surviving: [0, 2, 5] → tokens [100, 300, 600]
    assert new_dt.shape == (1, 3)
    assert new_ri.shape[0] == 1  # only the [0, 2, 5] path
    print("✅ test_prune_all_wasted passed")


if __name__ == "__main__":
    test_no_prune()
    test_prune_single_leaf()
    test_prune_subtree()
    test_subtree_consistency()
    test_root_never_pruned()
    test_prune_all_wasted()
    print("\n🎉 All tests passed!")
