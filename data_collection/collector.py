"""Data collection utilities for EAGLE speculative decoding.

Supports two modes:
- phase0: minimal features/labels only (prompts/cycles/nodes tables)
- full: phase0 data plus raw tensors from topK_genrate/evaluate_posterior per cycle
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Literal

import pandas as pd
import torch

CollectorMode = Literal["phase0", "full"]


class DataCollector:
    def __init__(self, output_dir: str, mode: CollectorMode = "phase0") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mode: CollectorMode = mode

        # Buffers
        self.prompts: List[Dict] = []
        self.cycles: List[Dict] = []
        self.nodes: List[Dict] = []

        # Counters
        self.cycle_counter = 0
        self.node_counter = 0

        # Extra payload directory for full mode
        self.full_payload_dir = self.output_dir / "full_cycles" if self.mode == "full" else None
        if self.full_payload_dir is not None:
            self.full_payload_dir.mkdir(parents=True, exist_ok=True)

    # Prompt-level
    def add_prompt(
        self,
        prompt_id: int,
        prompt_text: str,
        prompt_len: int,
        source: str = "mt-bench",
        turn_idx: Optional[int] = None,
    ) -> None:
        self.prompts.append(
            {
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
                "prompt_len": prompt_len,
                "source": source,
                "turn_idx": turn_idx,
            }
        )

    # Cycle-level
    def start_cycle(self, prompt_id: int, cycle_idx: int, turn_idx: Optional[int] = None) -> int:
        cycle_id = self.cycle_counter
        self.cycles.append(
            {
                "cycle_id": cycle_id,
                "prompt_id": prompt_id,
                "cycle_idx": cycle_idx,
                "turn_idx": turn_idx,
                # Labels set in finalize_cycle
                "accepted_length": -1,
                "best_path_idx": -1,
            }
        )
        self.cycle_counter += 1
        return cycle_id

    def finalize_cycle(self, cycle_id: int, accepted_length: int, best_path_idx: int) -> None:
        for cycle in self.cycles:
            if cycle["cycle_id"] == cycle_id:
                cycle["accepted_length"] = accepted_length
                cycle["best_path_idx"] = best_path_idx
                break

    # Node-level
    def add_nodes(self, cycle_id: int, node_features: List[Dict], node_labels: List[bool]) -> None:
        for node_idx, (feat, label) in enumerate(zip(node_features, node_labels)):
            self.nodes.append(
                {
                    "node_id": self.node_counter,
                    "cycle_id": cycle_id,
                    "node_idx": node_idx,
                    # Features
                    "draft_token_id": feat.get("draft_token_id"),
                    "draft_prob": feat.get("draft_prob"),
                    "tree_depth": feat.get("tree_depth"),
                    "sibling_rank": feat.get("sibling_rank"),
                    # Label
                    "is_accepted": bool(label),
                }
            )
            self.node_counter += 1

    # Full-mode payload
    def log_full_cycle(
        self,
        cycle_id: int,
        draft_tokens: torch.Tensor,
        retrieve_indices: torch.Tensor,
        tree_mask: torch.Tensor,
        tree_position_ids: torch.Tensor,
        sample_p: Optional[torch.Tensor],
        debug_info: Optional[Dict] = None,
    ) -> None:
        if self.mode != "full" or self.full_payload_dir is None:
            return

        payload = {
            "draft_tokens": draft_tokens.detach().cpu(),
            "retrieve_indices": retrieve_indices.detach().cpu(),
            "tree_mask": tree_mask.detach().cpu(),
            "tree_position_ids": tree_position_ids.detach().cpu(),
            "sample_p": sample_p.detach().cpu() if sample_p is not None else None,
            "debug_info": None,
        }
        if debug_info is not None:
            payload["debug_info"] = {
                k: [t.detach().cpu() for t in v] if isinstance(v, list) else (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in debug_info.items()
            }

        torch.save(payload, self.full_payload_dir / f"cycle_{cycle_id:06d}.pt")

    # Flush to disk
    def save(self, metadata: Optional[Dict] = None) -> None:
        print(f"[DataCollector] Saving dataset to {self.output_dir}")

        df_prompts = pd.DataFrame(self.prompts)
        df_cycles = pd.DataFrame(self.cycles)
        df_nodes = pd.DataFrame(self.nodes)

        df_prompts.to_parquet(self.output_dir / "prompts.parquet", index=False)
        df_cycles.to_parquet(self.output_dir / "cycles.parquet", index=False)
        df_nodes.to_parquet(self.output_dir / "nodes.parquet", index=False)

        meta = metadata.copy() if metadata is not None else {}
        meta.update(
            {
                "mode": self.mode,
                "num_prompts": len(self.prompts),
                "num_cycles": len(self.cycles),
                "num_nodes": len(self.nodes),
            }
        )

        with open(self.output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(
            f"[DataCollector] Saved {len(self.prompts)} prompts, {len(self.cycles)} cycles, {len(self.nodes)} nodes"
        )
        if not df_nodes.empty:
            print(f"[DataCollector] Node acceptance rate: {df_nodes['is_accepted'].mean():.3f}")
        if not df_cycles.empty:
            print(f"[DataCollector] Mean cycle tau (accepted_length): {df_cycles['accepted_length'].mean():.2f}")
