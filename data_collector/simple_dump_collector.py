"""
Simple Dump Data Collector

Collects raw tensors from EAGLE-3 draft-verify cycles for manual examination.
Saves per-cycle .pt files with minimal processing, now including the raw
`scores_list`, `ss_token_list`, and `top_scores_index` emitted by
`topK_genrate()` so Phase 0 features can be reconstructed offline.

Also records per-cycle stage-level timing (verify, evaluate, update) and
one-time prefill timing.  Timing is measured at the caller (ea_model.py)
using torch.cuda.synchronize() + time.time(), then passed in here.
"""
import os
import json
import csv
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class SimpleDumpCollector:
    """
    Collector for simple tensor dumps during EAGLE-3 inference.
    
    Saves one .pt file per draft-verify cycle containing:
    - Draft generation outputs (draft_tokens, retrieve_indices, tree_mask, tree_position_ids)
    - Verification inputs (logits, candidates)
    - Verification outputs (best_candidate, accept_length, sample_p)
    - Derived metrics and metadata
    """
    
    def __init__(self, output_dir: str, question_id: int, turn_id: int = 0):
        """
        Initialize collector for a specific question and turn.
        
        Args:
            output_dir: Base output directory (e.g., data_output_simple_dump_20260122_143022/mt_bench)
            question_id: Question identifier
            turn_id: Turn index for multi-turn conversations (default: 0 for single-turn benchmarks)
        """
        self.output_dir = Path(output_dir)
        self.question_id = question_id
        self.turn_id = turn_id
        self.question_dir = self.output_dir / f"question_{question_id}_turn_{turn_id}"
        self.question_dir.mkdir(parents=True, exist_ok=True)
        
        self.cycle_count = 0
        self.cycle_data_buffer = {}  # Stores data for current cycle
        self.summary_rows = []  # For CSV summary
        self.prefill_time_s = None  # One-time prefill timing (set via set_prefill_time)
        
        print(f"[SimpleDumpCollector] Initialized for question {question_id}, turn {turn_id}")
        print(f"[SimpleDumpCollector] Output: {self.question_dir}")
    
    def start_cycle(self):
        """Mark the beginning of a new draft-verify cycle."""
        self.cycle_data_buffer = {
            'cycle_id': self.cycle_count,
            'question_id': self.question_id,
            'turn_id': self.turn_id,
        }
    
    def collect_draft_outputs(
        self,
        draft_tokens: torch.Tensor,
        retrieve_indices: torch.Tensor,
        tree_mask: torch.Tensor,
        tree_position_ids: torch.Tensor,
        scores_list: Optional[torch.Tensor] = None,
        ss_token_list: Optional[torch.Tensor] = None,
        top_scores_index: Optional[torch.Tensor] = None,
    ):
        """
        Collect outputs from topK_genrate() (draft generation).
        
        Args:
            draft_tokens: Token IDs in draft tree, shape (draft_tree_size,)
            retrieve_indices: Parent-child indices, shape (draft_tree_size,)
            tree_mask: Attention mask, shape (draft_tree_size, draft_tree_size) [NOT SAVED - unused by build_tables.py]
            tree_position_ids: Position IDs, shape (draft_tree_size,)
        """
        self.cycle_data_buffer['draft_tokens'] = draft_tokens.cpu()
        self.cycle_data_buffer['retrieve_indices'] = retrieve_indices.cpu()
        # tree_mask: (draft_tree_size, draft_tree_size) — not used by build_tables.py or any analysis script.
        # Commented out to save disk space. Re-enable if attention mask analysis is needed.
        # self.cycle_data_buffer['tree_mask'] = tree_mask.cpu()
        self.cycle_data_buffer['tree_position_ids'] = tree_position_ids.cpu()
        if scores_list is not None:
            self.cycle_data_buffer['scores_list'] = scores_list.cpu()
        # ss_token_list: not used by build_tables.py or any downstream analysis script.
        # Commented out to save disk space. Re-enable if Phase 0 feature reconstruction is needed.
        # if ss_token_list is not None:
        #     self.cycle_data_buffer['ss_token_list'] = ss_token_list.cpu()
        if top_scores_index is not None:
            self.cycle_data_buffer['top_scores_index'] = top_scores_index.cpu()
    
    def collect_verification_inputs(
        self,
        logits: torch.Tensor,
        candidates: torch.Tensor,
    ):
        """
        Collect inputs to evaluate_posterior() (verification).

        Accepts either:
        - Full logits for all candidates: shape (num_candidates, seq_len, vocab_size)
        - Or per-position logits for a single candidate: shape (seq_len, vocab_size)

        Args:
            logits: Target model logits (see shapes above) [NOT SAVED - unused by build_tables.py]
            candidates: Candidate sequences, shape (num_candidates, max_length)
        """
        # logits: (num_candidates, seq_len, vocab_size) — by far the largest tensor (e.g. 60 * 6 * 32000 float16).
        # Not used by build_tables.py or any analysis script; not needed for ML predictor features.
        # Commented out to save disk space. Re-enable if logit-level posterior analysis is needed.
        # self.cycle_data_buffer['logits'] = logits.cpu()
        self.cycle_data_buffer['candidates'] = candidates.cpu()
    
    def collect_verification_outputs(
        self,
        best_candidate: torch.Tensor,
        accept_length: int,
        sample_p: torch.Tensor,
    ):
        """
        Collect outputs from evaluate_posterior() (verification).
        
        Args:
            best_candidate: Accepted token sequence, shape (accept_length,)
            accept_length: Number of tokens accepted
            sample_p: Posterior probabilities, shape (num_candidates,)
        """
        # Ensure accept_length is a plain int for JSON serialization later
        if torch.is_tensor(accept_length):
            accept_length = int(accept_length.item())
        self.cycle_data_buffer['best_candidate'] = best_candidate.cpu()
        self.cycle_data_buffer['accept_length'] = accept_length
        self.cycle_data_buffer['sample_p'] = sample_p.cpu()
    
    def collect_timing(self, timing_dict: Dict[str, float]):
        """
        Store per-cycle stage timing measured by the caller.
        
        Args:
            timing_dict: Dict with keys like 'verify_s', 'evaluate_s', 'update_s',
                         each mapping to wall-clock seconds for that stage.
        """
        self.cycle_data_buffer['timing'] = timing_dict
    
    def set_prefill_time(self, prefill_time_s: float):
        """
        Record the one-time prefill duration (initialize_tree).
        Called once before the main generation loop starts.
        
        Args:
            prefill_time_s: Wall-clock seconds for the prefill stage.
        """
        self.prefill_time_s = prefill_time_s
    
    def end_cycle(self):
        """
        Finalize current cycle: compute metrics, save .pt file, update summary.
        """
        # Compute derived metrics
        derived_metrics = self._compute_derived_metrics()
        
        # Create metadata
        metadata = {
            'cycle_id': self.cycle_data_buffer['cycle_id'],
            'question_id': self.cycle_data_buffer['question_id'],
            'turn_id': self.cycle_data_buffer['turn_id'],
            'timestamp': datetime.now().isoformat(),
            'shapes': {},
            'dtypes': {},
            'derived_metrics': derived_metrics,
        }
        
        # Populate shapes and dtypes
        tensor_keys = [
            'draft_tokens', 'retrieve_indices', 'tree_position_ids',
            # 'tree_mask',       # Commented out: not collected (saves space)
            # 'logits',          # Commented out: not collected (saves space)
            # 'ss_token_list',   # Commented out: not collected (saves space)
            'candidates', 'best_candidate', 'sample_p',
            'scores_list', 'top_scores_index',
        ]
        for key in tensor_keys:
            if key in self.cycle_data_buffer:
                tensor = self.cycle_data_buffer[key]
                metadata['shapes'][key] = tuple(tensor.shape)
                metadata['dtypes'][key] = str(tensor.dtype)
        
        # Prepare data package
        data_package = {
            'cycle_id': self.cycle_data_buffer['cycle_id'],
            'question_id': self.cycle_data_buffer['question_id'],
            'turn_id': self.cycle_data_buffer['turn_id'],
            'draft_tokens': self.cycle_data_buffer.get('draft_tokens'),
            'retrieve_indices': self.cycle_data_buffer.get('retrieve_indices'),
            # 'tree_mask': self.cycle_data_buffer.get('tree_mask'),  # Not collected: unused, saves space
            'tree_position_ids': self.cycle_data_buffer.get('tree_position_ids'),
            # 'logits': self.cycle_data_buffer.get('logits'),        # Not collected: unused, saves space
            'candidates': self.cycle_data_buffer.get('candidates'),
            'best_candidate': self.cycle_data_buffer.get('best_candidate'),
            'accept_length': self.cycle_data_buffer.get('accept_length'),
            'sample_p': self.cycle_data_buffer.get('sample_p'),
            'scores_list': self.cycle_data_buffer.get('scores_list'),
            # 'ss_token_list': self.cycle_data_buffer.get('ss_token_list'),  # Not collected: unused, saves space
            'top_scores_index': self.cycle_data_buffer.get('top_scores_index'),
            'timing': self.cycle_data_buffer.get('timing'),
            'metadata': metadata,
        }
        
        # Save cycle .pt file
        cycle_file = self.question_dir / f"cycle_{self.cycle_count:03d}.pt"
        torch.save(data_package, cycle_file)
        
        # Update summary
        accept_length = self.cycle_data_buffer.get('accept_length')
        if torch.is_tensor(accept_length):
            accept_length = int(accept_length.item())
        # Include per-cycle timing in summary if available
        timing = self.cycle_data_buffer.get('timing', {})
        summary_row = {
            'question_id': self.question_id,
            'turn_id': self.turn_id,
            'cycle_id': self.cycle_count,
            'accept_length': accept_length,
            **derived_metrics,
            **{f'timing_{k}': v for k, v in timing.items()},
        }
        self.summary_rows.append(summary_row)
        
        print(f"[SimpleDumpCollector] Saved cycle {self.cycle_count} -> {cycle_file.name}")
        
        # Increment cycle counter
        self.cycle_count += 1
        self.cycle_data_buffer = {}
    
    def _compute_derived_metrics(self) -> Dict[str, Any]:
        """
        Compute derived metrics from collected tensors.
        
        Returns:
            Dictionary of metric name -> value
        """
        metrics = {}
        
        # Draft tree statistics
        if 'draft_tokens' in self.cycle_data_buffer:
            draft_tokens = self.cycle_data_buffer['draft_tokens']
            metrics['total_draft_nodes'] = len(draft_tokens)
        
        if 'tree_position_ids' in self.cycle_data_buffer:
            tree_position_ids = self.cycle_data_buffer['tree_position_ids']
            metrics['tree_depth'] = int(tree_position_ids.max().item()) + 1
        
        # Acceptance metrics
        if 'accept_length' in self.cycle_data_buffer:
            accept_length = self.cycle_data_buffer['accept_length']
            if torch.is_tensor(accept_length):
                accept_length = int(accept_length.item())
            
            if 'total_draft_nodes' in metrics:
                metrics['acceptance_rate'] = accept_length / metrics['total_draft_nodes']
        
        # Candidate selection metrics
        if 'sample_p' in self.cycle_data_buffer:
            sample_p = self.cycle_data_buffer['sample_p']
            
            # Max posterior probability
            metrics['max_posterior_prob'] = sample_p.max().item()
        
        # Verification efficiency
        if 'candidates' in self.cycle_data_buffer and 'accept_length' in self.cycle_data_buffer:
            num_candidates = len(self.cycle_data_buffer['candidates'])
            accept_length = self.cycle_data_buffer['accept_length']
            metrics['num_candidates'] = num_candidates
            if num_candidates > 0:
                metrics['verification_yield'] = accept_length / num_candidates
        
        return metrics
    
    def finalize_question(self):
        """
        Finalize data collection for this question.
        Saves metadata.json with summary statistics.
        """
        # Aggregate statistics
        if self.summary_rows:
            total_cycles = len(self.summary_rows)
            total_accepted = sum(row['accept_length'] for row in self.summary_rows)
            total_drafted = sum(row['total_draft_nodes'] for row in self.summary_rows)
            avg_acceptance_rate = total_accepted / total_drafted if total_drafted > 0 else 0
            avg_tau = total_accepted / total_cycles
        else:
            total_cycles = 0
            total_accepted = 0
            total_drafted = 0
            avg_acceptance_rate = 0
            avg_tau = 0
        
        # Aggregate timing if available
        timing_keys = [k for k in self.summary_rows[0] if k.startswith('timing_')] if self.summary_rows else []
        timing_totals = {k: sum(row.get(k, 0) for row in self.summary_rows) for k in timing_keys}
        timing_means = {k.replace('timing_', 'avg_'): v / total_cycles for k, v in timing_totals.items()} if total_cycles > 0 else {}
        
        metadata = {
            'question_id': self.question_id,
            'turn_id': self.turn_id,
            'total_cycles': total_cycles,
            'total_tokens_accepted': total_accepted,
            'total_tokens_drafted': total_drafted,
            'average_acceptance_rate': avg_acceptance_rate,
            'average_tau_per_cycle': avg_tau,
            'prefill_time_s': self.prefill_time_s,
            **timing_totals,
            **timing_means,
            'summary': self.summary_rows,
        }
        
        # Save metadata.json
        metadata_file = self.question_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[SimpleDumpCollector] Finalized question {self.question_id}, turn {self.turn_id}")
        print(f"[SimpleDumpCollector]   Total cycles: {total_cycles}")
        print(f"[SimpleDumpCollector]   Avg τ (tau): {avg_tau:.2f} tokens/cycle")
        print(f"[SimpleDumpCollector]   Avg α (alpha): {avg_acceptance_rate:.3f}")
    
    @staticmethod
    def create_run_summary(output_dir: str):
        """
        Create run_summary.csv aggregating all questions in the output directory.
        
        Args:
            output_dir: Base output directory containing question_* subdirs
        """
        output_path = Path(output_dir)
        summary_rows = []
        
        # Collect all metadata.json files
        # Pattern now matches both old (question_X) and new (question_X_turn_Y) formats
        for question_dir in sorted(output_path.glob("question_*")):
            metadata_file = question_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    summary_rows.extend(metadata.get('summary', []))
        
        if not summary_rows:
            print(f"[SimpleDumpCollector] No summary data found in {output_dir}")
            return
        
        # Write CSV
        csv_file = output_path / "run_summary.csv"
        fieldnames = list(summary_rows[0].keys())
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        
        print(f"[SimpleDumpCollector] Created {csv_file}")
        print(f"[SimpleDumpCollector]   Total cycles: {len(summary_rows)}")
