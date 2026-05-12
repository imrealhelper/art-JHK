"""Entry point for physics-informed speculative draft ART experiments."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import numpy as np
import torch

from transformer.draft_art import DraftARTConfig
from transformer.draft_eval import (
    build_full_art_model,
    evaluate_draft_model,
    evaluate_full_art_baseline,
    load_draft_checkpoint,
    load_repository_tensors,
    normalize_split,
)
from transformer.draft_losses import DraftLossConfig
from transformer.draft_train import DraftChunkDataset, train_draft_model
from transformer.speculative import speculative_inference
from utils.metrics import save_metrics
from utils.plotting import (
    plot_example_trajectory,
    plot_metric_bar,
    plot_prefix_histogram,
    plot_runtime_comparison,
)
from utils.reproducibility import collect_run_metadata, set_global_seed


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training and evaluation.

    Returns:
        Namespace containing runtime, training, loss, checkpoint, and plotting
        options. All time variables use ``terminal_time_s``, ``num_intervals``,
        and ``time_step_s`` terminology.
    """
    parser = argparse.ArgumentParser(description="Physics-informed speculative draft ART")
    parser.add_argument("--mode", choices=["train_draft", "eval", "train_and_eval"], default="train_and_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--draft-chunk-length", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--use-distillation", action="store_true")
    parser.add_argument("--use-physics-loss", action="store_true")
    parser.add_argument("--use-acceptance-loss", action="store_true")
    parser.add_argument("--use-speculative-inference", action="store_true")
    parser.add_argument("--full-art-checkpoint", type=str, default=None)
    parser.add_argument("--draft-checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/pisd_art")
    parser.add_argument("--num-eval-samples", type=int, default=5)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--dataset-dir", type=str, default="dataset")
    parser.add_argument("--loss-variant", choices=["imitation", "imitation_distillation", "imitation_physics", "imitation_distillation_physics", "full"], default="full")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--terminal-time-s", type=float, default=6000.0)
    parser.add_argument("--u-max-m-s", type=float, default=10.0)
    parser.add_argument("--accept-threshold", type=float, default=1.0)
    return parser.parse_args()


def apply_loss_variant(args: argparse.Namespace) -> argparse.Namespace:
    """Apply ablation switch semantics to individual loss flags."""
    if args.loss_variant == "imitation":
        args.use_distillation = False
        args.use_physics_loss = False
        args.use_acceptance_loss = False
    elif args.loss_variant == "imitation_distillation":
        args.use_distillation = True
        args.use_physics_loss = False
        args.use_acceptance_loss = False
    elif args.loss_variant == "imitation_physics":
        args.use_distillation = False
        args.use_physics_loss = True
        args.use_acceptance_loss = False
    elif args.loss_variant == "imitation_distillation_physics":
        args.use_distillation = True
        args.use_physics_loss = True
        args.use_acceptance_loss = False
    elif args.loss_variant == "full":
        args.use_distillation = True
        args.use_physics_loss = True
        args.use_acceptance_loss = True
    return args


def print_tables(metrics: Dict[str, Any]) -> None:
    """Print runtime and performance summary tables to stdout."""
    print("\n================ Runtime Summary ================")
    print(f"{'Method':30s} {'Mean [s]':>12s} {'Std [s]':>12s}")
    runtime_rows = [
        ("Full ART inference", metrics.get("full_art", {}).get("full_art_inference_time_mean_s"), metrics.get("full_art", {}).get("full_art_inference_time_std_s")),
        ("Draft inference", metrics.get("draft", {}).get("draft_inference_time_mean_s"), metrics.get("draft", {}).get("draft_inference_time_std_s")),
        ("Speculative inference", metrics.get("speculative", {}).get("speculative_inference_time_mean_s"), metrics.get("speculative", {}).get("speculative_inference_time_std_s")),
        ("Full ART + SCP", metrics.get("scp", {}).get("art_scp_runtime_mean_s")),
        ("Speculative + SCP", metrics.get("scp", {}).get("speculative_scp_runtime_mean_s")),
    ]
    for method, mean, std in runtime_rows:
        mean_s = "nan" if mean is None else f"{float(mean):.6f}"
        std_s = "nan" if std is None else f"{float(std):.6f}"
        print(f"{method:30s} {mean_s:>12s} {std_s:>12s}")
    print("\n================ Performance Summary =============")
    print(f"{'Method':30s} {'Fuel [m/s]':>12s} {'Pos Err [m]':>14s} {'Vel Err [m/s]':>16s} {'KOZ Max Viol':>14s}")
    for key, label in [("full_art", "Full ART"), ("draft", "Draft"), ("physics_draft", "Physics Draft"), ("speculative", "Speculative + Verifier"), ("scp", "ART + SCP")]:
        row = metrics.get(key, {})
        print(f"{label:30s} {float(row.get('fuel_cost_m_s', np.nan)):12.6f} {float(row.get('terminal_position_error_m', np.nan)):14.6f} {float(row.get('terminal_velocity_error_m_s', np.nan)):16.6f} {float(row.get('koz_max_violation', np.nan)):14.6f}")


def main() -> None:
    """Run train, eval, or train-and-eval pipeline.

    The pipeline reuses the repository ART model class for the baseline, repository
    tensors when present, and a documented synthetic fallback when external Google
    Drive artifacts are absent. Outputs are ``metrics.json``, ``metrics.csv``,
    ``run_metadata.json``, optional plots, and a draft checkpoint.
    """
    args = apply_loss_variant(parse_args())
    repo_root = os.path.abspath(os.path.dirname(__file__))
    output_dir = os.path.join(args.output_dir, args.loss_variant) if args.loss_variant else args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    set_global_seed(args.seed)

    tensors, dataset_label = load_repository_tensors(os.path.join(repo_root, args.dataset_dir), args.seed)
    data, stats, split = normalize_split(tensors)
    num_intervals = data["states"].shape[1]
    time_step_s = args.terminal_time_s / max(num_intervals, 1)
    train_dataset = DraftChunkDataset(
        data["states"][:split], data["actions"][:split], data["rtgs"][:split], data["ctgs"][:split], args.draft_chunk_length
    )
    draft_config = DraftARTConfig(
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        draft_chunk_length=args.draft_chunk_length,
        max_sequence_length=max(128, num_intervals + 1),
    )
    loss_config = DraftLossConfig(u_max_m_s=args.u_max_m_s, accept_threshold=args.accept_threshold)
    draft_model = None
    training_time_s = 0.0
    draft_checkpoint = args.draft_checkpoint
    if args.mode in {"train_draft", "train_and_eval"}:
        result = train_draft_model(
            train_dataset=train_dataset,
            config=draft_config,
            loss_config=loss_config,
            device=device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            output_dir=output_dir,
            use_distillation=args.use_distillation,
            use_physics_loss=args.use_physics_loss,
            use_acceptance_loss=args.use_acceptance_loss,
        )
        draft_model = result.model
        training_time_s = result.training_time_s
        draft_checkpoint = result.checkpoint_path
        print(f"draft_training_time_s: {training_time_s:.6f}")
    if args.mode in {"eval", "train_and_eval"}:
        if draft_model is None:
            if draft_checkpoint is not None and os.path.exists(draft_checkpoint):
                draft_model = load_draft_checkpoint(draft_checkpoint, device)
            else:
                draft_model = __import__("transformer.draft_art", fromlist=["PhysicsInformedDraftART"]).PhysicsInformedDraftART(draft_config).to(device)
        full_art_model = build_full_art_model(6, 3, device, args.full_art_checkpoint)
        full_metrics, rep_states, _ = evaluate_full_art_baseline(full_art_model, data, stats, split, args.num_eval_samples, device)
        draft_metrics, draft_rep_states, _ = evaluate_draft_model(draft_model, data, stats, split, args.num_eval_samples, device)
        metrics: Dict[str, Any] = {
            "draft_training_time_s": training_time_s,
            "full_art": full_metrics,
            "draft": draft_metrics,
            "physics_draft": draft_metrics if args.use_physics_loss else {},
            "scp": {
                "scp_runtime_mean_s": float("nan"),
                "scp_success_rate": float("nan"),
                "scp_iteration_count_mean": float("nan"),
                "scp_infeasibility_rate": float("nan"),
                "note": "SCP aggregation is not run by default; repository optimization/ocp.py remains available for warm-start refinement.",
            },
        }
        if args.use_speculative_inference:
            initial_state = data["states"][split : split + 1, 0]
            initial_action = torch.zeros(1, 3)
            initial_fuel = data["rtgs"][split : split + 1, :1].view(1, 1)
            initial_ctg = data["ctgs"][split : split + 1, :1].view(1, 1)
            result = speculative_inference(
                initial_state=initial_state,
                initial_action=initial_action,
                initial_fuel_to_go=initial_fuel,
                initial_constraint_to_go=initial_ctg,
                full_art_verifier=None,
                draft_model=draft_model,
                num_intervals=num_intervals,
                draft_chunk_length=args.draft_chunk_length,
                time_step_s=time_step_s,
                control_scale_m_s=loss_config.control_scale_m_s,
                accept_threshold=args.accept_threshold,
                u_max_m_s=args.u_max_m_s,
                device=device,
            )
            prefixes = result.accepted_prefix_lengths
            rejection_rate = result.rejection_count / max(len(prefixes), 1)
            metrics["speculative"] = {
                "speculative_inference_time_mean_s": result.timing_dictionary["speculative_inference_time_s"],
                "speculative_inference_time_std_s": 0.0,
                "mean_accepted_prefix_length": float(np.mean(prefixes)) if prefixes else 0.0,
                "median_accepted_prefix_length": float(np.median(prefixes)) if prefixes else 0.0,
                "rejection_rate": float(rejection_rate),
                "full_art_call_count": result.full_art_call_count,
                **draft_metrics,
            }
            metrics["accepted_prefix_lengths"] = prefixes
        else:
            metrics["speculative"] = {}
        metrics["end_to_end_runtime_mean_s"] = float(
            metrics.get("draft_training_time_s", 0.0)
            + metrics.get("full_art", {}).get("full_art_inference_time_mean_s", 0.0)
            + metrics.get("draft", {}).get("draft_inference_time_mean_s", 0.0)
        )
        metrics["end_to_end_runtime_std_s"] = 0.0
        save_metrics(metrics, output_dir)
        metadata = collect_run_metadata(
            seed=args.seed,
            device=device,
            repo_root=repo_root,
            dataset_path=dataset_label,
            train_samples=split,
            eval_samples=min(args.num_eval_samples, data["states"].shape[0] - split),
            model_parameter_count=sum(p.numel() for p in draft_model.parameters()),
            full_art_checkpoint=args.full_art_checkpoint,
            draft_checkpoint=draft_checkpoint,
            extra={
                "terminal_time_s": args.terminal_time_s,
                "num_intervals": num_intervals,
                "time_step_s": time_step_s,
                "loss_variant": args.loss_variant,
                "use_speculative_inference": args.use_speculative_inference,
            },
        )
        with open(os.path.join(output_dir, "run_metadata.json"), "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, sort_keys=True)
        if args.plot:
            plot_dir = os.path.join(output_dir, "plots")
            plot_runtime_comparison({
                "Full ART inference": full_metrics.get("full_art_inference_time_mean_s", 0.0),
                "Draft inference": draft_metrics.get("draft_inference_time_mean_s", 0.0),
                "Speculative inference": metrics.get("speculative", {}).get("speculative_inference_time_mean_s", 0.0),
                "Full ART + SCP": 0.0,
                "Speculative + SCP": 0.0,
            }, plot_dir)
            plot_metric_bar({"Full ART": full_metrics.get("fuel_cost_m_s", 0.0), "Draft": draft_metrics.get("fuel_cost_m_s", 0.0)}, "Fuel [m/s]", "fuel_cost_comparison.png", plot_dir)
            plot_metric_bar({"Full ART": full_metrics.get("terminal_position_error_m", 0.0), "Draft": draft_metrics.get("terminal_position_error_m", 0.0)}, "Terminal position error [m]", "terminal_position_error.png", plot_dir)
            plot_metric_bar({"Full ART": full_metrics.get("terminal_velocity_error_m_s", 0.0), "Draft": draft_metrics.get("terminal_velocity_error_m_s", 0.0)}, "Terminal velocity error [m/s]", "terminal_velocity_error.png", plot_dir)
            plot_prefix_histogram(metrics.get("accepted_prefix_lengths", []), plot_dir)
            plot_example_trajectory({"Full ART warm-start": rep_states, "Physics-informed draft warm-start": draft_rep_states}, plot_dir)
        print_tables(metrics)
        print(f"\nSaved metrics to {os.path.join(output_dir, 'metrics.json')}")
    else:
        metadata = collect_run_metadata(
            seed=args.seed,
            device=device,
            repo_root=repo_root,
            dataset_path=dataset_label,
            train_samples=split,
            eval_samples=0,
            model_parameter_count=sum(p.numel() for p in draft_model.parameters()) if draft_model else 0,
            full_art_checkpoint=args.full_art_checkpoint,
            draft_checkpoint=draft_checkpoint,
            extra={"loss_variant": args.loss_variant},
        )
        with open(os.path.join(output_dir, "run_metadata.json"), "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
