"""Command-line argument helpers for AdaptiveStreamFL."""

from __future__ import annotations

import argparse


def str2bool(value):
    """Parse flexible boolean values used by legacy command lines."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def apply_component_mode(config, mode):
    """Map a named component mode to switches for a single run."""
    config.component_mode = mode
    if mode == "baseline":
        config.dynamic_k_adjustment = False
        config.enable_prototype_learning = False
        config.enable_hierarchical_federated_learning = False
        config.enable_uncertainty_quantification = False
    elif mode == "k_only":
        config.dynamic_k_adjustment = True
        config.enable_prototype_learning = False
        config.enable_hierarchical_federated_learning = False
        config.enable_uncertainty_quantification = False
    elif mode == "prototype_only":
        config.dynamic_k_adjustment = False
        config.enable_prototype_learning = True
        config.enable_hierarchical_federated_learning = False
        config.enable_uncertainty_quantification = False
    elif mode == "full":
        config.dynamic_k_adjustment = True
        config.enable_prototype_learning = True
        config.enable_hierarchical_federated_learning = True
        config.enable_uncertainty_quantification = True


def args_parser(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="covtype",
        choices=[
            "covtype",
            "kddcup99",
            "mnist",
            "adult",
            "electricity",
            "poker",
            "occupancy",
            "shuttle",
        ],
        help="Dataset name.",
    )
    parser.add_argument("--clients", type=int, default=10, help="Number of clients")
    parser.add_argument("--max_mc", type=int, default=200, help="Max client micro-clusters")
    parser.add_argument("--global_mc", type=int, default=1000, help="Max global micro-clusters")
    parser.add_argument("--features", type=int, default=54, help="Number of dataset features")
    parser.add_argument("--decay_rate", type=float, default=0.00002, help="Decay rate")
    parser.add_argument("--local_init", type=int, default=50, help="Local initial clusters")
    parser.add_argument("--reporting_interval", type=int, default=1000, help="Reporting interval")
    parser.add_argument("--percent_init", type=float, default=0.1, help="Initial cluster percentage")
    parser.add_argument(
        "--run_type",
        choices=["FedStream", "AdaptiveStreamFL"],
        default="AdaptiveStreamFL",
        help="Run mode",
    )
    parser.add_argument("--data_dir", type=str, default="dataset", help="Directory containing prepared .npy datasets")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory for runtime logs and CSV results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible experiments")
    parser.add_argument("--adaptive_mode", type=str2bool, default=True, const=True, nargs="?", help="Enable adaptive mode")
    parser.add_argument(
        "--dynamic_k_adjustment",
        type=str2bool,
        default=True,
        const=True,
        nargs="?",
        help="Enable dynamic k-neighbor adjustment",
    )
    parser.add_argument("--min_accuracy_threshold", type=float, default=0.85, help="Minimum accuracy threshold")
    parser.add_argument(
        "--enable_prototype_learning",
        type=str2bool,
        default=True,
        const=True,
        nargs="?",
        help="Enable adaptive prototype learning",
    )

    # Legacy options accepted for older command lines. They are parsed but not
    # used by the clean reference implementation.
    parser.add_argument("--global_init", type=int, default=50, help=argparse.SUPPRESS)
    parser.add_argument("--initial_stream_size", type=int, default=1000, help=argparse.SUPPRESS)
    parser.add_argument("--client_initial_size", type=int, default=500, help=argparse.SUPPRESS)
    parser.add_argument("--weight_const", type=float, default=0.6, help=argparse.SUPPRESS)
    parser.add_argument("--skip_validation", type=str2bool, default=False, const=True, nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("--smart_cluster_merge", type=str2bool, default=True, const=True, nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("--adaptive_weight_balance", type=str2bool, default=True, const=True, nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("--performance_monitoring", type=str2bool, default=True, const=True, nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("--adaptation_frequency", type=int, default=100, help=argparse.SUPPRESS)
    parser.add_argument("--enable_component_tracking", type=str2bool, default=False, const=True, nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("--component_mode", choices=["baseline", "k_only", "prototype_only", "full"], default="full", help=argparse.SUPPRESS)
    parser.add_argument("--prototype_quality_threshold", type=float, default=0.8, help=argparse.SUPPRESS)

    return parser.parse_args(argv)
