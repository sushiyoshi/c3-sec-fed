"""Benchmark federated averaging with optional FHE-protected aggregation."""
from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from federated_cifar10 import run_federated_training


MODES = ["plaintext", "tfhe", "ckks"]


def build_training_args(benchmark_args: argparse.Namespace, mode: str) -> argparse.Namespace:
    args = argparse.Namespace(
        rounds=benchmark_args.rounds,
        clients=benchmark_args.clients,
        clients_per_round=benchmark_args.clients_per_round,
        local_epochs=benchmark_args.local_epochs,
        batch_size=benchmark_args.batch_size,
        lr=benchmark_args.lr,
        momentum=benchmark_args.momentum,
        weight_decay=benchmark_args.weight_decay,
        iid=benchmark_args.iid,
        alpha=benchmark_args.alpha,
        seed=benchmark_args.seed,
        data_dir=benchmark_args.data_dir,
        device=benchmark_args.device,
        aggregation_mode=mode,
        tfhe_bit_width=benchmark_args.tfhe_bit_width,
        tfhe_scaling=benchmark_args.tfhe_scaling,
        ckks_batch_size=benchmark_args.ckks_batch_size,
        ckks_depth=benchmark_args.ckks_depth,
        ckks_scaling_mod_size=benchmark_args.ckks_scaling_mod_size,
        quiet=True,
        max_examples_per_client=benchmark_args.max_examples_per_client,
        use_fake_data=benchmark_args.use_fake_data,
        fake_train_size=benchmark_args.fake_train_size,
        fake_test_size=benchmark_args.fake_test_size,
    )
    return args


def run_benchmarks(args: argparse.Namespace) -> Dict[str, Dict[str, List[float]]]:
    results = {}
    for mode in MODES:
        if mode not in args.modes:
            continue
        print(f"\n=== Running federated learning with {mode.upper()} aggregation ===")
        training_args = build_training_args(args, mode)
        history = run_federated_training(training_args)
        results[mode] = history
    return results


def history_to_records(results: Dict[str, Dict[str, List[float]]]) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    for mode, history in results.items():
        cumulative = 0.0
        for round_idx, round_number in enumerate(history.get("round", [])):
            duration = float(history["round_duration"][round_idx])
            cumulative += duration
            accuracy = float(history["test_accuracy"][round_idx])
            records.append(
                {
                    "mode": mode,
                    "round": int(round_number),
                    "test_accuracy": accuracy,
                    "test_accuracy_pct": accuracy * 100.0,
                    "round_duration": duration,
                    "cumulative_time": cumulative,
                }
            )
    return records


MODE_COLORS = {
    "plaintext": (31, 119, 180),
    "tfhe": (214, 39, 40),
    "ckks": (44, 160, 44),
}


def _extract_mode_series(records: List[Dict[str, float]], mode: str) -> Dict[str, List[float]]:
    rounds = []
    accuracy = []
    cumulative = []
    for record in records:
        if record["mode"] != mode:
            continue
        rounds.append(record["round"])
        accuracy.append(record["test_accuracy_pct"])
        cumulative.append(record["cumulative_time"])
    return {"round": rounds, "accuracy": accuracy, "cumulative": cumulative}


def _draw_axes(
    draw: ImageDraw.ImageDraw,
    bbox: Iterable[int],
    x_label: str,
    y_label: str,
    title: str,
    font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = bbox
    draw.rectangle([left, top, right, bottom], outline=(200, 200, 200), width=1)
    draw.text((left, top - 20), title, fill=(0, 0, 0), font=font)
    draw.text(((left + right) / 2 - 20, bottom + 5), x_label, fill=(0, 0, 0), font=font)
    draw.text((left - 50, (top + bottom) / 2), y_label, fill=(0, 0, 0), font=font)


def _scale_points(
    values: List[float],
    min_val: float,
    max_val: float,
    start: float,
    length: float,
) -> List[float]:
    if not values:
        return []
    span = max(max_val - min_val, 1e-9)
    return [start + ((value - min_val) / span) * length for value in values]


def plot_comparison(records: List[Dict[str, float]], modes: List[str], output_path: str) -> None:
    subset = [record for record in records if record["mode"] in modes]
    if not subset:
        return

    width, height = 1200, 480
    margin = 60
    panel_width = (width - (3 * margin)) // 2
    panel_height = height - (2 * margin)
    accuracy_bbox = (
        margin,
        margin,
        margin + panel_width,
        margin + panel_height,
    )
    runtime_bbox = (
        2 * margin + panel_width,
        margin,
        2 * margin + 2 * panel_width,
        margin + panel_height,
    )

    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    _draw_axes(draw, accuracy_bbox, "Round", "Accuracy (%)", "Accuracy over rounds", font)
    _draw_axes(draw, runtime_bbox, "Round", "Cumulative time (s)", "Runtime over rounds", font)

    all_rounds = sorted({record["round"] for record in subset})
    max_round = max(all_rounds) if all_rounds else 1

    max_accuracy = max(record["test_accuracy_pct"] for record in subset)
    min_accuracy = min(record["test_accuracy_pct"] for record in subset)
    max_time = max(record["cumulative_time"] for record in subset)
    min_time = min(record["cumulative_time"] for record in subset)

    acc_left, acc_top, acc_right, acc_bottom = accuracy_bbox
    run_left, run_top, run_right, run_bottom = runtime_bbox

    round_positions = _scale_points(
        list(range(1, max_round + 1)),
        1,
        max_round,
        acc_left + 10,
        (acc_right - 10) - (acc_left + 10),
    )

    for mode in modes:
        series = _extract_mode_series(records, mode)
        if not series["round"]:
            continue
        color = MODE_COLORS.get(mode, (0, 0, 0))
        x_points = [round_positions[r - 1] for r in series["round"]]
        acc_y = _scale_points(
            series["accuracy"],
            min_accuracy,
            max_accuracy,
            acc_bottom - 10,
            -(acc_bottom - acc_top - 20),
        )
        run_y = _scale_points(
            series["cumulative"],
            min_time,
            max_time,
            run_bottom - 10,
            -(run_bottom - run_top - 20),
        )
        if len(x_points) >= 2:
            draw.line(list(zip(x_points, acc_y)), fill=color, width=3)
            draw.line(list(zip(x_points, run_y)), fill=color, width=3)
        else:
            for x, y in zip(x_points, acc_y):
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
            for x, y in zip(x_points, run_y):
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)

    legend_x = width - margin - 180
    legend_y = margin
    for mode in modes:
        color = MODE_COLORS.get(mode, (0, 0, 0))
        draw.rectangle(
            [legend_x, legend_y, legend_x + 20, legend_y + 12],
            fill=color,
            outline=color,
        )
        draw.text(
            (legend_x + 28, legend_y),
            mode.upper(),
            fill=(0, 0, 0),
            font=font,
        )
        legend_y += 18

    image.save(output_path, format="PNG")
    print(f"Saved comparison plot: {output_path}")


def summarize_results(records: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for record in records:
        mode = record["mode"]
        summary[mode] = {
            "final_accuracy_pct": record["test_accuracy_pct"],
            "total_time_s": record["cumulative_time"],
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark plaintext vs. FHE aggregation.")
    parser.add_argument("--rounds", type=int, default=5, help="Number of global rounds per run.")
    parser.add_argument("--clients", type=int, default=5, help="Number of simulated clients.")
    parser.add_argument(
        "--clients-per-round",
        type=int,
        default=None,
        help="Number of clients participating in each round (defaults to all).",
    )
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per round.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for client loaders.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for client optimizers.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD.")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
        help="Weight decay applied during local training.",
    )
    parser.add_argument("--iid", action="store_true", help="Use IID data partitioning.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet concentration parameter for non-IID splits.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for all runs.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(os.getcwd(), "data"),
        help="Directory where CIFAR-10 is cached.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device to use (defaults to CUDA when available).",
    )
    parser.add_argument(
        "--tfhe-bit-width",
        type=int,
        default=16,
        help="Maximum ciphertext bit-width for TFHE quantization.",
    )
    parser.add_argument(
        "--tfhe-scaling",
        type=float,
        default=2**15,
        help="Default scaling factor applied before TFHE encryption.",
    )
    parser.add_argument(
        "--ckks-batch-size",
        type=int,
        default=8192,
        help="Number of CKKS slots per ciphertext.",
    )
    parser.add_argument(
        "--ckks-depth",
        type=int,
        default=2,
        help="Multiplicative depth for CKKS crypto contexts.",
    )
    parser.add_argument(
        "--ckks-scaling-mod-size",
        type=int,
        default=59,
        help="Scaling modulus size for CKKS contexts.",
    )
    parser.add_argument(
        "--use-fake-data",
        action="store_true",
        help="Use synthetic CIFAR-like data instead of downloading the real dataset.",
    )
    parser.add_argument(
        "--fake-train-size",
        type=int,
        default=1024,
        help="Synthetic training set size when --use-fake-data is provided.",
    )
    parser.add_argument(
        "--fake-test-size",
        type=int,
        default=256,
        help="Synthetic evaluation set size when --use-fake-data is provided.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=MODES,
        choices=MODES,
        help="Aggregation modes to benchmark.",
    )
    parser.add_argument(
        "--max-examples-per-client",
        type=int,
        default=128,
        help="Cap on the number of training samples each client uses during benchmarking.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.getcwd(), "benchmark_artifacts"),
        help="Directory where benchmark plots will be stored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    results = run_benchmarks(args)
    if not results:
        print("No benchmark runs executed; check the requested modes.")
        return
    records = history_to_records(results)
    summary = summarize_results(records)
    print("\nSummary (higher accuracy is better, lower runtime is better):")
    for mode, metrics in summary.items():
        print(
            f"  {mode.upper():9s} | final accuracy: {metrics['final_accuracy_pct']:.2f}% | "
            f"total time: {metrics['total_time_s']:.2f}s"
        )

    comparisons = [
        ["plaintext", "tfhe"],
        ["plaintext", "ckks"],
        ["plaintext", "tfhe", "ckks"],
    ]
    for modes in comparisons:
        available = [mode for mode in modes if mode in results]
        if len(available) < 2:
            continue
        filename = f"accuracy_runtime_{'_'.join(available)}.png"
        output_path = os.path.join(args.output_dir, filename)
        plot_comparison(records, available, output_path)


if __name__ == "__main__":
    main()
