"""Command-line interface for the StarryNite pipeline."""

from __future__ import annotations

import logging
import sys

import click


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def main(verbose: bool) -> None:
    """StarryNite: Nuclear detection, tracking, and lineaging for C. elegans."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


@main.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="YAML config file.")
def detect(config: str) -> None:
    """Run nuclear detection only."""
    from starrynite.config import load_config
    from starrynite.pipeline import run_detection

    cfg = load_config(config)
    detections = run_detection(cfg)
    click.echo(f"Detected nuclei in {len(detections)} timepoints.")
    for t in sorted(detections.keys()):
        click.echo(f"  t={t:03d}: {len(detections[t].centroids)} nuclei")


@main.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="YAML config file.")
def run(config: str) -> None:
    """Run the full pipeline: detect -> track -> export."""
    from starrynite.config import load_config
    from starrynite.pipeline import run_pipeline

    cfg = load_config(config)
    detections, tracking, export_path = run_pipeline(cfg)
    click.echo(f"Pipeline complete. Output: {export_path}")


@main.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="YAML config file.")
def evaluate(config: str) -> None:
    """Evaluate detection/tracking against ground truth."""
    from starrynite.config import load_config
    from starrynite.io.ground_truth import load_ground_truth
    from starrynite.pipeline import run_detection
    from starrynite.evaluation.detection_eval import evaluate_detection, summarize_detection_metrics

    cfg = load_config(config)

    if cfg.data.ground_truth_dir is None:
        click.echo("Error: ground_truth_dir must be set in config for evaluation.", err=True)
        sys.exit(1)

    gt = load_ground_truth(cfg.data.ground_truth_dir)
    detections = run_detection(cfg)

    metrics = []
    for t in sorted(set(detections.keys()) & set(gt.keys())):
        m = evaluate_detection(detections[t], gt[t], anisotropy=cfg.imaging.anisotropy)
        metrics.append(m)
        click.echo(f"  t={t:03d}: P={m.precision:.3f} R={m.recall:.3f} F1={m.f1:.3f} (TP={m.true_positives} FP={m.false_positives} FN={m.false_negatives})")

    summary = summarize_detection_metrics(metrics)
    click.echo(f"\nOverall: P={summary['precision']:.3f} R={summary['recall']:.3f} F1={summary['f1']:.3f}")


if __name__ == "__main__":
    main()
