"""Smoke tests for the AdaptiveStreamFL command-line entry points."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def write_tiny_occupancy_dataset(data_dir: Path) -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(80, 5))
    labels = (features[:, 0] + 0.5 * features[:, 1] > 0).astype(float)
    data = np.column_stack([features, labels])
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "occupancy.npy", data)


class CommandLineSmokeTests(unittest.TestCase):
    def run_cli(self, run_type: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            output_dir = tmp_path / "results"
            write_tiny_occupancy_dataset(data_dir)

            command = [
                sys.executable,
                "-m",
                "adaptivestreamfl",
                "--dataset",
                "occupancy",
                "--data_dir",
                str(data_dir),
                "--output_dir",
                str(output_dir),
                "--clients",
                "2",
                "--run_type",
                run_type,
                "--percent_init",
                "0.2",
                "--local_init",
                "2",
                "--global_init",
                "2",
                "--max_mc",
                "10",
                "--global_mc",
                "20",
                "--reporting_interval",
                "5",
                "--skip_validation",
                "true",
                "--seed",
                "42",
            ]
            result = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            csv_files = list(output_dir.rglob("*.csv"))
            self.assertTrue(csv_files, msg=f"No CSV files produced. stdout:\n{result.stdout}")

    def test_fedstream_baseline_smoke(self) -> None:
        self.run_cli("FedStream")

    def test_adaptivestreamfl_smoke(self) -> None:
        self.run_cli("AdaptiveStreamFL")


if __name__ == "__main__":
    unittest.main()
