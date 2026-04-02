"""Tests for I/O modules: TIFF loading, ground truth parsing, nuclei I/O."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from starrynite_py.io.tiff_loader import discover_timepoints, load_timepoint
from starrynite_py.io.ground_truth import (
    load_ground_truth,
    read_nuclei_file,
    discover_nuclei_files,
    parse_nuclei_line,
)
from starrynite_py.io.nuclei_io import write_nuclei_file, read_nuclei_positions
from starrynite_py.io.acetree_export import export_acetree_zip
from starrynite_py.config import load_config


class TestGroundTruthParsing:
    """Test ground truth nuclei file parsing."""

    def test_parse_single_line(self):
        line = "1, 1, -1, 1, -1, 333, 112, 122.0, 35, P1, 1703, 0, 0, 0, , 0, 0, 0, 0, 0, "
        nuc = parse_nuclei_line(line)
        assert nuc is not None
        assert nuc.id == 1
        assert nuc.valid is True
        assert nuc.predecessor_id == -1
        assert nuc.successor_id == 1
        assert nuc.x == 333.0
        assert nuc.y == 112.0
        assert nuc.z == 122.0
        assert nuc.diameter == 35.0
        assert nuc.name == "P1"
        assert nuc.intensity == 1703.0

    def test_parse_unnamed_nucleus(self):
        line = "3, 0, -1, -1, -1, 61, 123, 64.0, 21, , 432, 0, 0, 0, , 0, 0, 0, 0, 0, "
        nuc = parse_nuclei_line(line)
        assert nuc is not None
        assert nuc.valid is False
        assert nuc.name == ""
        assert nuc.diameter == 21.0

    def test_parse_empty_line(self):
        assert parse_nuclei_line("") is None
        assert parse_nuclei_line("   ") is None

    def test_discover_nuclei_files(self, embryo1_nuclei):
        if not embryo1_nuclei.exists():
            pytest.skip("Test data not available")
        files = discover_nuclei_files(embryo1_nuclei)
        assert len(files) > 0
        # Should be sorted by timepoint
        times = [t for t, _ in files]
        assert times == sorted(times)
        # First timepoint should be 1
        assert times[0] == 1

    def test_load_ground_truth_embryo1(self, embryo1_nuclei):
        if not embryo1_nuclei.exists():
            pytest.skip("Test data not available")
        gt = load_ground_truth(embryo1_nuclei)
        assert len(gt) > 0

        # t001 should have a small number of nuclei (early embryo)
        if 1 in gt:
            t1 = gt[1]
            assert len(t1.nuclei) >= 2  # At least P1 and AB
            valid = t1.valid_nuclei
            named = t1.named_nuclei
            assert len(named) >= 2
            names = {n.name for n in named}
            assert "P1" in names or "AB" in names

    def test_nucleus_count_growth(self, embryo1_nuclei):
        """Verify nucleus count grows over developmental time."""
        if not embryo1_nuclei.exists():
            pytest.skip("Test data not available")
        gt = load_ground_truth(embryo1_nuclei)
        times = sorted(gt.keys())
        early_count = len(gt[times[0]].valid_nuclei)
        late_count = len(gt[times[-1]].valid_nuclei)
        assert late_count > early_count


class TestTiffLoader:
    """Test TIFF file discovery and loading."""

    def test_discover_timepoints(self, embryo1_images):
        if not embryo1_images.exists():
            pytest.skip("Test data not available")
        timepoints = discover_timepoints(embryo1_images, "nih_diSPIM_deconv_1")
        assert len(timepoints) > 0
        times = [t for t, _ in timepoints]
        assert times == sorted(times)

    def test_load_single_timepoint(self, embryo1_images):
        if not embryo1_images.exists():
            pytest.skip("Test data not available")
        timepoints = discover_timepoints(embryo1_images, "nih_diSPIM_deconv_1")
        if not timepoints:
            pytest.skip("No TIFF files found")

        _, path = timepoints[0]
        volume = load_timepoint(path)
        assert volume.ndim == 3  # Z, Y, X
        assert volume.dtype == np.float32
        assert volume.shape[0] > 1  # Multiple Z slices


class TestNucleiIO:
    """Test nuclei file write/read roundtrip."""

    def test_roundtrip(self):
        positions = np.array([[100.0, 200.0, 50.0], [150.0, 250.0, 60.0]])
        diameters = np.array([30.0, 25.0])
        names = ["P1", "AB"]

        with tempfile.NamedTemporaryFile(mode="w", suffix="-nuclei", delete=False) as f:
            path = f.name

        write_nuclei_file(path, positions, diameters, names=names)

        # Read back
        nuclei = read_nuclei_file(path)
        assert len(nuclei) == 2
        assert nuclei[0].name == "P1"
        assert nuclei[1].name == "AB"
        assert abs(nuclei[0].x - 100.0) < 1.0
        assert abs(nuclei[0].diameter - 30.0) < 1.0

        Path(path).unlink()


class TestAceTreeExport:
    """Test acetree_py ZIP export."""

    def test_export_zip(self):
        timepoints = {
            1: {
                "positions": np.array([[100.0, 200.0, 50.0], [150.0, 250.0, 60.0]]),
                "diameters": np.array([30.0, 25.0]),
                "names": ["P1", "AB"],
            },
            2: {
                "positions": np.array([[102.0, 198.0, 51.0], [148.0, 252.0, 59.0]]),
                "diameters": np.array([31.0, 26.0]),
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test.zip"
            result = export_acetree_zip(out_path, timepoints)
            assert result.exists()
            assert result.stat().st_size > 0

            # Verify ZIP contents
            import zipfile
            with zipfile.ZipFile(result) as zf:
                names = zf.namelist()
                assert "nuclei/t001-nuclei" in names
                assert "nuclei/t002-nuclei" in names


class TestConfigLoading:
    """Test YAML configuration loading."""

    def test_load_nih_dispim_config(self, sample_config_path):
        if not sample_config_path.exists():
            pytest.skip("Config file not available")
        cfg = load_config(sample_config_path)
        assert cfg.imaging.xy_res == 0.1625
        assert cfg.tracker == "ultrack"
        assert cfg.stardist.model_name == "3D_demo"
