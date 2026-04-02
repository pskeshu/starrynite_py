"""Shared test fixtures."""

from pathlib import Path

import pytest

# Test data paths
TEST_DATA_BASE = Path(r"D:\nih-ls")
EMBRYO1_DIR = TEST_DATA_BASE / "nih_diSPIM_deconv_1"
EMBRYO2_DIR = TEST_DATA_BASE / "nih_diSPIM_deconv_2"
EMBRYO3_DIR = TEST_DATA_BASE / "nih_diSPIM_deconv_3"


@pytest.fixture
def embryo1_images():
    return EMBRYO1_DIR / "images"


@pytest.fixture
def embryo1_nuclei():
    return EMBRYO1_DIR / "tracks" / "nuclei"


@pytest.fixture
def embryo2_nuclei():
    return EMBRYO2_DIR / "tracks" / "nuclei"


@pytest.fixture
def embryo3_nuclei():
    return EMBRYO3_DIR / "tracks" / "nuclei"


@pytest.fixture
def sample_config_path():
    return Path(r"C:\Users\christensenr\Documents\GitHub\starrynite-python\configs\nih_dispim.yaml")
