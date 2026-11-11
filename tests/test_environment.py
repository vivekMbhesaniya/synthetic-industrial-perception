"""Guard tests: fail loudly if the project runs in the wrong environment."""

import sys
from pathlib import Path

import pytest

import synperc


def test_python_version_is_312_or_newer():
    assert sys.version_info >= (3, 12), f"Wrong interpreter: {sys.executable}"


@pytest.mark.skipif(sys.platform != "win32", reason="Local-machine guard only")
def test_running_inside_sip_environment():
    env_name = Path(sys.prefix).name
    assert env_name == "sip", f"Expected the 'sip' environment, got {sys.prefix}"


def test_package_is_imported_from_src():
    package_dir = Path(synperc.__file__).resolve().parent
    assert package_dir.name == "synperc"
    assert package_dir.parent.name == "src", f"Imported from {package_dir}"
    assert synperc.__version__ == "0.1.0"