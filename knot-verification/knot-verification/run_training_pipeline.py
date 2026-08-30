#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "ultralytics>=8.3.0",
#   "torch>=2.1",
#   "torchvision>=0.16",
#   "transformers>=4.42.0",
#   "scikit-learn>=1.4",
#   "numpy",
#   "pandas",
#   "pillow",
#   "opencv-python-headless",
#   "joblib",
#   "tqdm",
#   "omegaconf",
#   "python-dotenv",
# ]
# ///
"""Run the complete knot-verification training and evaluation workflow.

For Hugging Face Jobs, run this file with ``hf jobs uv run``.  The job
container is ephemeral, so the required OpenCV system library is installed
at the beginning of each job when it is not already available.
"""
import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path


# A UV Job normally mounts only the submitted script at /data.  Set
# KNOT_PROJECT_DIR to a volume containing the complete repository when
# launching on Hugging Face Jobs (for example, /workspace).
PROJECT_ROOT = Path(os.environ.get("KNOT_PROJECT_DIR", Path(__file__).resolve().parent)).resolve()


def ensure_opencv_system_dependencies() -> None:
    """Install libGL only when the current Linux container lacks it."""
    if os.name != "posix":
        return

    try:
        ctypes.CDLL("libGL.so.1")
        return
    except OSError:
        pass

    apt_get = shutil.which("apt-get")
    is_root = hasattr(os, "geteuid") and os.geteuid() == 0
    if not apt_get or not is_root:
        raise RuntimeError(
            "OpenCV requires libGL.so.1, but this job cannot install it. "
            "Use a Docker image that installs libgl1 and libglib2.0-0."
        )

    print("Installing OpenCV system dependencies (libgl1, libglib2.0-0)...")
    subprocess.run([apt_get, "update"], check=True)
    subprocess.run(
        [apt_get, "install", "-y", "--no-install-recommends", "libgl1", "libglib2.0-0"],
        check=True,
    )


def run(script: str) -> None:
    script_path = PROJECT_ROOT / script
    if not script_path.is_file():
        raise FileNotFoundError(
            f"Required script not found: {script_path}. For Hugging Face Jobs, "
            "mount the complete project and set KNOT_PROJECT_DIR to that mount."
        )
    print(f"\n{'=' * 72}\nRunning: python {script}\n{'=' * 72}", flush=True)
    subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT, check=True)


def main() -> None:
    if not PROJECT_ROOT.is_dir():
        raise FileNotFoundError(f"Project directory does not exist: {PROJECT_ROOT}")

    # Do not let a locally committed .env point the cloud job at a Windows
    # path.  All child scripts must resolve their project files in the mount.
    os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
    yolo_config_dir = PROJECT_ROOT / ".ultralytics"
    yolo_config_dir.mkdir(exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config_dir))

    ensure_opencv_system_dependencies()

    if os.getenv("AUDIT_YOLO_ONLY") == "1":
        run("audit_yolo_dataset.py")
        return

    if os.getenv("DIAGNOSTICS_ONLY") == "1":
        run("diagnose_pipeline.py")
        return

    if os.getenv("YOLO_OVERFIT_ONLY") == "1":
        run("overfit_yolo_sanity.py")
        return

    # Regenerate these in the Linux mount. In particular, data.yaml must not
    # retain an absolute Windows path created by a local dataset build.
    run("data_prep/make_splits.py")
    run("data_prep/build_yolo_dataset.py")
    run("train_detector.py")
    run("data_prep/build_classifier_dataset.py")
    run("train_classifier.py")
    run("evaluate.py")


if __name__ == "__main__":
    main()
