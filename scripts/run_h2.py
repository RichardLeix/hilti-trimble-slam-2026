#!/usr/bin/env python3
"""Run the fixed H2 configuration and export metric camera-0 TUM trajectories."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "normal_source_states": 12, "retained_states": 11,
    "active_landmark_cap": 240, "camera_information": [1.0, 0.5],
    "landmark_quality_weighting": "C_quality",
    "landmark_admission": "B3_native_camera1_floor_with_protected_prior",
}


def environment(parent=None):
    env = dict(os.environ if parent is None else parent)
    for key in list(env):
        if key.startswith("HILTI_") or key in ("LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT"):
            del env[key]
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", OPENCV_FOR_THREADS_NUM="1")
    return env


def configuration(path):
    path = path.absolute()
    cfg = json.loads(path.read_text())
    for key, value in EXPECTED.items():
        if cfg.get(key) != value:
            raise ValueError("Not the fixed H2 configuration: " + key)
    files = {key: (path.parent / cfg[key]).absolute() for key in
             ("imu", "camera", "loop", "pose_graph", "global", "backend", "startup")}
    for file in files.values():
        if not file.is_file():
            raise FileNotFoundError(file)
    return files


def run(args):
    files = configuration(args.config)
    bag, vocab, masks = args.bag.absolute(), args.vocabulary.absolute(), args.mask_dir.absolute()
    if not bag.is_dir() or not any(bag.glob("*.db3")):
        raise ValueError("--bag must be the original ROS2 SQLite bag directory")
    if not vocab.is_file():
        raise FileNotFoundError(vocab)
    mask_paths = {}
    for camera in (0, 1):
        name = "hilti_mask_cam" + str(camera) + ".png"
        official = masks / ("mask_cam" + str(camera) + ".png")
        selected = masks / name if (masks / name).is_file() else official
        if not selected.is_file():
            raise FileNotFoundError("Supply the official calibration mask: " + str(official))
        mask_paths[name] = selected
    prefix = args.output_prefix.absolute()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if any(prefix.parent.glob(prefix.name + "_*")):
        raise FileExistsError("Output prefix already exists; choose a fresh prefix")
    # Only external mask locations change. Every calibration number is retained.
    camera = Path(str(prefix) + "_camera.yaml")
    text = files["camera"].read_text()
    text, count = re.subn(r"(?m)^(\s*mask:\s*)(hilti_mask_cam[01]\.png)\s*$",
                         lambda m: m[1] + json.dumps(str(mask_paths[m[2]])), text)
    if count != 2:
        raise ValueError("Expected the two frozen camera mask references")
    camera.write_text(text)
    build = args.build_dir.absolute()
    executable, exporter = build / "bin/run_hilti_sequence", build / "bin/export_hilti_submission"
    for binary in (executable, exporter):
        if not binary.is_file():
            raise FileNotFoundError("Build H2 first: " + str(binary))
    command = [str(executable), str(bag), str(files["imu"]), str(camera),
               str(files["loop"]), str(vocab), str(files["pose_graph"]), str(files["global"]),
               str(prefix), "--backend-config", str(files["backend"]),
               "--startup-config", str(files["startup"])]
    env = environment()
    subprocess.run(command, env=env, check=True)
    subprocess.run([str(exporter), str(bag), str(files["imu"]), str(camera),
                    str(prefix) + "_local.csv", str(prefix) + "_local_keyframes.csv",
                    str(prefix) + "_global_keyframes.csv", args.sequence,
                    str(prefix) + "_trajectories"], env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    verify = sub.add_parser("verify", help="Run synthetic H2 unit checks; no dataset or GT")
    verify.add_argument("--build-dir", type=Path, default=ROOT / "build")
    replay = sub.add_parser("run")
    replay.add_argument("--bag", type=Path, required=True)
    replay.add_argument("--vocabulary", type=Path, required=True)
    replay.add_argument("--mask-dir", type=Path, required=True)
    replay.add_argument("--sequence", required=True)
    replay.add_argument("--output-prefix", type=Path, required=True)
    replay.add_argument("--config", type=Path, default=ROOT / "config/h2.json")
    replay.add_argument("--build-dir", type=Path, default=ROOT / "build")
    args = parser.parse_args()
    try:
        if args.mode == "verify":
            configuration(ROOT / "config/h2.json")
            subprocess.run(["ctest", "--test-dir", str(args.build_dir.absolute()),
                            "--output-on-failure"], env=environment(), check=True)
        else:
            run(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, "error: " + str(error) + "\n")


if __name__ == "__main__":
    main()
