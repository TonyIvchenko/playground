"""Build standalone Hugging Face Space bundles for service deployment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import textwrap


@dataclass(frozen=True)
class SpaceConfig:
    key: str
    title: str
    emoji: str
    service_package: str
    training_script: str
    sample_csv: str
    description: str


CONFIGS = {
    "hurricane": SpaceConfig(
        key="hurricane",
        title="Hurricane Intensity Risk",
        emoji="🌀",
        service_package="hurricane_service",
        training_script="train_hurricane_intensity_model.py",
        sample_csv="hurricane_training_sample.csv",
        description="Self-contained hurricane intensity-risk API with embedded Gradio UI.",
    ),
    "wildfire": SpaceConfig(
        key="wildfire",
        title="Wildfire Ignition Risk",
        emoji="🔥",
        service_package="wildfire_service",
        training_script="train_wildfire_ignition_model.py",
        sample_csv="wildfire_training_sample.csv",
        description="Self-contained wildfire ignition-risk API with embedded Gradio UI.",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--service",
        required=True,
        choices=sorted(CONFIGS.keys()),
        help="Service key to bundle",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for standalone Space context",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_readme(target: Path, cfg: SpaceConfig) -> None:
    front_matter = textwrap.dedent(
        f"""\
        ---
        title: {cfg.title}
        emoji: {cfg.emoji}
        colorFrom: red
        colorTo: orange
        sdk: docker
        app_port: 7860
        ---
        """
    )
    body = textwrap.dedent(
        f"""\
        # {cfg.title}

        {cfg.description}

        ## Endpoints
        - `GET /health`
        - `POST /predict`
        - `GET /service-metadata`
        - `GET /ui`
        """
    )
    target.write_text(front_matter + "\n" + body)


def _write_dockerfile(target: Path, cfg: SpaceConfig) -> None:
    dockerfile = textwrap.dedent(
        f"""\
        FROM python:3.11-slim

        WORKDIR /app

        COPY requirements.txt /tmp/requirements.txt
        RUN pip install --no-cache-dir -r /tmp/requirements.txt

        COPY src /app/src
        COPY scripts /app/scripts
        COPY data /app/data

        ENV PYTHONPATH=/app/src
        ENV MODEL_BUNDLE_PATH=/app/model/model_bundle.joblib
        ENV API_PORT=7860
        ENV UI_PATH=/

        RUN python /app/scripts/{cfg.training_script} \\
            --allow-demo-data \\
            --input-csv /app/data/sample/{cfg.sample_csv} \\
            --output-path /app/model/model_bundle.joblib \\
            --model-version 2026.03.v1

        EXPOSE 7860

        CMD ["python", "-m", "{cfg.service_package}.main"]
        """
    )
    target.write_text(dockerfile)


def build_bundle(cfg: SpaceConfig, output_dir: Path) -> Path:
    root = repo_root()
    target = output_dir
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    _copy_tree(root / "src" / cfg.service_package, target / "src" / cfg.service_package)
    _copy_file(root / "scripts" / cfg.training_script, target / "scripts" / cfg.training_script)
    _copy_file(root / "data" / "sample" / cfg.sample_csv, target / "data" / "sample" / cfg.sample_csv)

    requirements = (root / "src" / cfg.service_package / "requirements.txt").read_text()
    (target / "requirements.txt").write_text(requirements)

    _write_dockerfile(target / "Dockerfile", cfg)
    _write_readme(target / "README.md", cfg)

    (target / ".dockerignore").write_text("__pycache__/\n*.pyc\n")
    return target


def main() -> None:
    args = parse_args()
    cfg = CONFIGS[args.service]
    output = args.output_dir
    if output is None:
        output = repo_root() / "dist" / "hf_spaces" / cfg.key

    built = build_bundle(cfg=cfg, output_dir=output)
    print(f"Built HF Space bundle: {built}")


if __name__ == "__main__":
    main()
