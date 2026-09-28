"""Validate runtime dependencies, algorithm config, and optional local weights."""

import argparse
import json
from pathlib import Path
import sys


def check(config_path: Path, check_model_files: bool = False):
    checks = []
    errors = []
    try:
        import torch

        checks.append({"name": "torch", "ok": True, "version": torch.__version__})
        checks.append(
            {
                "name": "cuda",
                "ok": torch.cuda.is_available(),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except ImportError as error:
        errors.append("torch: {}".format(error))

    try:
        import yaml

        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        required = ("algorithm", "backbone", "point_encoder", "fusion", "fast_system")
        missing = [name for name in required if name not in payload]
        if missing:
            errors.append("config missing sections: {}".format(", ".join(missing)))
        checks.append({"name": "config", "ok": not missing, "path": str(config_path)})
    except (ImportError, OSError, ValueError) as error:
        payload = {}
        errors.append("config: {}".format(error))

    try:
        import transformers

        checks.append(
            {"name": "transformers", "ok": True, "version": transformers.__version__}
        )
    except ImportError as error:
        errors.append("transformers: {}".format(error))

    if check_model_files and payload:
        model_path = Path(payload["backbone"]["local_model_path"])
        required_files = ("config.json", "preprocessor_config.json")
        missing_files = [name for name in required_files if not (model_path / name).is_file()]
        has_weights = any(model_path.glob("*.safetensors"))
        if not has_weights:
            missing_files.append("*.safetensors")
        checks.append(
            {
                "name": "local_model",
                "ok": not missing_files,
                "path": str(model_path),
            }
        )
        if missing_files:
            errors.append("local model missing: {}".format(", ".join(missing_files)))

    return {"ok": not errors, "checks": checks, "errors": errors}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/power_vln/algorithm.yaml")
    )
    parser.add_argument("--check-model-files", action="store_true")
    arguments = parser.parse_args(argv)
    result = check(arguments.config, arguments.check_model_files)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
