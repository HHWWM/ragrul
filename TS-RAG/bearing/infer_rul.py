from __future__ import annotations

import argparse
import json

from bearing.bearing_inference_runtime import BearingRULRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot RUL inference (optimized runtime)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--bearing_dir", type=str, required=True)
    args = parser.parse_args()

    runtime = BearingRULRuntime(args.config)
    result = runtime.predict(args.bearing_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
