from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aerolab.cli import main  # noqa: E402

if __name__ == "__main__":
    log_dir = PROJECT_ROOT / "outputs"
    log_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = (log_dir / "aerolab-app.out.log").open("a", encoding="utf-8", buffering=1)
    sys.stderr = (log_dir / "aerolab-app.err.log").open("a", encoding="utf-8", buffering=1)
    raise SystemExit(
        main(
            [
                "app",
                "--host",
                "127.0.0.1",
                "--port",
                "8765",
                "--root",
                str(PROJECT_ROOT),
            ]
        )
    )
