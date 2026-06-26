"""Single-container dispatcher.

One image, built five times with a different ``COMPONENT`` build-arg (see
deploy/Dockerfile). Each Cloud Run unit sets COMPONENT and this routes to the
right launch command — module runner for the batch Jobs, server for the
Services. Heterogeneous launch shapes (python -m / uvicorn / streamlit) are why
a dispatcher is cleaner than a per-component ENTRYPOINT.
"""

from __future__ import annotations

import os
import runpy
import subprocess
import sys

COMPONENT = os.environ.get("COMPONENT", "")
PORT = os.environ.get("PORT", "8080")

JOB_MODULES = {
    "profiles": "preprocessing.sources.profiles.runner",
    "weblinks": "preprocessing.sources.weblinks.runner",
    "ingest": "preprocessing.ingest.runner",
}


def main() -> None:
    if COMPONENT in JOB_MODULES:
        # Pass through any extra CLI args (e.g. --limit) after the script name.
        sys.argv = [COMPONENT, *sys.argv[1:]]
        runpy.run_module(JOB_MODULES[COMPONENT], run_name="__main__")
    elif COMPONENT == "api":
        subprocess.run(
            ["uvicorn", "serving.api.app:app", "--host", "0.0.0.0", "--port", PORT],
            check=True,
        )
    elif COMPONENT == "frontend":
        subprocess.run(
            [
                "streamlit", "run", "serving/frontend/app.py",
                "--server.port", PORT,
                "--server.address", "0.0.0.0",
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false",
            ],
            check=True,
        )
    else:
        sys.exit(f"unknown COMPONENT={COMPONENT!r}; expected one of "
                 f"{sorted([*JOB_MODULES, 'api', 'frontend'])}")


if __name__ == "__main__":
    main()
