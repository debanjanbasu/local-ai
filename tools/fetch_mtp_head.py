"""Download the pinned MTP head release asset and install it beside the model.

The head is published as a GitHub release asset; the bytes are streamed,
size- and SHA256-checked against the pin below, and published without
replacing an existing file (the engine also re-checks the payload digest on
every load).
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from kaggle_bonsai_job import stream_verified

URL = (
    "https://github.com/debanjanbasu/local-ai/releases/download/"
    "mtp-head-mixed-v1/mtp-head-ptq1-v1.bin"
)
SIZE = 166_969_344
SHA256 = "83cc72279159c12255784f8a0a08ddc916519d466811276e4ff621eb7b88bd1b"
DEFAULT_DESTINATION = Path("models/bonsai2-27b-mtp/mtp-head-ptq1-v1.bin")


def fetch(destination: Path = DEFAULT_DESTINATION, opener=None) -> dict:
    """Stream the pinned asset into `destination`, verified and no-clobber."""
    opener = opener or urllib.request.urlopen
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        URL, headers={"Accept-Encoding": "identity", "User-Agent": "local-ai-fetch/1"}
    )
    with opener(request, timeout=120) as response:
        return stream_verified(response, destination, SIZE, SHA256, maximum_size=SIZE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help=f"installed head path (default: {DEFAULT_DESTINATION})",
    )
    args = parser.parse_args()
    record = fetch(args.destination)
    print(args.destination, record["size"], record["sha256"])


if __name__ == "__main__":
    main()
