"""Freeze a self-contained private Kaggle script without submitting it.

Shared by the Bonsai checkpoint, MTP head and CUDA reference job builders.
The staged `convert.py` embeds the named tool sources verbatim with their
SHA256 digests, so the cloud job imports exactly the reviewed local code.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def stage_script(
    kernel: str,
    output: Path,
    bindings: dict,
    body: str,
    source_files: tuple[str, ...],
    *,
    gpu: bool = True,
) -> Path:
    """Write `convert.py` plus `kernel-metadata.json` for a private kernel."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", kernel):
        raise ValueError("kernel must be owner/slug")
    if type(gpu) is not bool:
        raise ValueError("gpu must be a boolean")
    sources = {
        name: Path(__file__).with_name(name).read_text(encoding="utf-8")
        for name in source_files
    }
    hashes = {
        name: hashlib.sha256(source.encode()).hexdigest()
        for name, source in sources.items()
    }
    bindings = {
        **bindings,
        "SOURCES": sources,
        "SOURCE_SHA256": hashes,
        "ARCHIVE_NAME": kernel.split("/")[1] + ".tar",
    }
    script = (
        "import json\n"
        + "".join(f"{name} = {value!r}\n" for name, value in bindings.items())
        + body
    )
    compile(script, "convert.py", "exec")
    metadata = {
        "id": kernel,
        "title": kernel.split("/")[1],
        "code_file": "convert.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    if gpu:
        metadata["machine_shape"] = "NvidiaTeslaT4"
    output.mkdir(parents=True, exist_ok=False)
    (output / "convert.py").write_text(script, encoding="utf-8")
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output
