"""Stage a private CPU-only Kaggle job that fetches the pinned Whisper large-v3 files.

Whisper large-v3 is not on Kaggle Models, and this machine does not download
from Hugging Face directly, so a private Kaggle CPU kernel downloads the public
MIT-licensed files at a pinned revision, checks every size and SHA-256 against
the converter's pins, and saves them as its output:

  python3 tools/kaggle_whisper_job.py --kernel <owner>/whisper-large-v3-fetch \\
      --output cache/whisper-job
  kaggle kernels push -p cache/whisper-job
  kaggle kernels output <owner>/whisper-large-v3-fetch -p cache/whisper-large-v3
  python3 tools/convert_whisper.py --source cache/whisper-large-v3

The kernel writes `model/model.safetensors`, `model/config.json`,
`model/generation_config.json`, `model/tokenizer.json` (openai/whisper-large-v3
@ REVISION) and `mel_filters.npz` (from the openai-whisper 20250625 sdist on
PyPI, whose SHA-256 is pinned too). Staging never submits anything.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import convert_whisper
import kaggle_staging

MODEL_ID = "openai/whisper-large-v3"
REVISION = "06f233fe06e710322aca913c1bc4249a0d71fce1"
SDIST = "openai_whisper-20250625.tar.gz"
SDIST_URL = ("https://files.pythonhosted.org/packages/source/o/openai-whisper/"
             "openai_whisper-20250625.tar.gz")
SDIST_SHA256 = "37a91a3921809d9f44748ffc73c0a55c9f366c85a3ef5c2ae0cc09540432eb96"
SOURCE_FILES = ("kaggle_whisper_job.py", "convert_whisper.py")

BODY = r'''
import hashlib
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

out = Path("/kaggle/working")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def check(path, size, digest):
    if os.path.getsize(path) != size or sha256(path) != digest:
        raise SystemExit(f"{path}: does not match its pin")


for name, (size, digest) in PINS.items():
    target = out / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if name.startswith("model/"):
        url = (f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/"
               f"{name.removeprefix('model/')}")
        with urllib.request.urlopen(url, timeout=600) as response, open(target, "wb") as dest:
            shutil.copyfileobj(response, dest, 1 << 22)
        check(target, size, digest)
sdist = Path("/tmp") / SDIST
with urllib.request.urlopen(SDIST_URL, timeout=600) as response, open(sdist, "wb") as dest:
    shutil.copyfileobj(response, dest, 1 << 22)
if sha256(sdist) != SDIST_SHA256:
    raise SystemExit("openai-whisper sdist does not match its pin")
with tarfile.open(sdist) as archive:
    member = archive.getmember("openai_whisper-20250625/whisper/assets/mel_filters.npz")
    with archive.extractfile(member) as source, open(out / "mel_filters.npz", "wb") as dest:
        shutil.copyfileobj(source, dest)
check(out / "mel_filters.npz", *PINS["mel_filters.npz"])
print(json.dumps({name: list(pin) for name, pin in PINS.items()}, indent=1))
'''


def stage(kernel: str, output: Path) -> Path:
    """Write the kernel script and metadata into `output` (must not exist)."""
    bindings = {
        "MODEL_ID": MODEL_ID,
        "REVISION": REVISION,
        "SDIST": SDIST,
        "SDIST_URL": SDIST_URL,
        "SDIST_SHA256": SDIST_SHA256,
        "PINS": {name: list(pin) for name, pin in convert_whisper.SOURCE_FILES.items()},
    }
    return kaggle_staging.stage_script(kernel, output, bindings, BODY, SOURCE_FILES, gpu=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kernel", required=True, help="owner/slug of the private kernel")
    parser.add_argument("--output", type=Path, required=True, help="new staging directory")
    args = parser.parse_args()
    print(stage(args.kernel, args.output))


if __name__ == "__main__":
    main()
