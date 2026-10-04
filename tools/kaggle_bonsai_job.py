"""Stage a private CPU-only Kaggle job for the pinned Bonsai 2 PTQ1_0 pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

MODEL_ID = "prism-ml/Ternary-Bonsai-2-27B-gguf"
REVISION = "6ed5e12bf84b7a63069882c91dd9e9218647d17b"
MODEL_FILE = "Ternary-Bonsai-2-27B-PTQ1_0.gguf"
MODEL_SIZE = 5_946_648_928
MODEL_SHA256 = "53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3"
SUPPORT_FILES = ("README.md", "LICENSE", "NOTICE.txt")
SUPPORT_LIMIT = 1024 * 1024
METADATA_LIMIT = 2 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024
HF_API = "https://huggingface.co/api/models/{model}/revision/{revision}?blobs=true"
HF_RESOLVE = "https://huggingface.co/{model}/resolve/{revision}/{name}?download=true"


def safe_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in (".", ".."):
        raise ValueError("unsafe source filename")
    return name


def publish_no_clobber(partial: Path, destination: Path) -> None:
    """Atomically publish a regular file without replacing an existing path."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    try:
        os.link(partial, destination)
    except FileExistsError:
        raise FileExistsError(destination) from None
    partial.unlink()


def stream_verified(
    source,
    destination: Path,
    expected_size: int,
    expected_sha256: str | None,
    *,
    chunk_size: int = CHUNK_SIZE,
    maximum_size: int | None = None,
) -> dict:
    """Copy bounded bytes and publish only after exact size/digest verification."""
    if type(chunk_size) is not int or not 0 < chunk_size <= CHUNK_SIZE:
        raise ValueError("chunk size must be positive and bounded")
    if (
        type(expected_size) is not int
        or expected_size < 0
        or (maximum_size is not None and expected_size > maximum_size)
    ):
        raise ValueError("expected size exceeds limit")
    if expected_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise ValueError("invalid expected SHA256")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(partial)
    digest, size = hashlib.sha256(), 0
    created = False
    try:
        with partial.open("xb") as output:
            created = True
            while True:
                data = source.read(chunk_size)
                if not data:
                    break
                if len(data) > chunk_size:
                    raise ValueError("source exceeded bounded read size")
                size += len(data)
                if size > expected_size or (
                    maximum_size is not None and size > maximum_size
                ):
                    raise ValueError("source exceeds expected size")
                digest.update(data)
                output.write(data)
        actual = digest.hexdigest()
        if size != expected_size:
            raise ValueError("source size mismatch")
        if expected_sha256 is not None and actual != expected_sha256:
            raise ValueError("source SHA256 mismatch")
        publish_no_clobber(partial, destination)
        return {"size": size, "sha256": actual}
    except BaseException:
        if created:
            partial.unlink(missing_ok=True)
        raise


def read_bounded(source, limit: int) -> bytes:
    data = source.read(limit + 1)
    if len(data) > limit:
        raise ValueError("metadata exceeds byte limit")
    return data


def fetch_metadata(opener=None) -> dict[str, dict]:
    opener = opener or urllib.request.urlopen
    url = HF_API.format(model=MODEL_ID, revision=REVISION)
    request = urllib.request.Request(
        url,
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": "local-ai-kaggle-bonsai/2",
        },
    )
    with opener(request, timeout=120) as response:
        payload = json.loads(read_bounded(response, METADATA_LIMIT))
    if payload.get("id") != MODEL_ID or payload.get("sha") != REVISION:
        raise ValueError("upstream model/revision mismatch")
    siblings = {}
    for entry in payload.get("siblings", []):
        name = entry.get("rfilename")
        if name in siblings:
            raise ValueError("duplicate upstream file metadata")
        siblings[name] = entry
    wanted = (MODEL_FILE,) + SUPPORT_FILES
    if any(name not in siblings for name in wanted):
        raise ValueError("upstream metadata missing required file")
    result = {}
    for name in wanted:
        entry = siblings[name]
        lfs = entry.get("lfs") or {}
        size = lfs.get("size", entry.get("size"))
        sha = lfs.get("sha256") or entry.get("sha256")
        if (
            type(size) is not int
            or size < 0
            or (
                sha is not None
                and (not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha))
            )
        ):
            raise ValueError("invalid upstream file metadata")
        result[name] = {"size": size, "sha256": sha}
    if result[MODEL_FILE] != {"size": MODEL_SIZE, "sha256": MODEL_SHA256}:
        raise ValueError("pinned model metadata mismatch")
    for name in SUPPORT_FILES:
        if result[name]["size"] > SUPPORT_LIMIT:
            raise ValueError("support file exceeds byte limit")
    return result


def open_source(filename, opener=None):
    opener = opener or urllib.request.urlopen
    safe_name(filename)
    url = HF_RESOLVE.format(
        model=MODEL_ID, revision=REVISION, name=urllib.parse.quote(filename)
    )
    request = urllib.request.Request(
        url,
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": "local-ai-kaggle-bonsai/2",
        },
    )
    return opener(request, timeout=120)


def run_job(
    *,
    archive_name: str,
    source_sha256: dict[str, str],
    checks_run: list[str],
    working_root=Path("/kaggle/working"),
    temp_root=Path("/kaggle/temp"),
    opener=None,
) -> Path:
    """Download the pinned pack and publish one verified no-clobber archive."""
    working_root, temp_root = Path(working_root), Path(temp_root)
    working_root.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    output = working_root / safe_name(archive_name)
    partial_archive = output.with_name(output.name + ".partial")
    if (
        output.exists()
        or output.is_symlink()
        or partial_archive.exists()
        or partial_archive.is_symlink()
    ):
        raise FileExistsError("Bonsai output already exists")
    opener = opener or urllib.request.urlopen
    upstream = fetch_metadata(opener)
    created_archive = False
    try:
        with tempfile.TemporaryDirectory(
            prefix="bonsai-download-", dir=temp_root
        ) as temporary:
            work, records = Path(temporary), {}
            for name in (MODEL_FILE,) + SUPPORT_FILES:
                details = upstream[name]
                print(
                    "Downloading pinned file:",
                    name,
                    details["size"],
                    "bytes",
                    flush=True,
                )
                with open_source(name, opener) as response:
                    records[name] = stream_verified(
                        response,
                        work / safe_name(name),
                        details["size"],
                        details["sha256"],
                        maximum_size=None if name == MODEL_FILE else SUPPORT_LIMIT,
                    )
            manifest = {
                "schema_version": 1,
                "upstream": {
                    "model_id": MODEL_ID,
                    "revision": REVISION,
                    "model_file": MODEL_FILE,
                    "model_lfs_sha256": MODEL_SHA256,
                },
                "files": records,
                "job": {
                    "frozen_source_sha256": source_sha256,
                    "checks_run": checks_run,
                    "python": __import__("sys").version,
                    "platform": __import__("platform").platform(),
                    "gpu_required": False,
                    "dependencies_installed": [],
                },
            }
            (work / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            with (
                partial_archive.open("xb") as raw,
                tarfile.open(fileobj=raw, mode="w") as archive,
            ):
                created_archive = True
                for path in sorted(work.iterdir()):
                    if not path.is_file() or path.is_symlink():
                        raise RuntimeError("unexpected output path")
                    archive.add(path, arcname=path.name, recursive=False)
            publish_no_clobber(partial_archive, output)
    except BaseException:
        if created_archive:
            partial_archive.unlink(missing_ok=True)
        raise
    return output


def install_archive(archive_path: Path, destination: Path) -> dict:
    """Verify an allowlisted archive before creating a new model directory.

    All weight bytes are checked against the pinned upstream digest, not merely
    the archive's self-reported checksum. No existing destination is replaced.
    A filesystem error during final publication can leave a new partial directory;
    the manifest is published last and must be present before a caller uses it.
    """
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    names = {MODEL_FILE, *SUPPORT_FILES, "manifest.json"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:") as archive:
        members = {}
        for member in archive:
            if (
                member.name not in names
                or member.name in members
                or not member.isfile()
            ):
                raise ValueError("unexpected or duplicate archive member")
            limit = MODEL_SIZE if member.name == MODEL_FILE else METADATA_LIMIT
            if member.size < 0 or member.size > limit:
                raise ValueError("archive member exceeds byte limit")
            members[member.name] = member
        if set(members) != names:
            raise ValueError("archive member coverage mismatch")
        with archive.extractfile(members["manifest.json"]) as source:
            manifest_bytes = read_bounded(source, METADATA_LIMIT)
        manifest = json.loads(manifest_bytes)
        if manifest.get("schema_version") != 1 or manifest.get("upstream") != {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "model_file": MODEL_FILE,
            "model_lfs_sha256": MODEL_SHA256,
        }:
            raise ValueError("archive is not the pinned Bonsai model")
        records = manifest.get("files", {})
        if set(records) != names - {"manifest.json"}:
            raise ValueError("manifest file coverage mismatch")
        if records[MODEL_FILE] != {"size": MODEL_SIZE, "sha256": MODEL_SHA256}:
            raise ValueError("archive model identity mismatch")
        with tempfile.TemporaryDirectory(
            prefix=".bonsai-verify-", dir=destination.parent
        ) as temporary:
            verified = Path(temporary)
            for name in sorted(records):
                record = records[name]
                if (
                    set(record) != {"size", "sha256"}
                    or record["size"] != members[name].size
                ):
                    raise ValueError("archive member size mismatch")
                if not isinstance(record["sha256"], str) or not re.fullmatch(
                    r"[0-9a-f]{64}", record["sha256"]
                ):
                    raise ValueError("archive member digest missing/invalid")
                with archive.extractfile(members[name]) as source:
                    stream_verified(
                        source,
                        verified / name,
                        record["size"],
                        record["sha256"],
                        maximum_size=MODEL_SIZE
                        if name == MODEL_FILE
                        else SUPPORT_LIMIT,
                    )
            (verified / "manifest.json").write_bytes(manifest_bytes)
            destination.mkdir()
            for name in sorted(records) + ["manifest.json"]:
                publish_no_clobber(verified / name, destination / name)
    return manifest


JOB = r"""
import hashlib
import sys
import types
from pathlib import Path

for name, source in SOURCES.items():
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256[name]:
        raise RuntimeError("Frozen source hash mismatch: " + name)
module = types.ModuleType("kaggle_bonsai_job")
module.__file__ = "<frozen:kaggle_bonsai_job.py>"
sys.modules[module.__name__] = module
exec(compile(SOURCES["kaggle_bonsai_job.py"], module.__file__, "exec"), module.__dict__)
tests = types.ModuleType("test_kaggle_bonsai_job")
tests.__file__ = "<frozen:test_kaggle_bonsai_job.py>"
exec(compile(SOURCES["test_kaggle_bonsai_job.py"], tests.__file__, "exec"), tests.__dict__)
checks = tests.run_cloud_checks()
output = module.run_job(archive_name=ARCHIVE_NAME, source_sha256=SOURCE_SHA256,
                        checks_run=checks)
print("Verified pinned Bonsai archive:", output.name, module.MODEL_SIZE,
      module.MODEL_SHA256, flush=True)
"""


def stage_job(kernel: str, output: Path) -> Path:
    from kaggle_staging import stage_script

    return stage_script(
        kernel,
        output,
        {},
        JOB,
        ("kaggle_bonsai_job.py", "test_kaggle_bonsai_job.py"),
        gpu=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True, help="new private owner/slug")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(stage_job(args.kernel, args.output))


if __name__ == "__main__":
    main()
