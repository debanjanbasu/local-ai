"""Offline checks for pinned, dependency-free Bonsai download staging."""

import ast
import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import kaggle_bonsai_job as job


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class BoundedSource(Response):
    def __init__(self, data, maximum):
        super().__init__(data)
        self.maximum = maximum

    def read(self, size=-1):
        if size < 0 or size > self.maximum:
            raise AssertionError("unbounded read")
        return super().read(size)


def run_cloud_checks():
    """Small offline checks executed from the exact frozen cloud sources."""
    data = b"frozen downloader check"
    digest = hashlib.sha256(data).hexdigest()
    with tempfile.TemporaryDirectory() as root:
        target = Path(root) / "missing" / "check"
        target.parent.mkdir(parents=True)
        result = job.stream_verified(
            BoundedSource(data, 7),
            target,
            len(data),
            digest,
            chunk_size=7,
            maximum_size=len(data),
        )
        if target.read_bytes() != data or result["sha256"] != digest:
            raise AssertionError("frozen stream check failed")
        bad = Path(root) / "bad"
        try:
            job.stream_verified(io.BytesIO(data), bad, len(data), "0" * 64)
        except ValueError:
            pass
        else:
            raise AssertionError("frozen digest rejection failed")
        if bad.exists() or bad.with_name("bad.partial").exists():
            raise AssertionError("frozen cleanup check failed")
    return [
        "frozen_source_sha256",
        "offline_stream_success",
        "offline_digest_rejection",
    ]


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = b"bounded test bytes" * 100
        self.digest = hashlib.sha256(self.data).hexdigest()

    def tearDown(self):
        self.temp.cleanup()

    def test_stream_correct_bytes_with_bounded_reads(self):
        target = self.root / "model.gguf"
        result = job.stream_verified(
            BoundedSource(self.data, 17),
            target,
            len(self.data),
            self.digest,
            chunk_size=17,
        )
        self.assertEqual(target.read_bytes(), self.data)
        self.assertEqual(result, {"size": len(self.data), "sha256": self.digest})

    def test_wrong_hash_size_oversize_and_truncation_leave_no_output(self):
        cases = [
            (self.data, len(self.data), "0" * 64, None),
            (self.data, len(self.data) - 1, self.digest, None),
            (self.data[:-1], len(self.data), self.digest, None),
            (self.data, len(self.data), self.digest, len(self.data) - 1),
        ]
        for number, (data, size, digest, limit) in enumerate(cases):
            target = self.root / f"bad-{number}"
            with self.assertRaises(ValueError):
                job.stream_verified(
                    io.BytesIO(data),
                    target,
                    size,
                    digest,
                    chunk_size=19,
                    maximum_size=limit,
                )
            self.assertFalse(target.exists())
            self.assertFalse(Path(str(target) + ".partial").exists())

    def test_path_symlinks_existing_targets_and_partials_are_retained(self):
        for name in ("../x", "a/b", ".", "", "white space"):
            with self.assertRaises(ValueError):
                job.safe_name(name)
        keep = self.root / "keep"
        keep.write_bytes(b"keep")
        link = self.root / "link"
        link.symlink_to(keep)
        partial_target = self.root / "partial-target"
        Path(str(partial_target) + ".partial").write_bytes(b"old")
        for target in (keep, link, partial_target):
            with self.assertRaises(FileExistsError):
                job.stream_verified(
                    io.BytesIO(self.data), target, len(self.data), self.digest
                )
        self.assertEqual(keep.read_bytes(), b"keep")
        self.assertEqual(Path(str(partial_target) + ".partial").read_bytes(), b"old")

    def test_destination_race_does_not_overwrite_racer(self):
        target = self.root / "race"

        def race(partial, destination):
            Path(destination).write_bytes(b"racer")
            raise FileExistsError(destination)

        with (
            patch.object(job.os, "link", side_effect=race),
            self.assertRaises(FileExistsError),
        ):
            job.stream_verified(
                io.BytesIO(self.data), target, len(self.data), self.digest
            )
        self.assertEqual(target.read_bytes(), b"racer")
        self.assertFalse(Path(str(target) + ".partial").exists())

    def test_partial_creation_race_and_invalid_read_sizes_preserve_files(self):
        target = self.root / "race"
        partial = self.root / "race.partial"
        original_open = Path.open

        def race(path, mode="r", *args, **kwargs):
            if path == partial and mode == "xb":
                with original_open(path, "wb") as stream:
                    stream.write(b"other downloader")
            return original_open(path, mode, *args, **kwargs)

        with patch.object(Path, "open", race), self.assertRaises(FileExistsError):
            job.stream_verified(
                io.BytesIO(self.data), target, len(self.data), self.digest
            )
        self.assertEqual(partial.read_bytes(), b"other downloader")
        for size in [0, -1, job.CHUNK_SIZE + 1]:
            with self.assertRaises(ValueError):
                job.stream_verified(
                    io.BytesIO(self.data),
                    self.root / "invalid",
                    len(self.data),
                    self.digest,
                    chunk_size=size,
                )

    def metadata(self, files=None, revision=None):
        files = files or {
            job.MODEL_FILE: (job.MODEL_SIZE, job.MODEL_SHA256),
            "README.md": (3, hashlib.sha256(b"readme").hexdigest()),
            "LICENSE": (3, hashlib.sha256(b"license").hexdigest()),
            "NOTICE.txt": (3, hashlib.sha256(b"notice").hexdigest()),
        }
        return {
            "id": job.MODEL_ID,
            "sha": revision or job.REVISION,
            "siblings": [
                {"rfilename": n, "lfs": {"size": s, "sha256": h}}
                for n, (s, h) in files.items()
            ],
        }

    def test_metadata_validation_mismatch_and_limit(self):
        def fetch(payload):
            return job.fetch_metadata(
                lambda *a, **k: Response(json.dumps(payload).encode())
            )

        self.assertEqual(fetch(self.metadata())[job.MODEL_FILE]["size"], job.MODEL_SIZE)
        for payload in (
            self.metadata(revision="0" * 40),
            self.metadata({job.MODEL_FILE: (1, "0" * 64)}),
            self.metadata(
                {
                    job.MODEL_FILE: (job.MODEL_SIZE, job.MODEL_SHA256),
                    **{n: (job.SUPPORT_LIMIT + 1, "0" * 64) for n in job.SUPPORT_FILES},
                }
            ),
        ):
            with self.assertRaises(ValueError):
                fetch(payload)
        with self.assertRaises(ValueError):
            job.fetch_metadata(
                lambda *a, **k: Response(b" " * (job.METADATA_LIMIT + 1))
            )

    def test_git_support_metadata_does_not_need_lfs_hash_but_model_does(self):
        payload = self.metadata()
        for entry in payload["siblings"][1:]:
            entry["size"] = entry.pop("lfs")["size"]
            entry["blobId"] = "1" * 40
        opener = lambda *a, **k: Response(json.dumps(payload).encode())
        records = job.fetch_metadata(opener)
        self.assertIsNone(records["LICENSE"]["sha256"])
        self.assertEqual(records[job.MODEL_FILE]["sha256"], job.MODEL_SHA256)
        model = payload["siblings"][0]
        model["size"] = model.pop("lfs")["size"]
        with self.assertRaises(ValueError):
            job.fetch_metadata(opener)

    def test_staged_job_is_private_cpu_pinned_frozen_and_imports_authoritative_code(
        self,
    ):
        staged = job.stage_job("owner/bonsai-unique", self.root / "staged")
        metadata = json.loads((staged / "kernel-metadata.json").read_text())
        self.assertTrue(metadata["is_private"])
        self.assertFalse(metadata["enable_gpu"])
        self.assertTrue(metadata["enable_internet"])
        self.assertNotIn("machine_shape", metadata)
        script = (staged / "convert.py").read_text()
        values = {
            n.targets[0].id: ast.literal_eval(n.value)
            for n in ast.parse(script).body
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id.isupper()
        }
        self.assertEqual(
            set(values["SOURCES"]),
            {"kaggle_bonsai_job.py", "test_kaggle_bonsai_job.py"},
        )
        for name, source in values["SOURCES"].items():
            self.assertEqual(
                values["SOURCE_SHA256"][name],
                hashlib.sha256(source.encode()).hexdigest(),
            )
        self.assertIn("module.run_job(", script)
        self.assertNotIn("def stream_verified", job.JOB)
        self.assertIn("run_cloud_checks", script)
        compile(script, "convert.py", "exec")

    def test_actual_generated_bootstrap_mocked_network_and_missing_temp_root(self):
        staged = job.stage_job("owner/bonsai-fixture", self.root / "stage")
        script = (staged / "convert.py").read_text()
        tree = ast.parse(script)
        values = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.isupper()
        }
        source = values["SOURCES"]["kaggle_bonsai_job.py"]
        model = b"tiny-model"
        supports = {
            "README.md": b"readme",
            "LICENSE": b"license",
            "NOTICE.txt": b"notice",
        }
        source = source.replace(
            f"MODEL_SIZE = {job.MODEL_SIZE:_}", f"MODEL_SIZE = {len(model)}"
        )
        source = source.replace(job.MODEL_SHA256, hashlib.sha256(model).hexdigest())
        values["SOURCES"]["kaggle_bonsai_job.py"] = source
        values["SOURCE_SHA256"]["kaggle_bonsai_job.py"] = hashlib.sha256(
            source.encode()
        ).hexdigest()
        metadata = {"id": job.MODEL_ID, "sha": job.REVISION, "siblings": []}
        payloads = {job.MODEL_FILE: model, **supports}
        for name, data in payloads.items():
            entry = {"rfilename": name, "size": len(data)}
            if name == job.MODEL_FILE:
                entry["lfs"] = {
                    "size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            else:
                entry["blobId"] = "2" * 40
            metadata["siblings"].append(entry)

        def opener(request, timeout=0):
            url = request.full_url
            if "/api/models/" in url:
                return Response(json.dumps(metadata).encode())
            name = url.split("/resolve/")[1].split("/", 1)[1].split("?", 1)[0]
            return Response(payloads[name])

        working = self.root / "missing" / "working"
        temp_root = self.root / "missing" / "temp"
        marker = "\nimport hashlib\nimport sys\nimport types\n"
        body = compile(script[script.index(marker) + 1 :], "convert.py", "exec")
        with (
            patch("urllib.request.urlopen", side_effect=opener),
            patch.dict(sys.modules),
        ):
            # Redirect only the cloud defaults in frozen source.
            values["SOURCES"]["kaggle_bonsai_job.py"] = (
                values["SOURCES"]["kaggle_bonsai_job.py"]
                .replace('Path("/kaggle/working")', f"Path({str(working)!r})")
                .replace('Path("/kaggle/temp")', f"Path({str(temp_root)!r})")
            )
            values["SOURCE_SHA256"]["kaggle_bonsai_job.py"] = hashlib.sha256(
                values["SOURCES"]["kaggle_bonsai_job.py"].encode()
            ).hexdigest()
            exec(body, values)  # noqa: S102 - execute our own frozen, hash-checked bootstrap
        archive = working / values["ARCHIVE_NAME"]
        self.assertTrue(archive.is_file())
        self.assertTrue(temp_root.is_dir())
        with tarfile.open(archive) as tar:
            manifest = json.load(tar.extractfile("manifest.json"))
        self.assertEqual(
            manifest["job"]["checks_run"],
            [
                "frozen_source_sha256",
                "offline_stream_success",
                "offline_digest_rejection",
            ],
        )
        self.assertEqual(
            manifest["files"][job.MODEL_FILE]["sha256"],
            hashlib.sha256(model).hexdigest(),
        )

    def make_archive(self, path, model, *, extra=None, corrupt=False):
        files = {
            job.MODEL_FILE: model,
            **{name: name.encode() for name in job.SUPPORT_FILES},
        }
        manifest = {
            "schema_version": 1,
            "upstream": {
                "model_id": job.MODEL_ID,
                "revision": job.REVISION,
                "model_file": job.MODEL_FILE,
                "model_lfs_sha256": job.MODEL_SHA256,
            },
            "files": {
                name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                for name, data in files.items()
            },
        }
        files["manifest.json"] = json.dumps(manifest).encode()
        if corrupt:
            files[job.MODEL_FILE] = b"x" * len(model)
        with tarfile.open(path, "w") as archive:
            for name, data in files.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
            if extra:
                archive.addfile(extra, io.BytesIO(b""))

    def test_archive_installs_only_verified_pinned_files_and_never_overwrites(self):
        model = b"verified model fixture"
        path = self.root / "archive.tar"
        with (
            patch.object(job, "MODEL_SIZE", len(model)),
            patch.object(job, "MODEL_SHA256", hashlib.sha256(model).hexdigest()),
        ):
            self.make_archive(path, model)
            destination = self.root / "models" / "bonsai"
            manifest = job.install_archive(path, destination)
            self.assertEqual((destination / job.MODEL_FILE).read_bytes(), model)
            self.assertEqual(
                json.loads((destination / "manifest.json").read_text()), manifest
            )
            with self.assertRaises(FileExistsError):
                job.install_archive(path, destination)
            self.assertEqual((destination / job.MODEL_FILE).read_bytes(), model)

    def test_archive_rejects_traversal_links_duplicates_and_corrupt_model(self):
        model = b"verified model fixture"
        extras = [tarfile.TarInfo("../escape"), tarfile.TarInfo("manifest.json")]
        link = tarfile.TarInfo("LICENSE")
        link.type = tarfile.SYMTYPE
        link.linkname = "/tmp/escape"
        extras.append(link)
        with (
            patch.object(job, "MODEL_SIZE", len(model)),
            patch.object(job, "MODEL_SHA256", hashlib.sha256(model).hexdigest()),
        ):
            for number, extra in enumerate([*extras, None]):
                path = self.root / f"bad-{number}.tar"
                self.make_archive(path, model, extra=extra, corrupt=extra is None)
                destination = self.root / f"bad-model-{number}"
                with self.assertRaises(ValueError):
                    job.install_archive(path, destination)
                self.assertFalse(destination.exists())
        self.assertFalse((self.root / "escape").exists())


if __name__ == "__main__":
    unittest.main()
