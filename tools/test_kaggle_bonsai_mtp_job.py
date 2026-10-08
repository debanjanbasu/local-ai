"""Offline checks for the pinned Bonsai 2 MTP head download staging."""

import ast
import hashlib
import io
import json
import re
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import kaggle_bonsai_job as shared
import kaggle_bonsai_mtp_job as head

TINY_HEAD = b"tiny-mtp-head-bytes"
SUPPORTS = {name: name.encode() for name in head.PIN["SUPPORT_FILES"]}

ENGINE_SOURCES = Path(__file__).resolve().parents[1] / "local-engine/src"
MTP_HEAD_SYMBOL = "DEFAULT_BONSAI_MTP_ARTIFACT"
STRING_LITERAL = re.compile(r"#*\"(?P<path>[^\"\n]*)\"#*")
# `= OTHER_CONST;` delegates the value; follow a couple of hops rather than a whole file.
MAX_HOPS = 3


def declaration_of(name):
    """`pub const NAME: &str = ...`, but tolerant: any visibility, any type, any spacing.

    A `::`-qualified expression (`Path::new(..)`) never counts as a declaration.
    """
    return re.compile(
        rf"(?<![:A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])\s*:(?!:)"
    )


def rust_sources():
    """Every engine source file as `{path relative to local-engine/src: text}`."""
    return {
        path.relative_to(ENGINE_SOURCES).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(ENGINE_SOURCES.rglob("*.rs"))
    }


def initialiser(text, start):
    """The expression assigned after the declaration whose name begins at `start`.

    Parenthesis-aware, so a `concat!` or `env!` spread over several lines is captured whole
    rather than truncated at the first newline — which is exactly where rustfmt puts it.
    """
    end = len(text)
    for index in range(start, len(text)):
        if text[index] in ";\n":
            end = index
            break
    equals = text.find("=", start, end)
    if equals == -1:
        return ""
    depth = 0
    for index in range(equals + 1, len(text)):
        char = text[index]
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif depth <= 0 and char in ";\n":
            return text[equals + 1 : index]
    return text[equals + 1 :]


def comparable_directory(literal):
    """Directory a Rust path literal resolves to, `..` segments collapsed away.

    Accepts a whole head path or a bare directory (`= SOME_DIR;`), told apart by whether
    the last segment looks like a file name. A `#[cfg(test)]` build legitimately reaches
    the same head through the workspace root (`env!("CARGO_MANIFEST_DIR")/../models/...`),
    so anchoring segments are removed. A literal that is absolute *without* anchoring stays
    absolute: it resolves somewhere the stager never writes, so it is not agreement.
    """
    normalized = literal.replace("\\", "/").strip()
    segments = [part for part in normalized.split("/") if part not in ("", ".")]
    anchored = len(segments) > 1 and segments[0] == ".."
    resolved = []
    for part in segments:
        if part == ".." and resolved:
            resolved.pop()
        elif part != "..":
            resolved.append(part)
    directory = "/".join(resolved[:-1] if "." in resolved[-1] else resolved) if resolved else ""
    if normalized.startswith("/") and not anchored and directory:
        return "/" + directory
    return directory


def mtp_head_directories(sources):
    """Every directory the engine's `DEFAULT_BONSAI_MTP_ARTIFACT` bindings resolve to.

    Read off the constant's own initialisers — literals concatenated in binding order, as
    `concat!` does — following `= OTHER_CONST;` delegations. Deliberately never scans a
    file for stray path literals: `bonsai_mtp.rs` keeps unrelated `../models/...` test
    fixtures, and a whole-file scan would let one of those vouch for the default.
    """
    directories = set()
    for text in sources.values():
        pending = [(text, MTP_HEAD_SYMBOL)]
        for _ in range(MAX_HOPS):
            following = []
            for source, name in pending:
                for match in declaration_of(name).finditer(source):
                    expression = initialiser(source, match.start())
                    if not expression:
                        continue
                    literal = "".join(
                        found.group("path") for found in STRING_LITERAL.finditer(expression)
                    )
                    directory = comparable_directory(literal) if literal else ""
                    if directory:
                        directories.add(directory)
                    elif not literal:
                        # A bare `= OTHER_CONST;` delegates the value; follow it. A value
                        # with no directory part makes no claim about *which* directory.
                        following += [
                            (source, word)
                            for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression)
                        ]
            pending = [(source, word) for source, word in following if word != MTP_HEAD_SYMBOL]
            if not pending:
                break
    return directories


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def tiny_pin():
    """The real pin with only the head's size/digest shrunk to a test payload."""
    return {
        **head.PIN,
        "MODEL_SIZE": len(TINY_HEAD),
        "MODEL_SHA256": hashlib.sha256(TINY_HEAD).hexdigest(),
    }


def fake_opener(pin):
    metadata = {"id": pin["MODEL_ID"], "sha": pin["REVISION"], "siblings": []}
    payloads = {pin["MODEL_FILE"]: TINY_HEAD, **SUPPORTS}
    for name, data in payloads.items():
        entry = {"rfilename": name, "size": len(data)}
        if name == pin["MODEL_FILE"]:
            entry["lfs"] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        else:
            entry["blobId"] = "2" * 40
        metadata["siblings"].append(entry)

    def opener(request, timeout=0):
        url = request.full_url
        if "/api/models/" in url:
            assert pin["MODEL_ID"] in url and pin["REVISION"] in url, url
            return Response(json.dumps(metadata).encode())
        assert f"/{pin['MODEL_ID']}/resolve/{pin['REVISION']}/" in url, url
        name = url.split("/resolve/")[1].split("/", 1)[1].split("?", 1)[0]
        return Response(payloads[name])

    return opener


class Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_pin_matches_upstream_teacher_and_stays_out_of_the_runtime_directory(self):
        self.assertEqual(head.PIN["MODEL_FILE"], "model_mtp.safetensors")
        self.assertEqual(head.PIN["MODEL_SIZE"], 849_400_392)
        self.assertEqual(
            head.PIN["MODEL_SHA256"],
            "7a4a18b2d02116ef184d1b0ee4af46d829825ff2c042f79cf37ef8a03c399218",
        )
        head_path = head.DEFAULT_DESTINATION / head.PIN["MODEL_FILE"]
        self.assertEqual(
            head_path, Path("models/bonsai2-27b-mtp-teacher/model_mtp.safetensors")
        )
        # What this protects: the BF16 teacher is training input, not a runtime head,
        # so it must not be published into the directory the engine loads its ternary
        # artifact from (the no-clobber installer would refuse it there anyway). The
        # check is deliberately location- and formatting-agnostic: locate the constant
        # by SYMBOL anywhere under local-engine/src, then compare the directory it
        # resolves to with DEFAULT_DESTINATION. Reformatting, cfg-splitting, renaming
        # the constant or moving it to another module must never fail a Kaggle staging
        # run. Do not tighten this into a literal grep of one source file.
        sources = rust_sources()
        holders = sorted(name for name, text in sources.items() if MTP_HEAD_SYMBOL in text)
        self.assertTrue(
            holders,
            f"{MTP_HEAD_SYMBOL} is declared nowhere under {ENGINE_SOURCES}",
        )
        declared = mtp_head_directories(sources)
        self.assertEqual(declared, {"models/bonsai2-27b-mtp"})
        self.assertNotIn(
            head.DEFAULT_DESTINATION.as_posix(),
            declared,
            msg=f"the teacher destination {head.DEFAULT_DESTINATION.as_posix()!r} is "
            "the engine's runtime head directory",
        )

    def test_binding_is_isolated_from_the_shared_downloader(self):
        module = head.pinned_downloader()
        self.assertEqual(module.MODEL_ID, head.PIN["MODEL_ID"])
        self.assertEqual(module.SUPPORT_FILES, head.PIN["SUPPORT_FILES"])
        self.assertIsNot(module, shared)
        # The target checkpoint's downloader keeps its own identity.
        self.assertEqual(shared.MODEL_ID, "prism-ml/Ternary-Bonsai-2-27B-gguf")
        self.assertEqual(shared.MODEL_FILE, "Ternary-Bonsai-2-27B-PTQ1_0.gguf")
        self.assertNotEqual(shared.MODEL_SHA256, module.MODEL_SHA256)

    def test_bind_rejects_malformed_pins(self):
        for broken in [
            {**head.PIN, "MODEL_SIZE": 0},
            {**head.PIN, "MODEL_SIZE": "849400392"},
            {**head.PIN, "MODEL_SHA256": head.PIN["MODEL_SHA256"][:-1]},
            {**head.PIN, "MODEL_SHA256": head.PIN["MODEL_SHA256"].upper()},
            {**head.PIN, "REVISION": "main"},
            {**head.PIN, "MODEL_FILE": "../model_mtp.safetensors"},
            {**head.PIN, "SUPPORT_FILES": ("README.md", "README.md")},
            {**head.PIN, "SUPPORT_FILES": ("README.md", "model_mtp.safetensors")},
            {key: value for key, value in head.PIN.items() if key != "REVISION"},
            {**head.PIN, "EXTRA": 1},
        ]:
            with self.assertRaises(ValueError, msg=broken):
                head.pinned_downloader(broken)

    def test_download_then_install_publishes_the_default_head_layout(self):
        pin = tiny_pin()
        module = head.pinned_downloader(pin)
        working, temp_root = self.root / "working", self.root / "temp"
        archive = module.run_job(
            archive_name="head.tar",
            source_sha256={"kaggle_bonsai_mtp_job.py": "0" * 64},
            checks_run=["offline"],
            working_root=working,
            temp_root=temp_root,
            opener=fake_opener(pin),
        )
        with tarfile.open(archive) as tar:
            names = set(tar.getnames())
            manifest = json.load(tar.extractfile("manifest.json"))
        self.assertEqual(names, {pin["MODEL_FILE"], *pin["SUPPORT_FILES"], "manifest.json"})
        self.assertEqual(manifest["upstream"]["model_id"], head.PIN["MODEL_ID"])
        self.assertEqual(manifest["upstream"]["revision"], head.PIN["REVISION"])
        self.assertFalse(manifest["job"]["gpu_required"])

        destination = self.root / "models" / "bonsai2-27b-mtp-teacher"
        installed = module.install_archive(archive, destination)
        self.assertEqual(installed["upstream"], manifest["upstream"])
        self.assertEqual((destination / "model_mtp.safetensors").read_bytes(), TINY_HEAD)
        self.assertEqual((destination / "NOTICE").read_bytes(), b"NOTICE")
        self.assertEqual((destination / "mtp_config.json").read_bytes(), b"mtp_config.json")
        # A published head is never replaced.
        with self.assertRaises(FileExistsError):
            module.install_archive(archive, destination)
        # The real pin rejects a head whose bytes do not match the upstream digest.
        with self.assertRaises(ValueError):
            head.install_head(archive, self.root / "models" / "other")
        self.assertFalse((self.root / "models" / "other").exists())

    def test_install_rejects_the_target_checkpoint_archive(self):
        model = b"not-a-head"
        work = self.root / "work"
        work.mkdir()
        records = {}
        files = {shared.MODEL_FILE: model, **{n: n.encode() for n in shared.SUPPORT_FILES}}
        for name, data in files.items():
            (work / name).write_bytes(data)
            records[name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        manifest = {
            "schema_version": 1,
            "upstream": {
                "model_id": shared.MODEL_ID,
                "revision": shared.REVISION,
                "model_file": shared.MODEL_FILE,
                "model_lfs_sha256": shared.MODEL_SHA256,
            },
            "files": records,
        }
        (work / "manifest.json").write_text(json.dumps(manifest))
        archive = self.root / "target.tar"
        with tarfile.open(archive, "w") as tar:
            for path in sorted(work.iterdir()):
                tar.add(path, arcname=path.name)
        with self.assertRaises(ValueError):
            head.install_head(archive, self.root / "models" / "bonsai2-27b-mtp-teacher")

    def test_staged_job_freezes_head_sources_and_binds_before_download(self):
        staged = head.stage_job("owner/bonsai-mtp-head", self.root / "staged")
        metadata = json.loads((staged / "kernel-metadata.json").read_text())
        self.assertFalse(metadata["enable_gpu"])
        self.assertTrue(metadata["is_private"])
        self.assertNotIn("machine_shape", metadata)
        script = (staged / "convert.py").read_text(encoding="utf-8")
        values = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in ast.parse(script).body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.isupper()
        }
        self.assertEqual(set(values["SOURCES"]), set(head.SOURCE_FILES))
        for name, source in values["SOURCES"].items():
            self.assertEqual(
                values["SOURCE_SHA256"][name], hashlib.sha256(source.encode()).hexdigest()
            )
            self.assertEqual(
                source, (Path(head.__file__).with_name(name)).read_text(encoding="utf-8")
            )
        self.assertEqual(values["ARCHIVE_NAME"], "bonsai-mtp-head.tar")

        # Run the frozen bootstrap with the network step replaced by a recorder.
        marker = "\nimport hashlib\nimport sys\nimport types\n"
        body = script[script.index(marker) + 1 :]
        self.assertIn("output = module.run_job(", body)
        body = body.replace(
            "output = module.run_job(",
            "output = RECORD(",
        )
        seen = {}

        def record(**kwargs):
            seen.update(kwargs)
            seen["module"] = sys.modules["kaggle_bonsai_job"]
            return Path("recorded.tar")

        values["RECORD"] = record
        with patch.dict(sys.modules):
            exec(compile(body, "convert.py", "exec"), values)  # noqa: S102 - our own frozen bootstrap
        self.assertEqual(seen["archive_name"], "bonsai-mtp-head.tar")
        self.assertEqual(seen["checks_run"][0], "frozen_source_sha256")
        # The frozen downloader was rebound to the head before the download.
        self.assertEqual(seen["module"].MODEL_ID, head.PIN["MODEL_ID"])
        self.assertEqual(seen["module"].MODEL_SHA256, head.PIN["MODEL_SHA256"])
        self.assertEqual(seen["module"].SUPPORT_FILES, head.PIN["SUPPORT_FILES"])
        self.assertEqual(shared.MODEL_ID, "prism-ml/Ternary-Bonsai-2-27B-gguf")

    def test_tampered_frozen_source_is_refused(self):
        staged = head.stage_job("owner/bonsai-mtp-head", self.root / "staged")
        script = (staged / "convert.py").read_text(encoding="utf-8")
        marker = "\nimport hashlib\nimport sys\nimport types\n"
        values = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in ast.parse(script).body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.isupper()
        }
        values["SOURCES"]["kaggle_bonsai_mtp_job.py"] += "\n# tampered\n"
        with patch.dict(sys.modules), self.assertRaises(RuntimeError):
            exec(compile(script[script.index(marker) + 1 :], "convert.py", "exec"), values)  # noqa: S102

    def test_stage_requires_output_and_install_uses_default_destination(self):
        with patch("sys.argv", ["prog", "--kernel", "owner/slug"]), self.assertRaises(SystemExit):
            head.main()
        with (
            patch("sys.argv", ["prog", "--install", "archive.tar"]),
            patch.object(head, "install_head", return_value={"upstream": {"ok": 1}}) as install,
        ):
            head.main()
        install.assert_called_once_with(Path("archive.tar"), head.DEFAULT_DESTINATION)


if __name__ == "__main__":
    unittest.main()
