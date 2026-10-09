import hashlib
import io
import tempfile
import unittest
from pathlib import Path

import fetch_mtp_head


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.payload = b"head bytes" * 7
        self.pin = (fetch_mtp_head.SIZE, fetch_mtp_head.SHA256)
        fetch_mtp_head.SIZE = len(self.payload)
        fetch_mtp_head.SHA256 = hashlib.sha256(self.payload).hexdigest()

    def tearDown(self):
        fetch_mtp_head.SIZE, fetch_mtp_head.SHA256 = self.pin

    def opener(self, data):
        def open_(request, timeout):
            self.assertEqual(request.full_url, fetch_mtp_head.URL)
            return FakeResponse(data)

        return open_

    def test_installs_verified_bytes_without_clobbering(self):
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "models" / "head.bin"
            record = fetch_mtp_head.fetch(destination, self.opener(self.payload))
            self.assertEqual(destination.read_bytes(), self.payload)
            self.assertEqual(record["sha256"], fetch_mtp_head.SHA256)
            with self.assertRaises(FileExistsError):
                fetch_mtp_head.fetch(destination, self.opener(self.payload))

    def test_rejects_wrong_bytes_and_leaves_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "head.bin"
            corrupt = b"x" + self.payload[1:]
            with self.assertRaises(ValueError):
                fetch_mtp_head.fetch(destination, self.opener(corrupt))
            self.assertEqual(list(Path(root).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
