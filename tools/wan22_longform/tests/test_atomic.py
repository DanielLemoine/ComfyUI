from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


from wan22_longform.atomic import copy_file, write_text


class AtomicArtifactTests(unittest.TestCase):
    def test_failed_text_publish_leaves_no_final_or_staging_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "evidence" / "history.json"

            with patch(
                "wan22_longform.atomic.os.replace",
                side_effect=OSError("fixture publish interruption"),
            ):
                with self.assertRaisesRegex(OSError, "publish interruption"):
                    write_text(destination, '{"complete": true}\n')

            self.assertFalse(destination.exists())
            self.assertFalse(list(destination.parent.glob(".history.json.*.staging")))

    def test_failed_copy_preserves_an_existing_final_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            destination = root / "outputs" / "segment.mp4"
            source.write_bytes(b"complete-video")
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"old-complete-video")

            def partial_copy(_source, target) -> None:
                target.write(b"partial-video")
                raise OSError("fixture copy interruption")

            with patch(
                "wan22_longform.atomic.shutil.copyfileobj", side_effect=partial_copy
            ):
                with self.assertRaisesRegex(OSError, "copy interruption"):
                    copy_file(source, destination, replace_existing=True)

            self.assertEqual(destination.read_bytes(), b"old-complete-video")
            self.assertFalse(list(destination.parent.glob(".segment.mp4.*.staging")))


if __name__ == "__main__":
    unittest.main()
