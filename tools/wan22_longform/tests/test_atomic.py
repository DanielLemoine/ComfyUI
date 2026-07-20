from __future__ import annotations

import tempfile
import threading
import unittest
import sys
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

import wan22_longform.atomic as atomic_module
from wan22_longform.atomic import copy_file, write_text


class AtomicArtifactTests(unittest.TestCase):
    def test_concurrent_new_text_publish_claims_exactly_one_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "evidence" / "history.json"
            barrier = threading.Barrier(2)
            results: list[object] = []
            original_link = atomic_module.os.link

            def synchronized_link(source: Path, target: Path) -> None:
                if Path(target) == destination:
                    barrier.wait(timeout=5)
                original_link(source, target)

            def publish(content: str) -> None:
                try:
                    write_text(destination, content)
                    results.append("success")
                except OSError as error:
                    results.append(error)

            with patch("wan22_longform.atomic.os.link", synchronized_link):
                contenders = [
                    threading.Thread(target=publish, args=("first\n",)),
                    threading.Thread(target=publish, args=("second\n",)),
                ]
                for contender in contenders:
                    contender.start()
                for contender in contenders:
                    contender.join(timeout=5)

            self.assertFalse(any(contender.is_alive() for contender in contenders))
            self.assertEqual(results.count("success"), 1)
            self.assertEqual(sum(isinstance(result, FileExistsError) for result in results), 1)
            self.assertIn(destination.read_text(encoding="utf-8"), {"first\n", "second\n"})

    def test_failed_text_publish_leaves_no_final_or_staging_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "evidence" / "history.json"

            with patch(
                "wan22_longform.atomic.os.link",
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
