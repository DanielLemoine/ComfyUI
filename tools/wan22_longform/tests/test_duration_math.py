from __future__ import annotations

import sys
import unittest
from fractions import Fraction
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from wan22_longform.assembly import AssemblyError, duration_seconds  # noqa: E402


class DurationMathTests(unittest.TestCase):
    def test_duration_is_exact_fraction_of_frame_count_and_fps(self) -> None:
        self.assertEqual(duration_seconds(81, Fraction(16, 1)), Fraction(81, 16))

    def test_duration_rejects_non_positive_inputs(self) -> None:
        with self.assertRaisesRegex(AssemblyError, "frame_count"):
            duration_seconds(0, Fraction(16, 1))
        with self.assertRaisesRegex(AssemblyError, "fps"):
            duration_seconds(81, Fraction(0, 1))


if __name__ == "__main__":
    unittest.main()
