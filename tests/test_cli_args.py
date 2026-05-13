import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run import parse_args


class CliArgsTest(unittest.TestCase):
    def test_timecode_output_is_enabled_by_default(self):
        with patch("sys.argv", ["run.py", "audio/example.m4a"]):
            args = parse_args()

        self.assertTrue(args.include_timecode)

    def test_timecode_argument_disables_timecode_output(self):
        with patch("sys.argv", ["run.py", "audio/example.m4a", "--timecode"]):
            args = parse_args()

        self.assertEqual(args.audio, Path("audio/example.m4a"))
        self.assertFalse(args.include_timecode)


if __name__ == "__main__":
    unittest.main()
