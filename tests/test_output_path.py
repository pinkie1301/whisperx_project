import unittest
from pathlib import Path

from scripts.run import resolve_output_path


class OutputPathTest(unittest.TestCase):
    def test_uses_custom_output_when_provided(self):
        self.assertEqual(
            resolve_output_path(Path("recording_30min_part1.m4a"), Path("out/custom.txt")),
            Path("out/custom.txt"),
        )

    def test_defaults_to_outputs_folder_with_txt_suffix(self):
        self.assertEqual(
            resolve_output_path(Path("recording_30min_part1.m4a"), None),
            Path("outputs/recording_30min_part1.txt"),
        )

    def test_ignores_audio_parent_directory_for_default_output(self):
        self.assertEqual(
            resolve_output_path(Path("audio/recording_30min_part2.m4a"), None),
            Path("outputs/recording_30min_part2.txt"),
        )


if __name__ == "__main__":
    unittest.main()
