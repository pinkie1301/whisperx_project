import unittest

from scripts.formatter import format_speaker_lines


class TranscriptFormatterTest(unittest.TestCase):
    def test_includes_timecode_at_start_of_each_line_by_default(self):
        segments = [
            {"speaker": "SPEAKER_00", "start": 65.4321, "text": "first segment"},
            {"speaker": "SPEAKER_00", "start": 3661.005, "text": "second segment"},
        ]

        self.assertEqual(
            format_speaker_lines(segments),
            [
                "[00:01:05.432] A: first segment",
                "[01:01:01.005] A: second segment",
            ],
        )

    def test_can_disable_timecode_prefix(self):
        segments = [
            {"speaker": "SPEAKER_00", "start": 65.4321, "text": "first segment"},
        ]

        self.assertEqual(
            format_speaker_lines(segments, include_timecode=False),
            ["A: first segment"],
        )

    def test_maps_six_speakers_by_first_appearance_without_merging_consecutive_segments(self):
        segments = [
            {"speaker": "SPEAKER_04", "text": "今天天氣真好。"},
            {"speaker": "SPEAKER_04", "text": "適合討論報告。"},
            {"speaker": "SPEAKER_02", "text": "沒錯，今天是晴天。"},
            {"speaker": "SPEAKER_01", "text": "我們先整理重點。"},
            {"speaker": "SPEAKER_05", "text": "我負責簡報。"},
            {"speaker": "SPEAKER_03", "text": "我負責文字。"},
            {"speaker": "SPEAKER_00", "text": "最後一起檢查。"},
        ]

        self.assertEqual(
            format_speaker_lines(segments, include_timecode=False),
            [
                "A: 今天天氣真好。",
                "A: 適合討論報告。",
                "B: 沒錯，今天是晴天。",
                "C: 我們先整理重點。",
                "D: 我負責簡報。",
                "E: 我負責文字。",
                "F: 最後一起檢查。",
            ],
        )

    def test_marks_missing_speaker_as_unknown_without_losing_text(self):
        segments = [
            {"text": "這段沒有講者標籤。"},
            {"speaker": "SPEAKER_00", "text": "下一段有講者。"},
        ]

        self.assertEqual(
            format_speaker_lines(segments, include_timecode=False),
            [
                "Unknown: 這段沒有講者標籤。",
                "A: 下一段有講者。",
            ],
        )


if __name__ == "__main__":
    unittest.main()
