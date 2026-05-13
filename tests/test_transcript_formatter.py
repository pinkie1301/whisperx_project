import unittest

from scripts.formatter import format_speaker_lines


class TranscriptFormatterTest(unittest.TestCase):
    def test_maps_six_speakers_by_first_appearance_and_merges_consecutive_segments(self):
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
            format_speaker_lines(segments),
            [
                "A: 今天天氣真好。適合討論報告。",
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
            format_speaker_lines(segments),
            [
                "Unknown: 這段沒有講者標籤。",
                "A: 下一段有講者。",
            ],
        )


if __name__ == "__main__":
    unittest.main()
