import unittest

from scripts.run import _diarization_to_records


class FakeSegment:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class FakeDiarization:
    def itertracks(self, yield_label=False):
        self.assert_yield_label = yield_label
        return [
            (FakeSegment(0.0, 1.25), "track-1", "SPEAKER_00"),
            (FakeSegment(1.25, 2.5), "track-2", "SPEAKER_01"),
        ]


class DiarizationRecordsTest(unittest.TestCase):
    def test_converts_pyannote_diarization_to_whisperx_record_shape(self):
        diarization = FakeDiarization()

        records = _diarization_to_records(diarization)

        self.assertEqual(
            records,
            [
                {
                    "segment": records[0]["segment"],
                    "label": "track-1",
                    "speaker": "SPEAKER_00",
                    "start": 0.0,
                    "end": 1.25,
                },
                {
                    "segment": records[1]["segment"],
                    "label": "track-2",
                    "speaker": "SPEAKER_01",
                    "start": 1.25,
                    "end": 2.5,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
