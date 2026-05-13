from __future__ import annotations

from collections.abc import Iterable, Mapping


SPEAKER_NAMES = tuple("ABCDEF")
UNKNOWN_SPEAKER = "Unknown"


def format_speaker_lines(segments: Iterable[Mapping[str, object]]) -> list[str]:
    """Return transcript lines grouped by consecutive speaker.

    Speaker IDs are mapped to A-F by first appearance so labels remain stable
    for a single transcript even when pyannote returns non-sequential IDs.
    """
    speaker_map: dict[str, str] = {}
    lines: list[tuple[str, str]] = []

    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue

        speaker_id = segment.get("speaker")
        speaker = _display_name_for_speaker(speaker_id, speaker_map)

        if lines and lines[-1][0] == speaker:
            previous_speaker, previous_text = lines[-1]
            lines[-1] = (previous_speaker, f"{previous_text}{text}")
        else:
            lines.append((speaker, text))

    return [f"{speaker}: {text}" for speaker, text in lines]


def _display_name_for_speaker(speaker_id: object, speaker_map: dict[str, str]) -> str:
    if not speaker_id:
        return UNKNOWN_SPEAKER

    speaker_key = str(speaker_id)
    if speaker_key not in speaker_map:
        speaker_map[speaker_key] = (
            SPEAKER_NAMES[len(speaker_map)]
            if len(speaker_map) < len(SPEAKER_NAMES)
            else speaker_key
        )
    return speaker_map[speaker_key]

