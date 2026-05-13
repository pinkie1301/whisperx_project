from __future__ import annotations

from collections.abc import Iterable, Mapping


SPEAKER_NAMES = tuple("ABCDEF")
UNKNOWN_SPEAKER = "Unknown"


def format_speaker_lines(
    segments: Iterable[Mapping[str, object]],
    *,
    include_timecode: bool = True,
) -> list[str]:
    """Return transcript lines labelled by speaker, one line per segment.

    Speaker IDs are mapped to A-F by first appearance so labels remain stable
    for a single transcript even when pyannote returns non-sequential IDs.
    """
    speaker_map: dict[str, str] = {}
    lines: list[str] = []

    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue

        speaker_id = segment.get("speaker")
        speaker = _display_name_for_speaker(speaker_id, speaker_map)
        speaker_text = f"{speaker}: {text}"
        if include_timecode:
            speaker_text = f"{_format_timecode(segment.get('start'))} {speaker_text}"
        lines.append(speaker_text)

    return lines


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


def _format_timecode(start: object) -> str:
    seconds = float(start or 0)
    milliseconds = int(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"[{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}]"

