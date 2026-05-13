from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
from typing import Optional

from .formatter import format_speaker_lines


DEFAULT_MODEL = "medium"


def main() -> None:
    args = parse_args()
    output_path = resolve_output_path(args.audio, args.output)
    output_path = transcribe_with_speakers(
        audio_path=args.audio,
        output_path=output_path,
        model_name=args.model,
        language=args.language,
        num_speakers=args.num_speakers,
        batch_size=args.batch_size,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN"),
    )
    print(f"Wrote transcript to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Chinese speaker-labelled .txt transcript with WhisperX."
    )
    parser.add_argument("audio", type=Path, help="Audio file path, e.g. recording.m4a")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .txt path. Default: same path as input audio with .txt suffix.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Whisper model. Use medium for 16GB Mac CPU, or large-v3 for higher quality.",
    )
    parser.add_argument("--language", default="zh", help="Language code. Default: zh")
    parser.add_argument("--num-speakers", type=int, default=6, help="Known speaker count.")
    parser.add_argument("--batch-size", type=int, default=4, help="CPU-safe batch size.")
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face read token. Defaults to HF_TOKEN environment variable.",
    )
    return parser.parse_args()


def resolve_output_path(audio_path: Path, output_path: Optional[Path]) -> Path:
    if output_path is not None:
        return output_path
    return Path("outputs") / audio_path.with_suffix(".txt").name


def transcribe_with_speakers(
    *,
    audio_path: Path,
    output_path: Path,
    model_name: str,
    language: str,
    num_speakers: int,
    batch_size: int,
    hf_token: Optional[str],
) -> Path:
    if hf_token is None:
        raise SystemExit("Missing Hugging Face token. Set HF_TOKEN or pass --hf-token.")

    _configure_runtime_environment()

    import pandas as pd
    import torch
    import whisperx
    from pyannote.audio import Pipeline

    device = "cpu"
    compute_type = "int8"
    audio = whisperx.load_audio(str(audio_path))

    model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        language=language,
    )
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    del model
    _release_memory(torch)

    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    del align_model
    _release_memory(torch)

    diarize_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=hf_token,
    )
    audio_data = {
        "waveform": torch.from_numpy(audio[None, :]),
        "sample_rate": 16000,
    }
    diarize_output = diarize_pipeline(audio_data, num_speakers=num_speakers)
    diarization = getattr(
        diarize_output,
        "exclusive_speaker_diarization",
        diarize_output.speaker_diarization,
    )
    diarize_segments = _diarization_to_dataframe(diarization, pd)
    result = whisperx.assign_word_speakers(
        diarize_segments,
        result,
        fill_nearest=True,
    )

    lines = format_speaker_lines(result["segments"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _release_memory(torch_module: object) -> None:
    gc.collect()
    if hasattr(torch_module, "mps") and torch_module.mps.is_available():
        torch_module.mps.empty_cache()


def _configure_runtime_environment() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def _diarization_to_dataframe(diarization: object, pandas_module: object) -> object:
    diarize_df = pandas_module.DataFrame(_diarization_to_records(diarization))
    return diarize_df


def _diarization_to_records(diarization: object) -> list[dict[str, object]]:
    return [
        {
            "segment": segment,
            "label": label,
            "speaker": speaker,
            "start": segment.start,
            "end": segment.end,
        }
        for segment, label, speaker in diarization.itertracks(yield_label=True)
    ]


if __name__ == "__main__":
    main()
