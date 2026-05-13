# WhisperX 中文講者逐字稿

這個資料夾提供一個本機腳本，將錄音轉成中文逐字稿，並用 6 位講者標籤輸出成 `.txt`。

## 環境

目前 WhisperX 需要 Python `>=3.10, <3.14`。你的系統 `python3` 是 3.9.6，所以請先安裝 Python 3.11 或 3.12。

macOS + 16GB RAM 建議設定：

- ASR model: `medium`
- device: `cpu`
- compute type: `int8`
- speakers: `6`
- output labels: `A` 到 `F`

## 安裝

```bash
cd /Users/pinkie/whisperx_project
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果你的機器沒有 `python3.11`，可先用 Homebrew 安裝：

```bash
brew install python@3.11
```

## Hugging Face token

先到 Hugging Face 接受 `pyannote/speaker-diarization-community-1` 的模型條款，建立 read token，然後設定：

```bash
export HF_TOKEN="你的 HuggingFace read token"
```

## 執行

```bash
source .venv/bin/activate
python -m scripts.run recording.m4a
```

未指定輸出路徑時，會輸出到原始音檔同名 `.txt`：

```txt
recording.txt
```

也可以用 argument 指定輸出路徑：

```bash
python -m scripts.run recording.m4a -o outputs/recording_speakers.txt
```

格式範例：

```txt
A: 今天天氣真好。
B: 沒錯，今天是晴天。
C: 我們先整理重點。
```

若要改成更高品質但更慢的模型：

```bash
python -m scripts.run recording.m4a --model large-v3
```

腳本會用 WhisperX 預先載入的 16kHz mono waveform 交給 pyannote，避開 macOS 上 `torchcodec` 找不到 FFmpeg dylib 時的音訊解碼問題。

## 測試

格式化邏輯不需要 WhisperX 也能測：

```bash
python3 -m unittest tests/test_transcript_formatter.py
```
