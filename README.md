# WhisperX 中文講者標記轉錄

這個專案使用 WhisperX 將音訊轉成中文文字稿，並用 pyannote diarization 產生講者標記。預設會輸出 6 位講者，標籤為 `A` 到 `F`，結果寫入 `outputs/`。

## 建議設定

- Python: `>=3.10, <3.14`
- macOS / CPU: `--device cpu --compute-type int8`
- Windows / CPU: `--device cpu --compute-type int8`
- Windows / NVIDIA CUDA: `--device cuda --compute-type float16`
- 預設 runtime: `--device auto --compute-type auto`
  - 偵測到 CUDA 時使用 `cuda` + `float16`
  - 否則使用 `cpu` + `int8`

## 依賴套件安裝

官方參考：

- WhisperX setup: <https://github.com/m-bain/whisperX>
- OpenAI Whisper setup: <https://github.com/openai/whisper#setup>
- PyTorch 安裝選擇器: <https://pytorch.org/get-started/locally/>

WhisperX 官方建議可直接用 PyPI 安裝：

```bash
python -m pip install whisperx
```

OpenAI Whisper setup 說明也要求系統安裝 `ffmpeg`。如果 `tiktoken` 沒有你平台的 prebuilt wheel，可能還需要 Rust toolchain。

本專案建議先依平台安裝 PyTorch，再安裝 `requirements.txt`。

## macOS CPU 安裝

```bash
cd /Users/pinkie/whisperx_project
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
python -m pip install -r requirements.txt
brew install ffmpeg
```

如果沒有 `python3.11`：

```bash
brew install python@3.11
```

## Windows CPU 安裝

PowerShell：

```powershell
cd D:\whisperx_project
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
python -m pip install -r requirements.txt
```

安裝 FFmpeg，擇一即可：

```powershell
choco install ffmpeg
```

```powershell
scoop install ffmpeg
```

## Windows CUDA 安裝

先安裝 NVIDIA driver，並確認 PyTorch 能看到 CUDA。CUDA toolkit 版本沒有固定要求，依 PyTorch 官方安裝選擇器選 Windows + Pip + CUDA 版本即可。WhisperX README 目前提到 GPU 加速可先安裝 CUDA toolkit 12.8。

常見流程：

```powershell
cd D:\whisperx_project
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

WhisperX 3.8.5 需要 `torch~=2.8.0`、`torchaudio~=2.8.0`、`torchvision~=0.23.0`。請鎖定版本安裝 CUDA 12.8 wheel：

```powershell
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

如果之前裝到不相容版本，例如 `torch 2.11.0+cu128`，先移除再重裝：

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

接著安裝或補齊本專案其他依賴：

```powershell
python -m pip install -r requirements.txt
```

驗證 CUDA：

```powershell
python -c "import torch, torchvision, torchaudio; print(torch.__version__); print(torchvision.__version__); print(torchaudio.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

輸出應類似：

```txt
2.8.0+cu128
0.23.0+cu128
2.8.0+cu128
True
12.8
NVIDIA GeForce RTX xxxx
```

如果 `torch.cuda.is_available()` 是 `False`，請先修正 PyTorch / NVIDIA driver / CUDA 安裝，再用 `--device cuda`。

## Hugging Face token

pyannote diarization 需要 Hugging Face read token。請先接受 `pyannote/speaker-diarization-community-1` 的模型條款，建立 read token，然後設定環境變數。

macOS / bash / zsh：

```bash
export HF_TOKEN="你的 Hugging Face read token"
```

Windows PowerShell，目前視窗有效：

```powershell
$env:HF_TOKEN="你的 Hugging Face read token"
```

Windows PowerShell，永久寫入使用者環境變數：

```powershell
setx HF_TOKEN "你的 Hugging Face read token"
```

## 執行

自動選擇 CUDA 或 CPU：

```bash
python -m scripts.run recording.m4a
```

指定輸出路徑：

```bash
python -m scripts.run recording.m4a -o outputs/recording_speakers.txt
```

強制 CPU：

```bash
python -m scripts.run recording.m4a --device cpu --compute-type int8
```

強制 CUDA：

```bash
python -m scripts.run recording.m4a --device cuda --compute-type float16 --batch-size 16
```

如果 CUDA 記憶體不足，可降低 batch size 或改用 `int8`：

```bash
python -m scripts.run recording.m4a --device cuda --compute-type int8 --batch-size 4
```

使用更大的 Whisper model：

```bash
python -m scripts.run recording.m4a --model large-v3
```

預設每一行會在最前面顯示 timecode。若要關閉 timecode：

```bash
python -m scripts.run recording.m4a --timecode
```

預設輸出：

```txt
outputs/recording.txt
```

輸出格式範例：

```txt
[00:00:03.120] A: 第一位講者的內容。
[00:00:08.450] B: 第二位講者的內容。
[00:00:12.300] C: 第三位講者的內容。
```

## 注意事項

- OpenAI Whisper setup 要求系統有 `ffmpeg`；Windows 可用 Chocolatey 或 Scoop，macOS 可用 Homebrew。
- WhisperX CPU/Mac 執行可使用 `--device cpu --compute-type int8`。
- WhisperX CUDA 執行通常使用 `float16`；低 VRAM 時可改 `int8` 或降低 `--batch-size`。
- 本腳本會用 WhisperX 載入的 16kHz mono waveform 交給 pyannote，避開某些平台上 `torchcodec` 或 FFmpeg dylib 解析音訊時的問題。

## 測試

```bash
python -m unittest discover -s tests
```
