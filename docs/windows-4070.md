# Windows run guide, RTX 4070

This is the path for Thomas's machine. The VM that holds the tests has no GPU and no microphone, so every step below that needs weights or a sound device is `NOT_RUN` there.

One command from the clone, after Python is on `PATH`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\Smoke-4070.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\Start-Maxi.ps1
```

`Smoke-4070.ps1` prints `PASS` or `NOT_RUN` for the self-test, llama-server on port 8080, ears on 8077, mouth on 8078, pynput, the presence window, and the microphone. `Start-Maxi.ps1` opens text mode with the `thomas-4070` profile. Add `-Mode presence` for the `/hold` script, or `-Mode selftest` for the memory regression alone. Neither script starts llama-server.

Run the self-test before any model. It does not need the GPU.

The 4070 has 12 GB. Load one text GGUF at a time. Do not put LFM3-Audio-2B BF16 in the same process as the friend brain. Memory writes stay in this Python program.

## 1. Self-test, before any model

From a PowerShell prompt in the clone:

```powershell
cd C:\src\hyper-ambient-maxi
$env:PYTHONPATH = "src"
py -3 -m hyper_ambient_maxi --selftest
py -3 -m pytest -q
```

`cmd.exe` is the same with `set PYTHONPATH=src`.

`--selftest` replays the diagnosis, the Airbus job, the rename to Biscuit, the French copies of those slots on a separate directory, the constraint warning, and a greeting check-in. A pass does not mean the microphone works.

`home\` is gitignored. Leave it that way. Do not commit it, and do not put an API key in `config\models.toml`. A local llama-server does not need `HAM_BRAIN_API_KEY`. If you later point the brain at a remote endpoint, set that variable in the shell. It is sent as a header and is not written into the journal.

## 2. llama-server

Use a llama.cpp build new enough for the quant you load. Bind the loopback only. `-ngl 99` offloads layers to the 4070. `-c 8192` matches the profile timeout budget for a single turn, not a long chat dump.

One of these, not all of them:

```powershell
llama-server.exe -m C:\models\LFM2.5-8B-A1B-Q4_K_M.gguf --host 127.0.0.1 --port 8080 -c 8192 -ngl 99 --reasoning off --alias lfm2.5-8b-a1b
```

`--reasoning off` matters for LFM3-1.7B and any template that thinks before it answers. A local check with `LFM3-1.7B-Q4_K_M` left `content` empty, filled `reasoning_content`, and with `max_tokens` 200 often stopped on `finish_reason=length` mid-thought. With reasoning left on, a 1.7B turn took about 18 seconds. Maxi does not speak `reasoning_content`. It strips `<think>` tags, cuts a hallucinated `User:` turn, and retries once with a larger budget if `content` is empty. The first request already sends `enable_thinking` false and `reasoning_effort` none unless the profile sets `think = true`. `llama-local`, `thomas-4070`, and `lfm3-audio` set `think = false`. Pass `--reasoning off` as well so the server spends the budget on the reply. The profile `max_tokens` (512 on `llama-local` and `thomas-4070`) is that budget. Raise it in `config\models.toml` if a model still stops early. A glued second answer (`involves.That’s awesome news!`) is cut, and `max_sentences` (default 3) caps what is spoken.

Point Maxi at a server that is not on port 8080 without editing the profile:

```powershell
$env:HAM_BRAIN_BASE_URL = "http://127.0.0.1:8080/v1"
py -3 -m hyper_ambient_maxi --profile llama-local --brain-url http://127.0.0.1:8080/v1 --home .\home --once "My dog is called Pixel."
```

`--brain-url` and `HAM_BRAIN_BASE_URL` override `brain_settings.base_url`. The API key, when you need one, stays `HAM_BRAIN_API_KEY`.

```powershell
llama-server.exe -m C:\models\TwIL-8B-Q4_K_M.gguf --host 127.0.0.1 --port 8080 -c 8192 -ngl 99 --reasoning off --alias twil-8b
```

```powershell
llama-server.exe -m C:\models\LFM3-1.7B-Q4_K_M.gguf --host 127.0.0.1 --port 8080 -c 8192 -ngl 99 --reasoning off --alias lfm3-1.7b
```

For that last one, set `brain_settings.model` to `lfm3-1.7b` so the request alias matches. The file is the one that was checked locally.

TwIL is also seen as `TwIL-8B-Q4_K_M-balanced.gguf`. Point `-m` at the file you actually have. The alias has to match `brain_settings.model` in the profile, or you edit the profile.

The other named files, still one at a time:

| Alias | File |
|---|---|
| `lfm3-1.7b` | `LFM3-1.7B-Q4_K_M.gguf` |
| `nemotron-3.5-bolt` | `Nemotron-3.5-Bolt-E4B-Q4_K_XL.gguf` |
| `vllm-d3-flash` | `vllm-d3-flash-Q4_K_M.gguf` |

`lfm2.5-8b-a1b` is the default friend brain. `vllm-d3-flash` was a decision model in AI Friend 2. Maxi does not ask it for permission to write.

Check the server with a browser or curl against `http://127.0.0.1:8080/v1/models`. Then:

```powershell
$env:PYTHONPATH = "src"
py -3 -m hyper_ambient_maxi --profile llama-local --home .\home --once "My dog is called Pixel."
py -3 -m hyper_ambient_maxi --profile llama-local --home .\home --once "What's my dog's name?"
```

If llama-server is down, the fact is still saved and the reply falls back to the scripted brain. The turn log says `brain=scripted-fallback`. That is a missed generation, not a missed write.

## 3. Voice profile

`thomas-4070` in `config\models.toml`:

| Piece | Setting |
|---|---|
| Brain | `http://127.0.0.1:8080/v1`, model `lfm2.5-8b-a1b` |
| Ears | `http-stt` at `http://127.0.0.1:8077`, `POST /v1/audio/transcriptions`, multipart field `file` |
| Mouth | `http-tts` at `http://127.0.0.1:8078`, `POST /v1/audio/speech`, JSON `{"input","voice":"female"}`, audio bytes back |

Hyper Ambient Mini's parakeet server already speaks the ears shape on port 8077. This repo can fill both ports itself, with no LFM3 and no microphone. One command each, in its own window:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\Smoke-4070.ps1 -Mode ears
powershell -ExecutionPolicy Bypass -File .\scripts\Smoke-4070.ps1 -Mode mouth
```

Those are `py -3 -m hyper_ambient_maxi --serve-ears` (faster-whisper, `127.0.0.1:8077`) and `--serve-mouth` (Piper, `127.0.0.1:8078`). With no `--profile` they use `thomas-4070`: Whisper `base`, on CUDA when a device is present and on CPU otherwise. Set `ears_settings.language` (or send a multipart `language` field) when the language is known; otherwise the server detects it. `initial_prompt` and `hotwords` are optional multipart fields. Decoding stays `beam_size` 5 and `temperature` 0. A missing cache prints `NOT_RUN` and names `python -m hyper_ambient_maxi --download-models`. That command, or `Smoke-4070.ps1 -Mode setup`, fetches Whisper `base` and the Piper voices once. It never runs by itself. `--smoke` then reports `ears-8077 PASS` and `mouth-8078 PASS`. A turn also sends `ears_settings.hotwords` plus the dog's name and people from memory.

`--voice-file in.wav` and `--wav in.wav` both follow the profile. On `thomas-4070` they post to `:8077` and `:8078`. `--voice-out reply.wav` names the mouth file for either mode; the default is `in.reply.wav`.

```powershell
py -3 -m hyper_ambient_maxi --profile thomas-4070 --home .\home --wav in.wav
```

If llama-server is down, the transcript is still stored and the reply falls back to the scripted brain. The mouth turns that reply into a WAV.

```powershell
py -3 -m hyper_ambient_maxi --profile thomas-4070 --home .\home --text
```

`--text` uses the brain only. Ears and mouth are for `--wav`. A WAV with no reachable STT server prints `NOT_RUN` and exits 3. A missing speaker does not roll back a write that already verified.

`--listen` exits 3. `--presence` without `--hotkey script` exits 3. `--hotkey pynput` is the optional global F9 hook (`pip install -e ".[hotkey]"`). Until that package is installed it prints `NOT_RUN` and does not grab a key. `--presence-window` opens the tkinter state window when Windows has a display, and prints `NOT_RUN` when tkinter or a display is missing. The headless presence loop is:

```powershell
py -3 -m hyper_ambient_maxi --profile thomas-4070 --presence --hotkey script --home .\home
```

```text
/say Ma mère a été diagnostiquée d'un cancer du sein, et j'ai peur.
/say Salut
/hold
/release Mon chien s'appelle Pixel.
```

`/hold` while a reply is on the frame clock cancels the rest of that playback. The words after `/release` are the utterance, because speech recognition is still the HTTP server, not this window. There is no tkinter Presence window in this tree.

## 4. LFM3-Audio-2B

The BF16 checkpoint stays where it already is, under the LFM Open License v1.0. It is ears and mouth: 16 kHz in, text and/or 24 kHz speech out. The model card says transcription and synthesis cover 18 languages, and voice chat is English-only. It is a poor fit for the memory write. Do not point the friend brain at it.

The profile `lfm3-audio` is `NOT_RUN` until you wrap that checkpoint behind the ears and mouth URLs. A GGUF sibling exists for llama.cpp b11960 / v0.8.0 or newer (`/v1/realtime`, `llama-tts`). That is a second server, not a replacement for the text GGUF on port 8080. French slot extraction already runs in this process, so a French transcript from the 18-language ASR path can be committed without the audio model doing the remembering.

## 5. CPU voice without a microphone

This is separate from the 4070 HTTP ears and mouth. It runs in-process:

```powershell
py -3 -m pip install -e ".[voice]"
# Cache Whisper tiny and the two low Piper voices under
# $env:USERPROFILE\.cache\hyper-ambient-maxi\
#   whisper\   (faster-whisper tiny, int8, CPU)
#   piper\en_US-lessac-low.onnx
#   piper\fr_FR-siwis-low.onnx
py -3 -m hyper_ambient_maxi --home .\home --voice-file in.wav
```

The command transcribes `in.wav`, stores memory with the scripted brain unless `--profile` says otherwise, and writes `in.reply.wav`. On the test VM, warm latency was 0.5 s for speech-to-text (tiny, int8, one CPU thread) and 0.03–0.04 s for Piper. A missing cache prints `NOT_RUN`. The microphone is still not opened.

## 6. Sharing the same home

Pi and OpenCode can use the store the 4070 just wrote:

```powershell
py -3 -m hyper_ambient_maxi --mcp --home C:\src\hyper-ambient-maxi\home
```

`friend_context` commits the user sentence and reads it back. `source_agent` is the client name. A skipped `friend_remember` does not drop the fact.

Optional local vectors, still no network:

```powershell
$env:HAM_EMBEDDINGS = "hash"
```

Unset it to stay on word overlap, which is the default. Hashing is not a sentence-transformer download.
