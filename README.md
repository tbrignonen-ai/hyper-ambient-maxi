# Hyper Ambient Maxi

Hyper Ambient Maxi is a local harness that listens, keeps a long-term memory of the user, and answers. The verified path is text in and text out, with no GPU and no microphone. Voice in and voice out use the same turn. On a machine without audio hardware or weights they stop with `NOT_RUN` and do not pretend to have heard or spoken.

The memory write does not belong to the model. Every turn extracts facts from the user sentence, appends them to a journal, fsyncs, rebuilds a snapshot, and reads them back before the reply is allowed to say they were kept.

## Architecture

```
text, WAV, or the presence script
        |
        v
+-------------------------------------------+
| ears                                      |
| energy gate, VAD, junk-transcript filter  |
| push-to-talk, optional F9 via pynput      |
| duplex clock: barge-in on fixture speech  |
+-------------------------------------------+
        |
        v
extract (English and French, including
negation, "non, en fait", demain / jeudi prochain)
        |
        v
journal append + fsync
snapshot rename + fsync
second open must show the same active slot
        |
        v
salience packet (lexical, optional local hash vectors)
open threads, constraint warning, greeting check-in
        |
        v
brain: scripted, or llama-server, or scripted-fallback
        |
        v
mouth: print, HTTP TTS, or NOT_RUN
        |
        v
presence view: state, last reply, last memory written
tkinter window when a display exists, otherwise NOT_RUN
```

## Memory guarantees

A turn writes before the brain speaks. These are the checks the tests lock in:

1. Facts are extracted from the user sentence. The model is not asked to call a remember tool.
2. The journal append is fsynced. The journal is the source of truth.
3. The snapshot is written to a temporary file, fsynced, and renamed into place. The directory is fsynced after the rename.
4. A second `MemoryStore` open of the same directory must show the new row as the single active slot, with the same value.
5. If that read fails, the reply says it was not saved, and a later open does not show the fact.
6. The same slot key supersedes. The previous row stays in the journal with status `superseded`.
7. Pronoun objects (`that`, `it`, `this`, `ça`) are refused. "Please remember that." stores nothing.
8. The prompt block is sentences only. It does not contain importance, relevance, recency, salience, priority, or score numbers, and it does not print `[high]` or `[low]`. Those fields may exist in `state.json` and are used only to order the block.
9. `friend_context` on the MCP server uses this same commit, so a skipped `friend_remember` still leaves a verified fact. `source_agent` is the client name.
10. A missing speaker, a missing display, and a brain that raises do not roll back a write that already verified. The brain failure is `scripted-fallback` after the save.

AI Friend 1.8.3 split "what do you know?" and "please store this" into two model-called tools, `friend_context` and `friend_remember`. On 5 November 2026 a three-harness run kept a dog's name and a coffee preference, and lost three other writes because that second call was skipped, returned "stored 0", or never superseded the old value.

The same commit runs inside `friend_context`. Pi and OpenCode can share `./home` over MCP, and a client that never calls `friend_remember` still leaves a verified fact. `source_agent` is the MCP `clientInfo.name` (`pi`, `opencode`), not a hardcoded `mcp`.

Slots are single-valued. A new dog name or employer supersedes the old row. The old row stays in the journal with status `superseded`. Pronoun objects (`that`, `it`, `this`, `ça`) are dropped, which is the "Remembered: that" junk fact from the 1.8.3 Linux report. The model-facing block is sentences only. Importance is stored on the snapshot for debugging and is not rendered. Salience uses the 1.8.3 formula to choose which rows fit, then drops the numbers. A profile question ("what do you remember about me?") fills remaining slots by importance, including health and job, not only personal facts.

French uses the same keys as English. "Ma mère a été diagnostiquée d'un cancer du sein" writes `health:mum-diagnosis`. "Je l'ai renommé Biscuit" supersedes `fact:dog-name`. A diagnosis opens `thread:health-mum`. "I'm done with mum's diagnosis" closes it. A prohibition ("Do not restart the training gateway…", "Ne redémarre pas la passerelle…") is stored as a constraint and does not warn on that same sentence. The next action that overlaps the rule adds "You may be about to act against" (or the French line) to the reply. A later greeting ("Hello", "Salut") may offer one check-in on the oldest open high-priority thread, such as a diagnosis or an appointment. A constraint thread speaks when an action is about to cross it.

The same slots exist for a name, an allergy, a city, and an appointment: "Je m'appelle Thomas", "Je suis allergique aux arachides", "J'habite à Lyon", "J'ai rendez-vous chez le médecin jeudi", and the English sentences of the same shape. "N'oublie pas que …" keeps the clause.

If a sentence looks durable (a diagnosis, a job, a rename, a preference) but does not match a slot pattern, the whole sentence is stored as an episode. A question is not stored. "What is the capital of France?" adds nothing.

## What was reused

| Source | What came across | What did not |
|---|---|---|
| [hyper-ambient](https://github.com/tbrignonen-ai/hyper-ambient) | The split into ears, brain, and mouth, and the rule that a local `llama-server` and any other chat model share one `/v1/chat/completions` client. | No source files. The clone has no `LICENSE`, so the OpenAI client, the energy gate, and the turn loop are new code. The GATE audit log and the container are not here. The presence view is Maxi's. |
| [hyper-ambient-mini](https://github.com/tbrignonen-ai/hyper-ambient-mini) | Apache-2.0 helpers in `src/hyper_ambient_maxi/ears/wavutil.py`: 16 kHz WAV framing, int16 RMS, linear resample, and the junk-transcript filter (subtitle credits and repeated-token hallucinations). See `NOTICE`. | The tkinter push-to-talk window, the global F9 hook, and the Windows `parakeet-server.exe` download. A running mini server on `127.0.0.1:8077` can still be the ears backend (`http-stt`). Push-to-talk here is the press / feed / release state machine, tested with PCM, not a hotkey. |
| AI Friend 1.8.3, "the friend who remembers" | MIT algorithms, reimplemented: salience ranking, open threads, the constraint-conflict warning, lexical overlap, the continuity router, and the MCP tool names `friend_context` and `friend_remember`. See `NOTICE`. The journal's write-then-read-back is Maxi's, including inside `friend_context`. | The Origin clone returned HTTP 403. The attached 1.8.3 archive was read for those algorithms and is not vendored. Its 279 tests were not run. The 1.8.3 context formatter printed relevance and importance; Maxi does not. |
| AI Friend 2 prototype | The note that a decision score must not authorize an action. Routing never decides whether a fact is written. | `decision/core.py` and `router.py` were not copied. The prototype sends an emotional sentence to a harness that may never call `friend_remember`. Maxi extracts that sentence first. |

## The three lost writes

These are the regression. `python -m hyper_ambient_maxi --selftest` replays them and reopens the directory before it checks.

| 2026-11-05 miss | What happened there | What Maxi does |
|---|---|---|
| Diagnosis. "My mum was diagnosed with breast cancer last week, and I'm scared." | OpenCode answered with sympathy and never called `friend_remember`. | The harness extracts `health:mum-diagnosis` before the reply. The reply can still be sympathetic. |
| Job. Mercury called `friend_remember` and stored 0. The notes do not include the original user sentence; the assistant reply mentions Airbus. | A remember call that stores nothing was treated as success. | "I just got a job at Airbus." must come back as `job:employer = Airbus` with `stored >= 1`. A later "I left Airbus and I work at Safran now." supersedes the slot. An odd employment sentence that does not match a pattern is kept verbatim. |
| Rename to Biscuit | The new name was never applied. | "I renamed him to Biscuit." replaces `fact:dog-name`. Recall returns Biscuit. Pixel remains as history. "What did I used to call my dog?" answers Pixel. |

The two writes that run did keep are still covered: "My dog is called Pixel." and "I prefer oat milk in my coffee." "Please remember that." stores nothing, and it does not create `fact:that`.

## Model configuration

`config/models.toml` is the card. `python -m hyper_ambient_maxi --show-models` prints it.

| Profile | Ears | Brain | Mouth | Status here |
|---|---|---|---|---|
| `text` (default) | text | scripted | text | **verified**. This is what the tests run. |
| `llama-local` | text | `llama-server` at `127.0.0.1:8080` | text | **NOT_RUN**. Needs the GGUF server. If it is down, the write still happens and the reply falls back to the scripted brain (`brain=scripted-fallback`). |
| `thomas-4070` | HTTP STT, default `:8077` | `llama-server`, model alias `lfm2.5-8b-a1b` | HTTP TTS, default `:8078` | Brain **NOT_RUN** without the GGUF server. Ears and mouth **PASS** when `--serve-ears` and `--serve-mouth` are up. No microphone. |
| `lfm3-audio` | LFM3-Audio transcript | `llama-server` (not the audio model) | LFM3-Audio speech | **NOT_RUN**. See below. |

Known text models, all Q4 GGUF, all meant to be served by llama.cpp `llama-server`. None are downloaded by this repo.

| Alias | File |
|---|---|
| `twil-8b` | `TwIL-8B-Q4_K_M.gguf` (also seen as `TwIL-8B-Q4_K_M-balanced.gguf`) |
| `lfm3-1.7b` | `LFM3-1.7B-Q4_K_M.gguf` |
| `lfm2.5-8b-a1b` | `LFM2.5-8B-A1B-Q4_K_M.gguf` |
| `nemotron-3.5-bolt` | `Nemotron-3.5-Bolt-E4B-Q4_K_XL.gguf` |
| `vllm-d3-flash` | `vllm-d3-flash-Q4_K_M.gguf` |

`lfm2.5-8b-a1b` is the default friend brain. It is the family the 1.8.3 Linux check used. TwIL-8B is the alternative when you want the model those harness turns actually called. `vllm-d3-flash` was the AI Friend 2 decision model. Maxi does not ask it for permission to write.

```sh
llama-server -m /path/to/LFM2.5-8B-A1B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 -c 8192 --alias lfm2.5-8b-a1b
PYTHONPATH=src python -m hyper_ambient_maxi --profile llama-local --once "My dog is called Pixel."
```

The API key, if a remote endpoint needs one, is `HAM_BRAIN_API_KEY`. It is sent as a header and is not written into the journal. Do not put it in `models.toml`. `HAM_BRAIN_BASE_URL` or `--brain-url` overrides the profile brain URL. Reasoning models should be started with `llama-server --reasoning off`. Llama and voice profiles also send `enable_thinking` false on the first request (`think = false` in the profile). A second answer glued on with no space, or a later greeting, is dropped, and `max_sentences` (default 3) caps the spoken reply. Maxi does not speak `reasoning_content`. See [docs/windows-4070.md](docs/windows-4070.md).

### LFM3-Audio-2B

Evaluated from the public model card, not executed. Thomas has the BF16 checkpoint on the RTX 4070. The card (`LiquidAI/LFM3-Audio-2B`) says:

- About 2B parameters. Input is text or 16 kHz audio. Output is text, 24 kHz speech, or both.
- Transcription, synthesis, and translation cover 18 languages (19 locales; British and American English count as one language). French is in that list, which is the gap Hyper Ambient's frozen card still treats as a separate faster-whisper plus Magpie stack.
- Voice chat, as opposed to transcription and synthesis, is documented as English-only today.
- The same card says the 2B model is a poor fit for multi-step reasoning. Memory is multi-step in the only way that matters here: a fact has to survive the next process. So LFM3-Audio is ears and mouth, not the steward and not the friend brain.
- Weights are under the LFM Open License v1.0. They are not in this repository.
- A GGUF sibling, `LiquidAI/LFM3-Audio-2B-GGUF`, runs in llama.cpp b11960 / v0.8.0 or newer, including `/v1/realtime` and `llama-tts`. That path is also `NOT_RUN` here. There is no CUDA device on the test VM, and the weights were not downloaded.

Hyper Ambient's own card (Granite 4.2 3B, faster-whisper large-v3, Magpie Sofia, optional MiniMax) is a different machine profile. Maxi does not vendor those weights. `thomas-4070` can point ears and mouth at HTTP servers in front of them if you already run them.

## Setup

Python 3.11 or newer. The core has no third-party dependencies.

```sh
git clone <this-repo> hyper-ambient-maxi
cd hyper-ambient-maxi
PYTHONPATH=src python -m hyper_ambient_maxi --selftest
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m hyper_ambient_maxi --home ./home --text
PYTHONPATH=src python -m hyper_ambient_maxi --home ./home --once "My dog is called Pixel."
PYTHONPATH=src python -m hyper_ambient_maxi --mcp --home ./home
```

`--mcp` speaks JSON-RPC on stdin/stdout (protocol `2024-11-05`). Point Pi or OpenCode at that command and the same `--home`. `friend_context` extracts and commits the user message, then returns the score-free block. `friend_remember` repeats the write and is idempotent. A failed read-back is a tool error, not `stored 0`.

`./home` is gitignored. It holds `journal.jsonl` (the source of truth), `state.json` (the snapshot), and `turns.jsonl` (one record per turn, including failed saves).

A WAV file on the text profile does not invent a transcript:

```sh
PYTHONPATH=src python -m hyper_ambient_maxi --wav turn.wav
# prints NOT_RUN and exits 3
```

`--listen` does the same. This build never opens a microphone.

`--voice-file in.wav` is the no-microphone path. It uses the profile ears and mouth. On the default text profile that is in-process faster-whisper and Piper. On `thomas-4070` it posts to `:8077` and `:8078`. `--wav in.wav` stays on the profile (a text profile still prints `NOT_RUN`) and writes the reply WAV when the mouth returns audio. `--voice-out reply.wav` names that file for either mode; the default is `in.reply.wav`. A missing voice cache prints `NOT_RUN` and exits 3.

Playback is a frame clock. Each step emits one 20 ms frame of the reply audio and reads one microphone frame. About 100 ms of speech cancels the rest of the buffer. A click shorter than that does not. Tests feed both streams as PCM. There is no sound device in that path.

```text
/say My mum was diagnosed with breast cancer last week.
/say Hello
/hold
/release My dog is called Pixel.
```

`python -m hyper_ambient_maxi --presence` does not grab a key. It prints `NOT_RUN` and exits 3. `--presence --hotkey script` reads the lines above on stdin. `ScriptedHotkey` is the same press/release contract for tests. `--hotkey pynput` is the optional F9 hook (`pip install -e '.[hotkey]'`). `--presence-window` paints state, reply, and the last memory; headless it prints `NOT_RUN`.

Ranking stays lexical unless `HAM_EMBEDDINGS=hash`. That flag adds a local hashing embedder in this process. It is used only when the words do not already overlap, and only if the cosine is at least 0.55. If `embed` raises, the score stays lexical. Tests pass a fixture embedder. Nothing in this path opens a network connection.

Speech regions are also cut before a turn is stored: a region starts after about 100 ms above the silence RMS and ends after about 400 ms of quiet. `PushToTalk` applies that cut on release and drops samples that arrive while the key is up. Those fixtures are separate from the barge-in clock above.

Optional extras: `pip install -e '.[voice]'` (faster-whisper, piper-tts, sounddevice) and `pip install -e '.[hotkey]'` (pynput). The voice extra does not mark the GPU path or the microphone as run. This VM also cached Whisper `tiny` and `base`, plus Piper `en_US-lessac-low` and `fr_FR-siwis-low`, under `~/.cache/hyper-ambient-maxi/`. Those files are not in the git tree. `HAM_VOICE_DIR` moves the cache. `HAM_WHISPER_MODEL` defaults to `base` when that model is already in the cache, otherwise `tiny`. `thomas-4070` asks `--serve-ears` for `base` on CUDA, and falls back to CPU when no device is present. A missing cache prints `NOT_RUN` and names `python -m hyper_ambient_maxi --download-models` (or `Smoke-4070.ps1 -Mode setup`). That command fetches Whisper `base` and the Piper voices once. A turn does not download. A known language is `ears_settings.language` or a multipart `language` field; otherwise detection stays on. `--voice-file` and `--wav` send `ears_settings.hotwords` plus the dog's name and people from memory. Decoding stays beam size 5 and temperature 0. French text is NFC before Piper, so "renommé" keeps its é. The round-trip test is skipped when that cache is absent. A stored name within edit distance 2 of Pixel or Biscuit still passes.

## Tests

From this VM, with Python 3.12:

```sh
PYTHONPATH=src python -m pytest -q
```

73 passed on this VM (Python 3.12). That includes the journal fsync spy, a dropped journal write, the diagnosis, the Airbus job, the Biscuit rename including "Actually I renamed him Biscuit" and the French forms, pronoun rejection, a packet with no priority or `[high]` marker, French slots including negation and "non, en fait", `demain` and `jeudi prochain`, the constraint warning, the greeting check-in, MCP `friend_context`, VAD, barge-in, the presence view without Tk, the smoke report, a brain client that does not speak `reasoning_content`, llama-local sending thinking disabled on the first request, a voice reply that keeps only the first answer when a second one is glued on (`involves.That’s awesome news!`), a system prompt that calls the user Thomas (or the name slot) and rewrites a leading "Hi Maxi", a no-microphone Piper to faster-whisper round trip, the same loop through HTTP ears on `:8077` and mouth on `:8078` with a name within edit distance 2 still accepted, `--voice-file` honoring the profile while `--wav` writes `--voice-out`, hotwords from memory, and an explicit `--download-models` that does not run on its own.

## Verified and NOT_RUN

| Piece | On this VM | How it is checked |
|---|---|---|
| Text profile, scripted brain, journal | **verified** | `pytest`, `--selftest` |
| French slots, negation, corrections, relative dates | **verified** | `tests/test_french.py` |
| Salience without scores, threads, constraint warning, greeting check-in | **verified** | self-test and `tests/test_french.py` |
| MCP `friend_context` write-then-read-back | **verified** | `tests/test_mcp.py` |
| VAD, push-to-talk buffer, duplex barge-in | **verified** on PCM fixtures | `tests/test_vad.py`, `tests/test_duplex.py` |
| Presence state (listening, thinking, speaking, last memory) | **verified** without a display | `tests/test_window.py` |
| Presence tkinter window | **NOT_RUN** headless | `--presence-window` exits 3 |
| Global F9 via pynput | **NOT_RUN** until `pip install -e '.[hotkey]'` | `--hotkey pynput` exits 3. `auto` never grabs a key |
| Microphone, `--listen` | **NOT_RUN** | exit 3 |
| faster-whisper `tiny` int8, one CPU thread | **verified**, no microphone. Warm speech-to-text **0.5 s** on this 4-core Xeon for both "My dog is Pixel." and "Je l'ai renommé Biscuit." | `tests/test_voice_roundtrip.py`. Skipped when the cache is absent |
| Piper `en_US-lessac-low` and `fr_FR-siwis-low`, CPU | **verified**, no speaker. Warm synthesis **0.04 s** English and **0.03 s** French for those same sentences | same test. espeak playback stays **NOT_RUN** (no audio device) |
| `llama-server` and the five GGUF files | **NOT_RUN** | `--smoke` prints `llama-server NOT_RUN` when port 8080 is closed |
| HTTP STT `:8077`, HTTP TTS `:8078` | **verified** on this VM. `--serve-ears` is faster-whisper and `--serve-mouth` is Piper. A WAV of "My dog is Pixel." then "Je l'ai renommé Biscuit." stored both names through `thomas-4070` with the scripted fallback while llama-server was down | `--smoke` prints `ears-8077 PASS` and `mouth-8078 PASS`. `tests/test_http_speech.py` skips when the voice cache is absent |
| LiquidAI/LFM3-Audio-2B, BF16 and GGUF | **NOT_RUN** | model card |

`python -m hyper_ambient_maxi --smoke` prints one line per piece. `PASS` and `NOT_RUN` are both a successful smoke. `FAIL` is a self-test that did not pass, and the process exits 1.

Thomas's RTX 4070 steps are in [docs/windows-4070.md](docs/windows-4070.md). From that checkout:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\Smoke-4070.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\Start-Maxi.ps1
```

`Start-Maxi.ps1` is text mode on the `thomas-4070` profile. It does not start llama-server. The guide was not executed on this VM.

A loud WAV plus an injected transcript does go through the junk filter and the memory commit. That checks the wiring. It is not speech recognition.

## Limits

- French covers the slots above, plus negation ("ne s'appelle pas", "n'aime pas", "ne travaille plus"), a leading "non, en fait", and appointment dates `demain` and `jeudi prochain`. "il s'appelle …" updates `fact:dog-name`. It is still not a general French parser.
- Duplex is a frame clock with fixture PCM. `--listen` does not open a microphone. The tkinter window paints the presence state when a display exists. Headless runs print `NOT_RUN`.
- `--hotkey pynput` is the global F9 backend. It is an optional extra and it is not started by `--hotkey auto`.
- The exact Airbus sentence from the harness log was not in the notes. The test uses "I just got a job at Airbus." and a follow-up change of employer.
- Odd phrasing is kept as an episode, which is recoverable by word overlap and is not as clean as a slot.
- Initiative is the constraint conflict and one greeting check-in. A contradiction nudge and standing-preference injection are not implemented. Appointment dates are stored on the fact; they do not schedule a reminder.
- The optional embedder is hashing, or a caller-supplied function. It is not a downloaded model. The default profile does not set `HAM_EMBEDDINGS`.
