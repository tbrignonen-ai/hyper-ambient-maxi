"""Command line. Text is the default. Voice backends print NOT_RUN."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from hyper_ambient_maxi import __version__
from hyper_ambient_maxi.brain.openai_compat import OpenAICompatBrain
from hyper_ambient_maxi.brain.scripted import ScriptedBrain
from hyper_ambient_maxi.config import ModelsFile, Profile, load_models
from hyper_ambient_maxi.loop import run_turn
from hyper_ambient_maxi.memory.store import MemoryStore
from hyper_ambient_maxi.selftest import run_selftest
from hyper_ambient_maxi.voice import hotwords_for, listen_status, run_audio_turn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hyper-ambient-maxi",
        description="Local voice harness with harness-owned memory.",
    )
    parser.add_argument("--home", default="home", help="memory directory")
    parser.add_argument("--profile", default=None, help="profile name in config/models.toml")
    parser.add_argument("--models", default=None, help="path to models.toml")
    parser.add_argument("--once", default=None, help="run one text turn and exit")
    parser.add_argument("--text", action="store_true", help="interactive text")
    parser.add_argument("--wav", default=None, help="run one turn from a WAV file")
    parser.add_argument(
        "--voice-file",
        default=None,
        help="WAV in through the profile ears and mouth (text profiles use faster-whisper and Piper)",
    )
    parser.add_argument(
        "--voice-out",
        default=None,
        help="reply WAV for --voice-file and --wav (default: <input>.reply.wav)",
    )
    parser.add_argument("--listen", action="store_true", help="microphone capture (not run in v0)")
    parser.add_argument("--show-models", action="store_true", help="print the model card")
    parser.add_argument("--selftest", action="store_true", help="run the memory regression")
    parser.add_argument("--mcp", action="store_true", help="MCP stdio server for Pi and OpenCode")
    parser.add_argument("--presence", action="store_true", help="presence loop (text script or NOT_RUN hotkey)")
    parser.add_argument("--presence-window", action="store_true", help="tkinter presence window, or NOT_RUN")
    parser.add_argument("--smoke", action="store_true", help="PASS or NOT_RUN for each local piece")
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="fetch Whisper base and the Piper voices once (never automatic)",
    )
    parser.add_argument(
        "--serve-ears",
        action="store_true",
        help="HTTP speech-to-text on 127.0.0.1:8077 (faster-whisper)",
    )
    parser.add_argument(
        "--serve-mouth",
        action="store_true",
        help="HTTP speech synthesis on 127.0.0.1:8078 (Piper)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="port for --serve-ears or --serve-mouth",
    )
    parser.add_argument(
        "--hotkey",
        default="auto",
        help="auto (NOT_RUN), script (stdin), or pynput (optional extra)",
    )
    parser.add_argument(
        "--brain-url",
        default=None,
        help="override the brain base URL (same as HAM_BRAIN_BASE_URL)",
    )
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args(argv)
    if args.brain_url and args.brain_url.strip():
        os.environ["HAM_BRAIN_BASE_URL"] = args.brain_url.strip()

    if args.version:
        print(__version__)
        return 0
    if args.mcp:
        from hyper_ambient_maxi.mcp_server import serve_stdio

        return serve_stdio(Path(args.home))
    if args.download_models:
        return _download_models()
    if args.smoke:
        return _smoke()
    if args.serve_ears or args.serve_mouth:
        if args.serve_ears and args.serve_mouth:
            print("Start ears and mouth as two commands.")
            return 2
        from hyper_ambient_maxi.ears.http_speech import serve

        profile = _serve_profile(args.models, args.profile)
        if args.serve_ears:
            return serve(
                "ears",
                port=args.port or 8077,
                model=profile.ears_model,
                language=profile.ears_language,
                device=profile.ears_device,
            )
        return serve("mouth", port=args.port or 8078)
    if args.presence_window:
        return _presence_window(Path(args.home), args.models, args.profile)
    if args.presence:
        return _presence(Path(args.home), args.hotkey, args.models, args.profile)

    models = load_models(args.models)
    if args.show_models:
        _print_models(models)
        return 0
    if args.selftest:
        home = Path(args.home)
        if args.home == "home":
            import tempfile

            home = Path(tempfile.mkdtemp(prefix="maxi-selftest-"))
        return run_selftest(home)

    profile = models.profile(args.profile)
    if args.listen:
        status, reason = listen_status()
        print(f"{status}: {reason}" if not reason.startswith("NOT_RUN") else reason)
        return 3 if status == "NOT_RUN" else 0
    voice_out = Path(args.voice_out) if args.voice_out else None
    if args.voice_file:
        return _audio_file(
            Path(args.home),
            profile,
            Path(args.voice_file),
            voice_out,
            local_fallback=True,
        )
    if args.wav:
        return _audio_file(
            Path(args.home),
            profile,
            Path(args.wav),
            voice_out,
            local_fallback=False,
        )
    if args.once is not None:
        return _once(Path(args.home), profile, args.once)
    if args.text or not sys.stdin.isatty():
        return _text(Path(args.home), profile)
    _print_models(models)
    print()
    print("Text:  python -m hyper_ambient_maxi --text")
    print("Once:  python -m hyper_ambient_maxi --once \"My dog is called Pixel.\"")
    print("Voice: python -m hyper_ambient_maxi --voice-file in.wav")
    print("Models: python -m hyper_ambient_maxi --download-models")
    print("Ears:  python -m hyper_ambient_maxi --serve-ears")
    print("Mouth: python -m hyper_ambient_maxi --serve-mouth")
    print("Check: python -m hyper_ambient_maxi --selftest")
    print("MCP:   python -m hyper_ambient_maxi --mcp --home ./home")
    print("Talk:  python -m hyper_ambient_maxi --presence --hotkey script")
    print("Smoke: python -m hyper_ambient_maxi --smoke")
    return 0


def build_brain(profile: Profile):
    if profile.brain == "scripted":
        return ScriptedBrain()
    if profile.brain == "openai-compat":
        return OpenAICompatBrain(
            base_url=profile.brain_base_url or "http://127.0.0.1:8080/v1",
            model=profile.brain_model or "local",
            timeout_s=profile.brain_timeout_s,
            max_tokens=profile.brain_max_tokens,
            think=profile.brain_think,
            max_sentences=profile.brain_max_sentences,
        )
    raise SystemExit(f"unknown brain {profile.brain}")


def _once(home: Path, profile: Profile, text: str) -> int:
    store = MemoryStore(home)
    result = run_turn(
        store,
        build_brain(profile),
        text,
        source_agent="maxi-text",
    )
    print(result.reply)
    stored = 0 if result.receipt is None else result.receipt.stored
    flag = "verified" if result.memory_ok else "NOT SAVED"
    print(f"memory: stored {stored} {flag} brain={result.brain}", file=sys.stderr)
    return 0 if result.memory_ok else 1


def _text(home: Path, profile: Profile) -> int:
    store = MemoryStore(home)
    brain = build_brain(profile)
    print("Maxi text mode. Empty line quits. Memory is in", home)
    while True:
        try:
            line = input("> ")
        except EOFError:
            print()
            return 0
        if not line.strip():
            return 0
        result = run_turn(store, brain, line, source_agent="maxi-text")
        print(result.reply)
        if not result.memory_ok:
            print("memory: NOT SAVED")


def _download_models() -> int:
    """Whisper base and the Piper voices. Serving never calls this."""
    from hyper_ambient_maxi.ears.local_speech import download_voice_models

    try:
        lines = download_voice_models()
    except Exception as exc:
        print(f"NOT_RUN: download failed ({exc})")
        return 3
    for line in lines:
        print(line)
    return 0


def _serve_profile(models_path: str | None, profile_name: str | None) -> Profile:
    """Ears settings for the speech server. The 4070 profile is the default."""
    models = load_models(models_path)
    try:
        return models.profile(profile_name or "thomas-4070")
    except KeyError:
        return models.profile(None)


def _audio_file(
    home: Path,
    profile: Profile,
    path: Path,
    out: Path | None,
    *,
    local_fallback: bool,
) -> int:
    """One WAV through the profile ears and mouth.

    ``--voice-file`` falls back to in-process faster-whisper and Piper
    when the profile has no speech backend. ``--wav`` stays on the
    profile, so a text profile still reports ``NOT_RUN``. Both write
    the reply WAV when the mouth returns audio. ``--voice-out`` names
    that file. Otherwise it is ``<input>.reply.wav``.
    """
    if not path.is_file():
        print(f"NOT_RUN: wav not found: {path}")
        return 3
    ears = profile.ears
    mouth = profile.mouth
    if local_fallback:
        if ears in {"text", "scripted", ""}:
            ears = "faster-whisper"
        if mouth in {"text", "scripted", ""}:
            mouth = "piper"
    destination = out or path.with_name(path.stem + ".reply.wav")
    store = MemoryStore(home)
    outcome = run_audio_turn(
        store,
        build_brain(profile),
        path.read_bytes(),
        ears=ears,
        mouth=mouth,
        ears_url=profile.ears_base_url,
        mouth_url=profile.mouth_base_url,
        allow_download=False,
        language=profile.ears_language,
        initial_prompt=profile.ears_initial_prompt,
        hotwords=hotwords_for(store, profile.ears_hotwords),
        whisper_model=profile.ears_model,
        whisper_device=profile.ears_device,
    )
    if outcome.status == "NOT_RUN" or outcome.turn is None:
        reason = outcome.reason or outcome.status
        print(reason if str(reason).startswith("NOT_RUN") else f"{outcome.status}: {reason}")
        if outcome.status == "NOT_RUN":
            return 3
        return 0 if outcome.status in {"silence", "junk"} else 1
    print(outcome.turn.reply)
    if outcome.reason and outcome.reason not in {
        "text",
        "ok",
        "http-tts",
        "piper",
        "faster-whisper",
    }:
        print(outcome.reason, file=sys.stderr)
    if outcome.audio:
        destination.write_bytes(outcome.audio)
        print(f"wrote {destination}")
    elif local_fallback:
        print(outcome.reason or "NOT_RUN: no reply audio")
        return 3
    return 0 if outcome.turn.memory_ok else 1


def _presence(home: Path, hotkey: str, models_path: str | None, profile_name: str | None) -> int:
    from hyper_ambient_maxi.ears.hotkey import open_hotkey
    from hyper_ambient_maxi.presence import PresenceSession

    status = open_hotkey(hotkey)
    if status.status != "ok":
        print(status.reason)
        return 3
    if hotkey == "pynput":
        return _run_pynput(status)
    models = load_models(models_path)
    profile = models.profile(profile_name)
    session = PresenceSession(MemoryStore(home), build_brain(profile))
    print("Presence. /hold, /release <words>, /say <words>. Empty line quits.")
    while True:
        try:
            line = input()
        except EOFError:
            print()
            return 0
        if not line.strip():
            return 0
        for row in session.command(line):
            print(row)


def _smoke() -> int:
    from hyper_ambient_maxi.smoke import format_smoke, run_smoke, smoke_exit

    checks = run_smoke()
    print(format_smoke(checks))
    return smoke_exit(checks)


def _presence_window(home: Path, models_path: str | None, profile_name: str | None) -> int:
    from hyper_ambient_maxi.presence import PresenceSession
    from hyper_ambient_maxi.presence_window import display_status, open_window

    status, reason = display_status()
    if status != "ok":
        print(reason)
        return 3
    models = load_models(models_path)
    session = PresenceSession(MemoryStore(home), build_brain(models.profile(profile_name)))
    return open_window(session)


def _run_pynput(hook) -> int:
    import time

    print(hook.reason)
    print("F9 press and release. Ctrl-C quits. No microphone is opened.")
    hook.start()
    try:
        while True:
            event = hook.poll()
            if event == "press":
                print("[listening]")
            elif event == "release":
                print("[idle]")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
        return 0
    finally:
        hook.stop()


def _print_models(models: ModelsFile) -> None:
    print(f"models: {models.path}")
    print(f"default: {models.default_profile}")
    for name, profile in models.profiles.items():
        gpu = "gpu" if profile.requires_gpu else "no-gpu"
        audio = "audio-hw" if profile.requires_audio_hardware else "no-audio-hw"
        print(
            f"  {name}: ears={profile.ears} brain={profile.brain} "
            f"mouth={profile.mouth} {gpu} {audio} {profile.run_status}"
        )
        if profile.brain_model:
            print(f"    brain model {profile.brain_model} @ {profile.brain_base_url}")
        if profile.speech_model_id:
            print(f"    speech {profile.speech_model_id} {profile.speech_dtype}")
    print("known GGUF brains:")
    for name, spec in models.known_brains.items():
        print(f"  {name}: {spec.get('file')} alias={spec.get('alias')}")
    print("known speech:")
    for name, spec in models.known_speech.items():
        print(f"  {name}: {spec.get('model_id')} {spec.get('run_status')}")
