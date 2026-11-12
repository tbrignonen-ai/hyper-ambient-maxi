"""Load ``config/models.toml``. Profiles say what to run; known_models
lists the GGUF files Thomas already has, without requiring the files.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

def _default_models_file() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "config" / "models.toml"
        if candidate.is_file():
            return candidate
    return here.parents[2] / "config" / "models.toml"


_DEFAULT_FILE = _default_models_file()


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    ears: str
    brain: str
    mouth: str
    requires_gpu: bool
    requires_audio_hardware: bool
    run_status: str
    brain_base_url: str
    brain_model: str
    brain_timeout_s: float
    brain_max_tokens: int
    brain_think: bool
    brain_max_sentences: int
    ears_base_url: str
    ears_model: str
    ears_language: str
    ears_device: str
    ears_hotwords: str
    ears_initial_prompt: str
    mouth_base_url: str
    speech_model_id: str
    speech_dtype: str


@dataclass(frozen=True)
class ModelsFile:
    default_profile: str
    profiles: dict[str, Profile]
    known_brains: dict[str, dict]
    known_speech: dict[str, dict]
    path: Path

    def profile(self, name: str | None = None) -> Profile:
        chosen = name or self.default_profile
        if chosen not in self.profiles:
            known = ", ".join(sorted(self.profiles))
            raise KeyError(f"unknown profile {chosen!r}; known: {known}")
        return self.profiles[chosen]


def load_models(path: Path | str | None = None) -> ModelsFile:
    file_path = Path(path) if path else _DEFAULT_FILE
    raw = tomllib.loads(file_path.read_text(encoding="utf-8"))
    profiles: dict[str, Profile] = {}
    for name, body in (raw.get("profiles") or {}).items():
        brain = body.get("brain_settings") or {}
        ears = body.get("ears_settings") or {}
        mouth = body.get("mouth_settings") or {}
        speech = body.get("speech") or {}
        profiles[name] = Profile(
            name=name,
            description=str(body.get("description") or "").strip(),
            ears=str(body.get("ears") or "text"),
            brain=str(body.get("brain") or "scripted"),
            mouth=str(body.get("mouth") or "text"),
            requires_gpu=bool(body.get("requires_gpu", False)),
            requires_audio_hardware=bool(body.get("requires_audio_hardware", False)),
            run_status=str(body.get("run_status") or "unverified"),
            brain_base_url=str(brain.get("base_url") or ""),
            brain_model=str(brain.get("model") or ""),
            brain_timeout_s=float(brain.get("timeout_s") or 120),
            brain_max_tokens=int(brain.get("max_tokens") or 512),
            brain_think=bool(brain["think"]) if "think" in brain else False,
            brain_max_sentences=(
                int(brain["max_sentences"]) if "max_sentences" in brain else 3
            ),
            ears_base_url=str(ears.get("base_url") or ""),
            ears_model=str(ears.get("model") or ""),
            ears_language=str(ears.get("language") or ""),
            ears_device=str(ears.get("device") or ""),
            ears_hotwords=str(ears.get("hotwords") or ""),
            ears_initial_prompt=str(ears.get("initial_prompt") or ""),
            mouth_base_url=str(mouth.get("base_url") or ""),
            speech_model_id=str(speech.get("model_id") or ""),
            speech_dtype=str(speech.get("dtype") or ""),
        )
    known = raw.get("known_models") or {}
    return ModelsFile(
        default_profile=str(raw.get("default_profile") or "text"),
        profiles=profiles,
        known_brains=dict(known.get("brains") or {}),
        known_speech=dict(known.get("speech") or {}),
        path=file_path,
    )
