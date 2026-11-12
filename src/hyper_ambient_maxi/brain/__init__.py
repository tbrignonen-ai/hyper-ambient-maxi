"""Reply generators. Memory is already committed before these run."""

from hyper_ambient_maxi.brain.openai_compat import OpenAICompatBrain
from hyper_ambient_maxi.brain.scripted import ScriptedBrain

__all__ = ["OpenAICompatBrain", "ScriptedBrain"]
