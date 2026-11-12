"""Local embeddings are optional. Lexical overlap still decides when they fail."""

from hyper_ambient_maxi.brain.scripted import ScriptedBrain
from hyper_ambient_maxi.loop import run_turn
from hyper_ambient_maxi.memory.embed import HashingEmbedder, cosine
from hyper_ambient_maxi.memory.salience import select_memories
from hyper_ambient_maxi.memory.store import Fact, MemoryStore


class _MapEmbedder:
    def embed(self, text: str) -> list[float]:
        folded = text.lower()
        if any(word in folded for word in ("health", "diagnos", "breast", "mum")):
            return [1.0, 0.0]
        return [0.0, 1.0]


class _Boom:
    def embed(self, text: str) -> list[float]:
        raise RuntimeError("no network")


def _fact() -> Fact:
    return Fact(
        id="mem_health",
        key="health:mum-diagnosis",
        type="health",
        summary="User's mum was diagnosed with breast cancer",
        value="breast cancer",
        status="active",
        created_at="2026-11-11T12:00:00.000000Z",
        source="test",
        source_agent="maxi-text",
        conversation_id="embed",
        evidence="",
        importance=0.95,
    )


def test_hashing_embedder_is_deterministic_and_local():
    embedder = HashingEmbedder()
    first = embedder.embed("breast cancer")
    assert first == embedder.embed("breast cancer")
    assert len(first) == 64
    assert cosine(first, embedder.embed("a completely different topic")) < 0.99


def test_embeddings_recover_a_paraphrase_and_a_failure_stays_lexical():
    fact = _fact()
    kwargs = {
        "mode": "friend",
        "limit": 3,
        "profile": False,
        "open_ids": set(),
    }
    assert select_memories([fact], "how is her health", **kwargs) == []
    found = select_memories([fact], "how is her health", embedder=_MapEmbedder(), **kwargs)
    assert [item.key for item in found] == ["health:mum-diagnosis"]
    assert select_memories([fact], "how is her health", embedder=_Boom(), **kwargs) == []
    dog = Fact(
        id="mem_dog",
        key="fact:dog-name",
        type="personal_fact",
        summary="User's dog is called Pixel",
        value="Pixel",
        status="active",
        created_at="2026-11-11T12:00:00.000000Z",
        source="test",
        source_agent="maxi-text",
        conversation_id="embed",
        evidence="",
        importance=0.8,
    )
    still = select_memories([dog], "what is my dog's name", embedder=_Boom(), **kwargs)
    assert [item.key for item in still] == ["fact:dog-name"]


def test_hash_flag_does_not_drop_a_lexical_hit(tmp_path, monkeypatch):
    monkeypatch.setenv("HAM_EMBEDDINGS", "hash")
    store = MemoryStore(tmp_path)
    brain = ScriptedBrain()
    run_turn(store, brain, "My dog is called Pixel.")
    reply = run_turn(store, brain, "What's my dog's name?")
    assert reply.reply == "Your dog's name is Pixel."
