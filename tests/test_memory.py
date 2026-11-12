"""Store durability and the extractor, without a model."""

from __future__ import annotations

import json
import os
import re

import pytest

from hyper_ambient_maxi.memory.extract import extract
from hyper_ambient_maxi.memory.packet import build_packet, render
from hyper_ambient_maxi.memory.recall import context_block, recall
from hyper_ambient_maxi.memory.store import (
    MemoryStore,
    MemoryWriteError,
    ProposedFact,
    Thread,
)

DIAGNOSIS = (
    "My mum was diagnosed with breast cancer last week, and I'm scared."
)
JOB = "I just got a job at Airbus."
RENAME = "I renamed him to Biscuit."
JOB_UPDATE = "I left Airbus and I work at Safran now."


def _commit(store: MemoryStore, text: str):
    extracted = extract(text)
    return store.commit(
        extracted.facts,
        source_agent="maxi-text",
        conversation_id="test",
    )


def test_commit_fsyncs_and_survives_reopen(tmp_path, monkeypatch):
    calls = {"n": 0}
    real = os.fsync

    def spy(fd):
        calls["n"] += 1
        return real(fd)

    monkeypatch.setattr(os, "fsync", spy)
    store = MemoryStore(tmp_path)
    receipt = _commit(store, "My dog is called Pixel.")
    assert receipt.verified
    assert receipt.stored == 1
    assert calls["n"] >= 1
    assert "Pixel" in (tmp_path / "journal.jsonl").read_text(encoding="utf-8")

    (tmp_path / "state.json").unlink()
    reopened = MemoryStore(tmp_path)
    fact = reopened.slot("fact:dog-name")
    assert fact is not None
    assert fact.value == "Pixel"
    assert fact.source_agent == "maxi-text"
    assert fact.source == "steward-extract"
    assert fact.conversation_id == "test"


def test_snapshot_lie_is_repaired_from_the_journal(tmp_path):
    store = MemoryStore(tmp_path)
    _commit(store, JOB)
    (tmp_path / "state.json").write_text(
        json.dumps({"version": 1, "journal_bytes": 10**9, "memories": []}),
        encoding="utf-8",
    )
    reopened = MemoryStore(tmp_path)
    fact = reopened.slot("job:employer")
    assert fact is not None
    assert fact.value == "Airbus"


def test_dropped_journal_write_is_not_reported_as_saved(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path)
    monkeypatch.setattr(store, "_append_journal", lambda records: None)
    with pytest.raises(MemoryWriteError):
        _commit(store, JOB)
    assert MemoryStore(tmp_path).slot("job:employer") is None


def test_repeat_is_idempotent(tmp_path):
    store = MemoryStore(tmp_path)
    first = _commit(store, "My dog is called Pixel.")
    second = _commit(store, "My dog is called Pixel.")
    assert first.stored == 1
    assert second.stored == 0
    assert second.already
    assert len([f for f in store.active() if f.key == "fact:dog-name"]) == 1


def test_pronoun_memory_is_refused(tmp_path):
    store = MemoryStore(tmp_path)
    extracted = extract("Please remember that.")
    assert extracted.rejected_pronoun
    assert extracted.facts == []
    receipt = store.commit(
        [
            ProposedFact(
                key="fact:that",
                type="personal_fact",
                summary="Remembered: that",
                value="that",
                evidence="Please remember that.",
            )
        ],
        source_agent="maxi-text",
        conversation_id="test",
    )
    assert receipt.stored == 0
    assert receipt.dropped_pronouns == 1
    assert store.active() == []


def test_codename_does_not_create_a_that_fact():
    extracted = extract("My test codename is Bluebird. Please remember it.")
    keys = {fact.key for fact in extracted.facts}
    assert keys == {"fact:test-codename"}
    assert extracted.facts[0].value == "Bluebird"
    assert extracted.facts[0].summary == "User's test codename is Bluebird"


def test_explicit_remember_keeps_the_clause_not_the_pronoun():
    extracted = extract("Remember that the gate code is 1234.")
    assert len(extracted.facts) == 1
    fact = extracted.facts[0]
    assert fact.value == "the gate code is 1234"
    assert "that" != fact.value
    assert not fact.key.endswith(":that")


def test_diagnosis_job_and_rename_extract():
    diagnosis = extract(DIAGNOSIS)
    assert diagnosis.facts[0].key == "health:mum-diagnosis"
    assert diagnosis.facts[0].value == "breast cancer"
    assert "scared" not in diagnosis.facts[0].value

    job = extract(JOB)
    assert [(fact.key, fact.value) for fact in job.facts] == [("job:employer", "Airbus")]

    role = extract("I just got a job at Airbus as a systems engineer.")
    assert {fact.key: fact.value for fact in role.facts} == {
        "job:employer": "Airbus",
        "job:role": "systems engineer",
    }

    rename = extract(RENAME)
    assert rename.facts[0].key == "fact:dog-name"
    assert rename.facts[0].value == "Biscuit"
    for line in (
        "Actually I renamed him Biscuit",
        "I renamed him Biscuit",
        "His name is Biscuit now.",
        "We call him Biscuit now.",
        "Je l'ai renommé Biscuit",
        "Il s'appelle maintenant Biscuit",
        "On l'appelle Biscuit",
    ):
        found = extract(line)
        assert found.facts, line
        assert found.facts[0].key == "fact:dog-name", line
        assert found.facts[0].value == "Biscuit", line
        assert found.durable_unparsed is False, line


def test_unparsed_job_cue_is_kept_verbatim():
    text = (
        "Something changed about my employment situation at Airbus, "
        "please keep it."
    )
    extracted = extract(text)
    assert extracted.durable_unparsed
    assert extracted.facts[0].type == "episode"
    assert extracted.facts[0].value == text
    assert all(fact.key not in {"fact:it", "fact:that"} for fact in extracted.facts)


def test_question_is_not_stored_as_a_memory():
    extracted = extract("What's my dog's name?")
    assert extracted.facts == []
    assert extract("What is the capital of France?").facts == []


def test_preference_and_oat_milk():
    extracted = extract("I prefer oat milk in my coffee.")
    assert extracted.facts[0].key == "preference:oat-milk-in-my-coffee"
    assert "oat milk" in extracted.facts[0].value


def test_packet_orders_priority_without_printing_it(tmp_path):
    low = Thread(
        id="thread:note",
        title="Note",
        summary="Dentist on Tuesday",
        status="open",
        priority="low",
        created_at="2026-11-01T00:00:00.000Z",
    )
    high = Thread(
        id="thread:health-mum",
        title="mum diagnosis",
        summary="User's mum was diagnosed with diabetes",
        status="open",
        priority="high",
        created_at="2026-11-02T00:00:00.000Z",
    )
    text = render([], [low, high], None, [])
    assert text.index("diabetes") < text.index("Dentist")
    _assert_no_rank_words(text)

    store = MemoryStore(tmp_path)
    for line in (
        "My mum was diagnosed with diabetes last week.",
        "I just got a job at Airbus.",
    ):
        extracted = extract(line)
        store.commit(
            extracted.facts,
            threads=extracted.threads,
            source_agent="maxi-text",
            conversation_id="test",
        )
    packet = build_packet(store, "what do you remember about me?")
    # The job and diagnosis threads repeat the fact summaries, so they
    # stay out of the prompt. Priority order is checked on render() above.
    assert packet.text.count("Airbus") == 1
    assert packet.text.count("diabetes") == 1
    assert "Open threads:" not in packet.text
    _assert_no_rank_words(packet.text)


def _assert_no_rank_words(text: str) -> None:
    lowered = text.lower()
    assert not re.search(r"\[(?:high|low|normal|medium|priority)\]", lowered)
    for word in ("high", "low", "priority", "importance", "salience", "relevance"):
        assert re.search(rf"\b{word}\b", lowered) is None, word


def test_prompt_block_has_no_scores(tmp_path):
    store = MemoryStore(tmp_path)
    _commit(store, DIAGNOSIS)
    _commit(store, "My dog is called Pixel.")
    block = context_block(recall(store, "what do you remember about me?"))
    assert "breast cancer" in block
    assert "Pixel" in block
    for banned in ("importance", "relevance", "recency", "0.95", "0.80"):
        assert banned not in block
    raw = (tmp_path / "state.json").read_text(encoding="utf-8")
    assert "importance" in raw


def test_recall_ignores_superseded_values(tmp_path):
    store = MemoryStore(tmp_path)
    _commit(store, "My dog is called Pixel.")
    _commit(store, RENAME)
    remembered = recall(store, "what's my dog's name?")
    assert [fact.value for fact in remembered] == ["Biscuit"]
    assert store.history("fact:dog-name")[0].value == "Pixel"
    assert store.history("fact:dog-name")[0].status == "superseded"


def test_sister_is_a_person_slot_not_an_episode():
    extracted = extract("My sister is called Léa.")
    assert [(fact.key, fact.value) for fact in extracted.facts] == [("person:sister", "Léa")]
    assert extracted.durable_unparsed is False
    assert not any(fact.key.startswith("episode:") for fact in extracted.facts)
    french = extract("Ma sœur s'appelle Léa.")
    assert [(fact.key, fact.value) for fact in french.facts] == [("person:sister", "Léa")]
    named = extract("My sister's name is Ada.")
    assert named.facts[0].key == "person:sister"
    assert named.facts[0].value == "Ada"


def test_earlier_dog_name_reaches_the_packet(tmp_path):
    from hyper_ambient_maxi.memory.packet import build_packet

    store = MemoryStore(tmp_path)
    _commit(store, "My dog is called Pixel.")
    _commit(store, RENAME)
    current = build_packet(store, "What's my dog's name?")
    assert "Biscuit" in current.text
    assert "Earlier values:" not in current.text
    assert "Pixel" not in current.text
    past = build_packet(store, "What did I used to call my dog?")
    assert "Earlier values:" in past.text
    assert "previously called Pixel" in past.text
    assert "Biscuit" in past.text


def test_stt_use_to_still_recalls_the_earlier_dog_name(tmp_path):
    from hyper_ambient_maxi.memory.packet import build_packet, asks_about_past

    store = MemoryStore(tmp_path)
    _commit(store, "My dog is called Pixel.")
    _commit(store, RENAME)
    assert asks_about_past("What did I use to call my dog?")
    past = build_packet(store, "What did I use to call my dog?")
    assert "previously called Pixel" in past.text
