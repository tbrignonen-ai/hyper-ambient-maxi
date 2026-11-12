"""The conversation loop keeps the writes the 2026-11-05 run lost."""

from __future__ import annotations

import json

from hyper_ambient_maxi.brain.openai_compat import OpenAICompatBrain
from hyper_ambient_maxi.brain.scripted import ScriptedBrain
from hyper_ambient_maxi.loop import run_turn
from hyper_ambient_maxi.memory.store import MemoryStore, MemoryWriteError

DIAGNOSIS = (
    "My mum was diagnosed with breast cancer last week, and I'm scared."
)


def _turn(store, text, brain=None):
    return run_turn(
        store,
        brain or ScriptedBrain(),
        text,
        conversation_id="loop",
        source_agent="maxi-text",
    )


def test_diagnosis_is_saved_while_the_reply_is_supportive(tmp_path):
    store = MemoryStore(tmp_path)
    result = _turn(store, DIAGNOSIS)
    assert result.memory_ok
    assert result.receipt is not None
    assert result.receipt.stored == 1
    assert "sorry" in result.reply.lower()
    assert "breast cancer" in result.reply.lower()
    disk = MemoryStore(tmp_path)
    fact = disk.slot("health:mum-diagnosis")
    assert fact is not None
    assert fact.value == "breast cancer"
    assert fact.source_agent == "maxi-text"


def test_airbus_job_is_not_a_stored_zero(tmp_path):
    store = MemoryStore(tmp_path)
    result = _turn(store, "I just got a job at Airbus.")
    assert result.receipt is not None
    assert result.receipt.stored == 1
    assert result.receipt.verified
    assert MemoryStore(tmp_path).slot("job:employer").value == "Airbus"

    update = _turn(store, "I left Airbus and I work at Safran now.")
    assert update.receipt is not None
    assert update.receipt.stored == 1
    assert update.receipt.corrections
    disk = MemoryStore(tmp_path)
    assert disk.slot("job:employer").value == "Safran"
    old = [fact for fact in disk.history("job:employer") if fact.value == "Airbus"]
    assert old and old[0].status == "superseded"
    answer = _turn(disk, "Where do I work?")
    assert answer.reply == "You work at Safran."


def test_pet_rename_supersedes_and_is_recalled_after_reopen(tmp_path):
    store = MemoryStore(tmp_path)
    _turn(store, "My dog is called Pixel.")
    renamed = _turn(store, "I renamed him to Biscuit.")
    assert "Biscuit" in renamed.reply
    assert "replaces" in renamed.reply
    disk = MemoryStore(tmp_path)
    assert disk.slot("fact:dog-name").value == "Biscuit"
    answer = _turn(disk, "What's my dog's name?")
    assert answer.reply == "Your dog's name is Biscuit."
    prior = _turn(disk, "What did I used to call my dog?")
    assert prior.reply == "You used to call your dog Pixel."
    profile = _turn(disk, "What do you remember about me?")
    assert "Biscuit" in profile.reply
    assert "Pixel" not in profile.reply


def test_kept_facts_still_persist_together(tmp_path):
    store = MemoryStore(tmp_path)
    _turn(store, "My dog is called Pixel.")
    _turn(store, "I prefer oat milk in my coffee.")
    _turn(store, DIAGNOSIS)
    _turn(store, "I just got a job at Airbus.")
    _turn(store, "I renamed him to Biscuit.")
    _turn(store, "Please remember that.")
    disk = MemoryStore(tmp_path)
    active = {fact.key: fact.value for fact in disk.active()}
    assert active["fact:dog-name"] == "Biscuit"
    assert active["health:mum-diagnosis"] == "breast cancer"
    assert active["job:employer"] == "Airbus"
    assert "oat milk" in active["preference:oat-milk-in-my-coffee"]
    assert "fact:that" not in active
    trivia = _turn(disk, "What is the capital of France?")
    assert trivia.receipt is not None
    assert trivia.receipt.stored == 0
    assert "don't have that stored" in trivia.reply
    assert len(MemoryStore(tmp_path).active()) == len(disk.active())


def test_unparsed_employment_sentence_is_still_recalled(tmp_path):
    store = MemoryStore(tmp_path)
    text = (
        "Something changed about my employment situation at Airbus, "
        "please keep it."
    )
    result = _turn(store, text)
    assert result.receipt is not None
    assert result.receipt.stored == 1
    assert result.receipt.facts[0].key.startswith("episode:")
    answer = _turn(MemoryStore(tmp_path), "What happened with Airbus?")
    assert "employment situation at Airbus" in answer.reply


def test_failed_commit_is_not_claimed(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path)

    def boom(*args, **kwargs):
        raise MemoryWriteError("disk full")

    monkeypatch.setattr(store, "commit", boom)
    result = _turn(store, "I just got a job at Airbus.")
    assert result.memory_ok is False
    assert "could not confirm" in result.reply
    assert MemoryStore(tmp_path).active() == []


def test_brain_failure_does_not_drop_the_write(tmp_path):
    class Boom:
        name = "openai-compat"

        def reply(self, *args, **kwargs):
            raise OSError("connection refused")

    store = MemoryStore(tmp_path)
    result = _turn(store, "My dog is called Pixel.", brain=Boom())
    assert result.memory_ok
    assert result.brain == "scripted-fallback"
    assert "Pixel" in result.reply
    assert MemoryStore(tmp_path).slot("fact:dog-name").value == "Pixel"


def test_turn_log_is_one_record_per_turn(tmp_path):
    store = MemoryStore(tmp_path)
    _turn(store, "My dog is called Pixel.")
    _turn(store, "What's my dog's name?")
    lines = (tmp_path / "turns.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert records[0]["memory_ok"] is True
    assert records[0]["source_agent"] == "maxi-text"
    assert records[0]["stored_ids"]
    assert records[1]["stored_ids"] == []


def test_openai_client_never_speaks_reasoning_and_keeps_the_key_out(monkeypatch):
    import urllib.request

    from hyper_ambient_maxi.brain.openai_compat import clean_reply
    from hyper_ambient_maxi.memory.store import Receipt

    bodies = []
    thought = "The user had a dog named Pixel. I will think for a long time before answering."

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(self.payload).encode()

    def fake_urlopen(request, timeout=0):
        bodies.append(json.loads(request.data.decode()))
        headers = {key.lower(): value for key, value in request.header_items()}
        assert headers["authorization"] == "Bearer local-test-key"
        assert b"local-test-key" not in request.data
        if len(bodies) == 1:
            return Response(
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": None, "reasoning_content": thought},
                        }
                    ]
                }
            )
        return Response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": (
                                "<think>still a secret plan</think>"
                                "Your dog's name is Biscuit.\n"
                                "User: What else?\n"
                                "Assistant: More."
                            ),
                            "reasoning_content": thought,
                        },
                    }
                ]
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("HAM_BRAIN_BASE_URL", "http://127.0.0.1:9090/v1")
    brain = OpenAICompatBrain(
        base_url="http://127.0.0.1:8080/v1",
        model="lfm3-1.7b",
        api_key="local-test-key",
        max_tokens=200,
    )
    assert brain.base_url == "http://127.0.0.1:9090/v1"
    text = brain.reply("What's my dog's name?", "<memory>\n- Biscuit\n</memory>", Receipt(0, True), [])
    assert text == "Your dog's name is Biscuit."
    assert "Pixel" not in text
    assert "secret" not in text
    assert "What else" not in text
    assert len(bodies) == 2
    assert bodies[0]["max_tokens"] == 96
    assert bodies[0]["chat_template_kwargs"]["enable_thinking"] is False
    assert bodies[0]["chat_template_kwargs"]["thinking"] is False
    assert bodies[0]["reasoning_effort"] == "none"
    assert bodies[0]["think"] is False
    assert bodies[0]["reasoning_budget"] == 0
    assert bodies[1]["max_tokens"] == 192
    assert bodies[1]["chat_template_kwargs"]["enable_thinking"] is False
    assert bodies[1]["reasoning_effort"] == "none"
    assert bodies[1]["think"] is False
    assert "\nUser:" in bodies[0]["stop"]
    assert clean_reply("Hello</think> there") == "Hello there"


def test_llama_local_disables_thinking_on_the_first_request(monkeypatch):
    import urllib.request

    from hyper_ambient_maxi.cli import build_brain
    from hyper_ambient_maxi.config import load_models
    from hyper_ambient_maxi.memory.store import Receipt

    bodies = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "Your dog's name is Biscuit."}}]}
            ).encode()

    def fake_urlopen(request, timeout=0):
        bodies.append(json.loads(request.data.decode()))
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.delenv("HAM_BRAIN_BASE_URL", raising=False)
    brain = build_brain(load_models().profile("llama-local"))
    assert brain.think is False
    text = brain.reply("What's my dog's name?", "<memory>\n- Biscuit\n</memory>", Receipt(0, True), [])
    assert text == "Your dog's name is Biscuit."
    assert len(bodies) == 1
    assert bodies[0]["max_tokens"] == 96
    assert bodies[0]["chat_template_kwargs"]["enable_thinking"] is False
    assert bodies[0]["think"] is False
    assert bodies[0]["reasoning_budget"] == 0
    assert bodies[0]["reasoning_effort"] == "none"


def test_think_true_leaves_the_first_request_alone(monkeypatch):
    import urllib.request

    from hyper_ambient_maxi.memory.store import Receipt

    bodies = []

    class Response:
        def __init__(self, content):
            self.content = content

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": self.content, "reasoning_content": "hidden"}}]}
            ).encode()

    def fake_urlopen(request, timeout=0):
        bodies.append(json.loads(request.data.decode()))
        if len(bodies) == 1:
            return Response(None)
        return Response("Your dog's name is Biscuit.")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    brain = OpenAICompatBrain(think=True, max_tokens=200)
    text = brain.reply("What's my dog's name?", "", Receipt(0, True), [])
    assert text == "Your dog's name is Biscuit."
    assert "hidden" not in text
    assert "enable_thinking" not in json.dumps(bodies[0])
    assert bodies[1]["chat_template_kwargs"]["enable_thinking"] is False


def test_think_off_caps_spoken_tokens_and_think_on_keeps_the_budget(monkeypatch):
    import urllib.request

    from hyper_ambient_maxi.brain.openai_compat import SPEAK_MAX_TOKENS
    from hyper_ambient_maxi.memory.store import Receipt

    bodies = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "Okay, Pixel it is."}}]}
            ).encode()

    def fake_urlopen(request, timeout=0):
        bodies.append(json.loads(request.data.decode()))
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    OpenAICompatBrain(max_tokens=512, think=False).reply(
        "My dog is called Pixel.", "", Receipt(0, True), []
    )
    assert bodies[-1]["max_tokens"] == SPEAK_MAX_TOKENS
    OpenAICompatBrain(max_tokens=32, think=False).reply(
        "My dog is called Pixel.", "", Receipt(0, True), []
    )
    assert bodies[-1]["max_tokens"] == 32
    OpenAICompatBrain(max_tokens=512, think=True).reply(
        "My dog is called Pixel.", "", Receipt(0, True), []
    )
    assert bodies[-1]["max_tokens"] == 512


def test_empty_content_after_retry_does_not_voice_the_thought(monkeypatch):
    import urllib.request

    from hyper_ambient_maxi.memory.store import Receipt

    thought = "unspoken chain of thought about Pixel"

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": "<think>only a thought</think>", "reasoning_content": thought},
                        }
                    ]
                }
            ).encode()

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout=0: Response())
    brain = OpenAICompatBrain(max_tokens=200)
    try:
        brain.reply("What's my dog's name?", "", Receipt(0, True), [])
    except RuntimeError as exc:
        assert "unspoken" not in str(exc)
        assert "Pixel" not in str(exc)
    else:
        raise AssertionError("empty content was spoken")


# The live LFM3 reply glued a second answer on with no space and no role tag.
_GLUED_REPLY = "Let me know what it involves.That\u2019s awesome news!"
_GLUED_REPLY_ASCII = "Let me know what it involves.That's awesome news!"


def test_glued_second_answer_keeps_the_first(monkeypatch):
    import urllib.request

    from hyper_ambient_maxi.brain.openai_compat import clean_reply
    from hyper_ambient_maxi.cli import build_brain
    from hyper_ambient_maxi.config import load_models
    from hyper_ambient_maxi.memory.store import Receipt

    for raw in (_GLUED_REPLY, _GLUED_REPLY_ASCII):
        assert clean_reply(raw) == "Let me know what it involves."
        assert "awesome" not in clean_reply(raw)
    longer = "I saved the job. " + _GLUED_REPLY + " You will do great."
    assert clean_reply(longer) == "I saved the job. Let me know what it involves."
    spaced = "Let me know what it involves. That\u2019s awesome news!"
    assert clean_reply(spaced) == "Let me know what it involves."
    assert clean_reply("Your dog is Biscuit. Hello, how are you?") == "Your dog is Biscuit."
    assert clean_reply("Hello. Your dog is Biscuit. He is well.") == (
        "Hello. Your dog is Biscuit. He is well."
    )
    assert clean_reply("Dr. Smith is your contact. He works at Airbus.") == (
        "Dr. Smith is your contact. He works at Airbus."
    )
    four = "Alpha is set. Beta is set. Gamma is set. Delta is set."
    assert clean_reply(four) == "Alpha is set. Beta is set. Gamma is set."
    assert clean_reply(four, max_sentences=2) == "Alpha is set. Beta is set."

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": _GLUED_REPLY}}]}
            ).encode()

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout=0: Response())
    monkeypatch.delenv("HAM_BRAIN_BASE_URL", raising=False)
    brain = build_brain(load_models().profile("llama-local"))
    assert brain.max_sentences == 3
    spoken = brain.reply("I just got a job at Airbus.", "", Receipt(1, True), [])
    assert spoken == "Let me know what it involves."
    assert "awesome" not in spoken


def test_the_user_is_not_addressed_as_maxi(monkeypatch, tmp_path):
    import urllib.request

    from hyper_ambient_maxi.brain.openai_compat import clean_reply, system_prompt
    from hyper_ambient_maxi.memory.store import Receipt

    prompt = system_prompt("Thomas")
    assert prompt.startswith("You are Maxi")
    assert "The user is Thomas" in prompt
    assert "Never call the user Maxi" in prompt
    assert "No emoji" in prompt
    assert "do not say it is missing" in system_prompt("Ada")
    assert "Do not repeat the user's sentence" in prompt
    assert "not Got it" in prompt
    assert "confirm the stored value" in prompt
    assert "say your dog, not my dog" in prompt
    assert "do not mention memory" in prompt
    assert "The user is Ada" in system_prompt("Ada")
    untitled = system_prompt("")
    assert "The user is Thomas" not in untitled
    assert "name is not stored yet" in untitled
    assert "do not call them Thomas" in untitled
    assert clean_reply("Hi Maxi") == "Hi"
    assert clean_reply("Hi Maxi, your dog is Biscuit.") == "Hi, your dog is Biscuit."
    assert clean_reply("Hello Maxi. He is well.") == "Hello. He is well."
    assert clean_reply("I am Maxi. Your dog is Biscuit.") == "I am Maxi. Your dog is Biscuit."
    assert clean_reply("Your dog is **Biscuit**.") == "Your dog is Biscuit."
    assert clean_reply("Hello 😊 Your dog is Biscuit.") == "Hello Your dog is Biscuit."
    assert clean_reply("`Biscuit` is the name.") == "Biscuit is the name."

    bodies = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "Hi Maxi, Biscuit is here."}}]}
            ).encode()

    def fake_urlopen(request, timeout=0):
        bodies.append(json.loads(request.data.decode()))
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.delenv("HAM_BRAIN_BASE_URL", raising=False)
    store = MemoryStore(tmp_path)
    run_turn(store, ScriptedBrain(), "My name is Ada.")
    brain = OpenAICompatBrain(base_url="http://127.0.0.1:9/v1", model="local")
    result = run_turn(store, brain, "Hello")
    assert result.reply == "Hi, Biscuit is here."
    system = bodies[0]["messages"][0]["content"]
    assert "The user is Ada" in system
    assert "Never call the user Maxi" in system
    assert brain.reply("Hello", "", Receipt(0, True), [], user_name="Thomas").startswith("Hi")


def test_sister_name_is_stored_and_recalled(tmp_path):
    store = MemoryStore(tmp_path)
    saved = _turn(store, "My sister is called Léa.")
    assert saved.memory_ok
    assert store.slot("person:sister").value == "Léa"
    asked = _turn(store, "What's my sister's name?")
    assert asked.reply == "Your sister is called Léa."


def test_openai_brain_sees_superseded_dog_name(monkeypatch, tmp_path):
    import urllib.request

    from hyper_ambient_maxi.brain.openai_compat import OpenAICompatBrain

    bodies = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "You used to call your dog Pixel."}}]}
            ).encode()

    def fake_urlopen(request, timeout=0):
        bodies.append(json.loads(request.data.decode()))
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    store = MemoryStore(tmp_path)
    _turn(store, "My dog is called Pixel.")
    _turn(store, "I renamed him to Biscuit.")
    brain = OpenAICompatBrain(base_url="http://127.0.0.1:9/v1", model="local")
    result = run_turn(store, brain, "What did I used to call my dog?")
    assert result.reply == "You used to call your dog Pixel."
    system = bodies[0]["messages"][0]["content"]
    assert "previously called Pixel" in system
    assert "Pixel" in system
    assert "Biscuit" in system


def test_default_prompt_does_not_name_the_user_thomas(tmp_path, monkeypatch):
    import urllib.request

    from hyper_ambient_maxi.brain.openai_compat import OpenAICompatBrain

    bodies = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "Hello."}}]}).encode()

    def fake_urlopen(request, timeout=0):
        bodies.append(json.loads(request.data.decode()))
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    store = MemoryStore(tmp_path)
    brain = OpenAICompatBrain(base_url="http://127.0.0.1:9/v1", model="local")
    run_turn(store, brain, "Hello")
    system = bodies[0]["messages"][0]["content"]
    assert "The user is Thomas" not in system
    assert "name is not stored yet" in system
    assert "No stored memories were selected" not in system
    assert "do not mention memory" in system


def test_polish_reply_rewrites_got_it_and_drops_echo_and_hedge():
    from hyper_ambient_maxi.brain.openai_compat import polish_reply
    from hyper_ambient_maxi.memory.store import Fact, Receipt

    assert polish_reply("Got it, I've noted that.", "I renamed him to Biscuit.") == (
        "Okay. I've noted that."
    )
    assert polish_reply("Got it.", "My dog is called Pixel.") == "Okay."
    assert polish_reply(
        "Okay. I've noted that your dog is named Pixel. Okay.",
        "My dog is called Pixel.",
    ) == "Okay. I've noted that your dog is named Pixel."
    echoed = polish_reply(
        "I renamed him to Biscuit.",
        "I renamed him to Biscuit.",
        receipt=Receipt(
            stored=1,
            verified=True,
            facts=[
                Fact(
                    id="1",
                    key="fact:dog-name",
                    type="personal_fact",
                    summary="User's dog is called Biscuit",
                    value="Biscuit",
                    status="active",
                    created_at="2026-11-12T00:00:00.000Z",
                    source="test",
                    source_agent="test",
                    conversation_id="t",
                    evidence="I renamed him to Biscuit.",
                    importance=0.8,
                )
            ],
        ),
    )
    assert echoed == "Okay, Biscuit it is."
    assert "I renamed him to Biscuit." not in echoed
    # A greeting echo is allowed.
    assert polish_reply("Hello.", "Hello.") == "Hello."
    # "Your dog" is not an echo of "My dog".
    assert polish_reply("Your dog is called Pixel.", "My dog is called Pixel.") == (
        "Your dog is called Pixel."
    )
    hedged = polish_reply(
        "I do not have that stored. The previous name was Pixel but now it is Biscuit.",
        "What did I used to call my dog?",
        context="<memory>\nEarlier values:\n- The dog was previously called Pixel\n</memory>",
    )
    assert hedged == "The previous name was Pixel but now it is Biscuit."
    assert "do not have" not in hedged.lower()
    # A question read back is also an echo.
    asked = polish_reply(
        "What did I use to call your dog?",
        "What did I used to call my dog?",
        notes=["Pixel"],
        context="<memory>\nEarlier values:\n- The dog was previously called Pixel\n</memory>",
    )
    assert asked == "That was previously Pixel."
    assert "what did i" not in asked.lower()
    # A genuine miss, with no memory, is left alone.
    assert polish_reply("I do not have that stored.", "What is my postal code?") == (
        "I do not have that stored."
    )


def test_polish_reply_confirms_saved_value_and_fixes_perspective():
    from hyper_ambient_maxi.brain.openai_compat import polish_reply
    from hyper_ambient_maxi.memory.store import Fact, Receipt

    biscuit = Fact(
        id="1",
        key="fact:dog-name",
        type="personal_fact",
        summary="User's dog is called Biscuit",
        value="Biscuit",
        status="active",
        created_at="2026-11-12T00:00:00.000Z",
        source="test",
        source_agent="test",
        conversation_id="t",
        evidence="I renamed him to Biscuit.",
        importance=0.8,
    )
    receipt = Receipt(stored=1, verified=True, facts=[biscuit])
    assert polish_reply("I'll remember that.", "I renamed him to Biscuit.", receipt=receipt) == (
        "Okay, Biscuit it is."
    )
    assert polish_reply("Okay, Biscuit it is.", "I renamed him to Biscuit.", receipt=receipt) == (
        "Okay, Biscuit it is."
    )
    assert polish_reply(
        "That sounds lovely! I can tell Pixel is a great companion.",
        "My dog is called Pixel.",
        receipt=Receipt(
            stored=1,
            verified=True,
            facts=[
                Fact(
                    id="2",
                    key="fact:dog-name",
                    type="personal_fact",
                    summary="User's dog is called Pixel",
                    value="Pixel",
                    status="active",
                    created_at="2026-11-12T00:00:00.000Z",
                    source="test",
                    source_agent="test",
                    conversation_id="t",
                    evidence="My dog is called Pixel.",
                    importance=0.8,
                )
            ],
        ),
    ) == "Okay, Pixel it is."
    assert polish_reply(
        "Okay. I've updated his name to Biscuit.",
        "I renamed him to Biscuit.",
        receipt=receipt,
    ) == "Okay. I've updated his name to Biscuit."
    # A leftover save ack must not trail a recall answer.
    assert polish_reply(
        "Your dog was previously called Pixel. Okay, Biscuit it is.",
        "What did I used to call my dog?",
    ) == "Your dog was previously called Pixel."
    assert polish_reply(
        "You asked earlier that your dog was previously called Pixel, so you used to call it Pixel. Okay, Pixel. It's saved.",
        "What did I used to call my dog?",
    ) == "Your dog was previously called Pixel, so you used to call it Pixel."
    assert polish_reply(
        "Your dog was previously called Pixel. Okay, then it is. I do not have any other stored details about your life right now.",
        "What did I used to call my dog?",
    ) == "Your dog was previously called Pixel."
    # A model that already names the value is left alone, but first person is fixed.
    assert polish_reply(
        "My dog was previously called Pixel.",
        "What did I used to call my dog?",
    ) == "Your dog was previously called Pixel."
    assert polish_reply(
        "Mon chien s'appelait Pixel.",
        "Comment s'appelait mon chien avant ?",
    ) == "Ton chien s'appelait Pixel."
    # Maxi's own first person stays.
    assert polish_reply("I'll remember that. I'm here.", "Thanks.") == (
        "I'll remember that. I'm here."
    )
    # A bare greeting must not talk about stored memories.
    assert polish_reply(
        "Hello. I do not have any stored memories for this turn.",
        "Hello.",
    ) == "Hello. I'm here."
    assert polish_reply(
        "Bonjour. Je n'ai pas de souvenir stocké.",
        "Bonjour.",
    ) == "Bonjour, je suis là."

    leaked = (
        "Okay, the user just told me their dog's name is Pixel. "
        "I already have that in memory from before."
    )
    pixel = Fact(
        id="3",
        key="fact:dog-name",
        type="personal_fact",
        summary="User's dog is called Pixel",
        value="Pixel",
        status="active",
        created_at="2026-11-12T00:00:00.000Z",
        source="test",
        source_agent="test",
        conversation_id="t",
        evidence="My dog is called Pixel.",
        importance=0.8,
    )
    assert polish_reply(
        leaked,
        "My dog is called Pixel.",
        receipt=Receipt(stored=1, verified=True, facts=[pixel]),
    ) == "Okay, Pixel it is."
    assert "the user" not in polish_reply(
        leaked,
        "My dog is called Pixel.",
        receipt=Receipt(stored=1, verified=True, facts=[pixel]),
    ).lower()
    planned = (
        "I just need to confirm it in my own words without repeating the sentence. "
        "I'll say something like: You told me you renamed him, and I have that noted as Biscuit."
    )
    assert polish_reply(
        planned,
        "I renamed him to Biscuit.",
        receipt=Receipt(
            stored=1,
            verified=True,
            facts=[
                Fact(
                    id="4",
                    key="fact:dog-name",
                    type="personal_fact",
                    summary="User's dog is called Biscuit",
                    value="Biscuit",
                    status="active",
                    created_at="2026-11-12T00:00:00.000Z",
                    source="test",
                    source_agent="test",
                    conversation_id="t",
                    evidence="I renamed him to Biscuit.",
                    importance=0.8,
                )
            ],
        ),
    ) == "Okay, Biscuit it is."
