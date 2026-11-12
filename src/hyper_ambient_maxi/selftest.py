"""The three writes the 2026-11-05 run lost, plus the two it kept.

Utterances are the harness sentences from that run where we have them.
The Airbus sentence is the class of line Mercury answered while storing
0; the original user wording was not in the notes, so the self-test uses
"I just got a job at Airbus." The rename is "I renamed him to Biscuit."
"""

from __future__ import annotations

import re
from pathlib import Path

from hyper_ambient_maxi.brain.scripted import ScriptedBrain
from hyper_ambient_maxi.loop import run_turn
from hyper_ambient_maxi.memory.packet import build_packet
from hyper_ambient_maxi.memory.store import MemoryStore

_UTTERANCES = (
    "My dog is called Pixel.",
    "I prefer oat milk in my coffee.",
    "My mum was diagnosed with breast cancer last week, and I'm scared.",
    "I just got a job at Airbus.",
    "I renamed him to Biscuit.",
    "Please remember that.",
)


def run_selftest(home: Path) -> int:
    store = MemoryStore(home)
    brain = ScriptedBrain()
    for utterance in _UTTERANCES:
        result = run_turn(
            store,
            brain,
            utterance,
            conversation_id="selftest",
            source_agent="maxi-text",
        )
        if not result.memory_ok:
            print(f"FAIL {utterance} {result.error}")
            return 1

    disk = MemoryStore(home)
    problems: list[str] = []
    dog = disk.slot("fact:dog-name")
    health = disk.slot("health:mum-diagnosis")
    job = disk.slot("job:employer")
    preferences = [f for f in disk.active() if f.key.startswith("preference:")]
    if dog is None or dog.value != "Biscuit":
        problems.append(f"dog={None if dog is None else dog.value}")
    if any(f.key == "fact:dog-name" and f.status == "active" and f.value == "Pixel" for f in disk.facts.values()):
        problems.append("pixel still active")
    pixel = [f for f in disk.history("fact:dog-name") if f.value == "Pixel"]
    if not pixel or pixel[0].status != "superseded":
        problems.append("pixel was not superseded")
    if health is None or health.value != "breast cancer":
        problems.append("diagnosis missing")
    if job is None or job.value != "Airbus":
        problems.append("job missing")
    if not preferences or "oat milk" not in preferences[0].value:
        problems.append("preference missing")
    if any(f.key in {"fact:that", "fact:it", "fact:this"} for f in disk.active()):
        problems.append("pronoun junk stored")
    block = build_packet(disk, "what do you remember about me?").text
    if re.search(
        r"importance|relevance|recency|salience|priority|\bhigh\b|\blow\b|"
        r"\[(?:high|low|normal|medium|priority)\]|\b0\.\d",
        block,
        re.I,
    ):
        problems.append("scores leaked into the prompt")
    if "breast cancer" not in block or "Biscuit" not in block or "Airbus" not in block:
        problems.append("profile recall missed a kept fact")
    health_thread = disk.threads.get("thread:health-mum")
    if health_thread is None or health_thread.status != "open":
        problems.append("diagnosis thread not open")
    rule = run_turn(
        disk,
        brain,
        "Do not restart the training gateway while pretraining is active.",
        conversation_id="selftest",
        source_agent="maxi-text",
    )
    if not rule.memory_ok or "about to act against" in rule.reply:
        problems.append("stating the rule was treated as breaking it")
    action = run_turn(
        disk,
        brain,
        "I'm about to restart the training gateway.",
        conversation_id="selftest",
        source_agent="maxi-text",
    )
    if "about to act against" not in action.reply:
        problems.append("constraint warning missing")
    hello = run_turn(
        disk,
        brain,
        "Hello",
        conversation_id="selftest",
        source_agent="maxi-text",
    )
    if "check in" not in hello.reply or "breast cancer" not in hello.reply:
        problems.append("open-thread check-in missing")
    if _french(home / "fr"):
        problems.append("french slots failed")
    if _rename_forms(home / "renames"):
        problems.append("rename forms failed")
    if problems:
        print("FAIL " + "; ".join(problems))
        return 1
    print("PASS")
    print(f"dog: {dog.value} (Pixel superseded)")
    print(f"health: {health.value}")
    print(f"job: {job.value}")
    print(f"preference: {preferences[0].value}")
    print("scores in prompt: no")
    print("french slots: ok")
    print("constraint warning: ok")
    print("open-thread check-in: ok")
    print("rename forms: ok")
    return 0


_RENAME_FORMS = (
    "Actually I renamed him Biscuit",
    "I renamed him to Biscuit.",
    "His name is Biscuit now.",
    "We call him Biscuit now.",
    "Je l'ai renommé Biscuit.",
    "Il s'appelle maintenant Biscuit.",
    "On l'appelle Biscuit.",
)


def _rename_forms(root: Path) -> bool:
    """Each phrasing must supersede Pixel. An episode is not a rename."""
    brain = ScriptedBrain()
    for index, line in enumerate(_RENAME_FORMS):
        home = root / str(index)
        store = MemoryStore(home)
        introduced = run_turn(
            store,
            brain,
            "My dog is called Pixel.",
            conversation_id="selftest-rename",
            source_agent="maxi-text",
        )
        if not introduced.memory_ok:
            return True
        renamed = run_turn(
            store,
            brain,
            line,
            conversation_id="selftest-rename",
            source_agent="maxi-text",
        )
        if renamed.receipt is None or renamed.receipt.stored < 1 or not renamed.memory_ok:
            return True
        if any(fact.type == "episode" for fact in renamed.receipt.facts):
            return True
        disk = MemoryStore(home)
        dog = disk.slot("fact:dog-name")
        if dog is None or dog.value != "Biscuit":
            return True
        if any(
            fact.key == "fact:dog-name" and fact.status == "active" and fact.value == "Pixel"
            for fact in disk.facts.values()
        ):
            return True
        asked = run_turn(disk, brain, "What's my dog's name?", conversation_id="selftest-rename")
        if asked.reply != "Your dog's name is Biscuit.":
            return True
    return False


def _french(home: Path) -> bool:
    """Same three facts in French, on a separate home so they do not
    supersede the English rows the checks above already read."""
    store = MemoryStore(home)
    brain = ScriptedBrain()
    lines = (
        "Mon chien s'appelle Pixel.",
        "Ma mère a été diagnostiquée d'un cancer du sein la semaine dernière, et j'ai peur.",
        "Je viens de décrocher un poste chez Airbus.",
        "Je l'ai renommé Biscuit.",
        "Souviens-toi de ça.",
        "Ne redémarre pas la passerelle d'entraînement pendant le pré-entraînement.",
        "Je vais redémarrer la passerelle.",
    )
    last = None
    for line in lines:
        last = run_turn(store, brain, line, conversation_id="selftest-fr", source_agent="maxi-text")
        if last is None or not last.memory_ok:
            return True
    disk = MemoryStore(home)
    dog = disk.slot("fact:dog-name")
    health = disk.slot("health:mum-diagnosis")
    job = disk.slot("job:employer")
    if dog is None or dog.value != "Biscuit":
        return True
    if health is None or "sein" not in health.value:
        return True
    if job is None or job.value != "Airbus":
        return True
    if any(fact.key in {"fact:ça", "fact:ca", "fact:cela"} for fact in disk.active()):
        return True
    if last is None or "aller contre" not in last.reply:
        return True
    name = run_turn(disk, brain, "Comment s'appelle mon chien ?", conversation_id="selftest-fr")
    if "Biscuit" not in name.reply:
        return True
    hi = run_turn(disk, brain, "Salut", conversation_id="selftest-fr")
    return "fil ouvert" not in hi.reply or "sein" not in hi.reply
