"""Deterministic replies for the offline profile and for tests.

This is the brain the verified profile uses. It only speaks facts that
are already in the store, so a passing test is not a model that happened
to remember.
"""

from __future__ import annotations

import re

from hyper_ambient_maxi.memory.store import Fact, Receipt


class ScriptedBrain:
    name = "scripted"

    def reply(
        self,
        user: str,
        context: str,
        receipt: Receipt,
        memories: list[Fact],
        notes: list[str] | None = None,
        warning: str | None = None,
        **_extra,
    ) -> str:
        del context  # the loop already decided which facts are visible
        notes = notes or []
        if warning and not _is_question(user):
            base = _acknowledge(user, receipt)
            return f"{base} {warning}".strip()
        if _is_question(user):
            answer = _answer(user, memories, notes)
            if warning:
                return f"{answer} {warning}"
            return answer
        return _acknowledge(user, receipt)


def _acknowledge(user: str, receipt: Receipt) -> str:
    french = _french(user)
    if receipt.stored and receipt.facts:
        summaries = "; ".join(fact.summary for fact in receipt.facts)
        if french and re.search(r"peur|inquiet|angoiss", user, re.I):
            lead = "Je suis désolé. Je vais m'en souvenir. "
        elif re.search(r"\bscared|afraid|worried|frighten", user, re.I):
            lead = "I'm sorry. I'll remember that. "
        elif french:
            lead = "Je vais m'en souvenir. "
        else:
            lead = "I'll remember that. "
        if receipt.corrections:
            tail = " Ça remplace ce que j'avais." if french else " That replaces what I had before."
            return f"{lead}{summaries}.{tail}"
        return f"{lead}{summaries}."
    if receipt.already:
        if french:
            return f"Je m'en souviens déjà. {receipt.already[0].summary}."
        return f"I already remember that. {receipt.already[0].summary}."
    return "Je suis là." if french else "I'm here."


def _is_question(user: str) -> bool:
    text = user.strip()
    if "?" in text:
        return True
    return bool(
        re.match(r"(?i)(?:what|where|who|when|why|how|do you|did i)\b", text)
    )


def _by_key(memories: list[Fact]) -> dict[str, Fact]:
    found: dict[str, Fact] = {}
    for fact in memories:
        found[fact.key] = fact
    return found


def _french(user: str) -> bool:
    return re.search(
        r"\b(je|j'ai|mon|ma|mes|tu|ton|ta|chez|chien|mère|mere|maman|préfère|prefere|souviens|rappelle|salut|bonjour|bonsoir|coucou|habite|allergique|appelle)\b|en fait",
        user,
        re.I,
    ) is not None


def _answer(user: str, memories: list[Fact], notes: list[str]) -> str:
    slots = _by_key(memories)
    french = _french(user)
    if re.search(r"\bdog\b|\bpuppy\b|\bpet\b|\bchien\b", user, re.I):
        if notes and re.search(r"used to|avant", user, re.I):
            if french:
                return f"Avant, ton chien s'appelait {notes[-1]}."
            return f"You used to call your dog {notes[-1]}."
        fact = slots.get("fact:dog-name")
        if fact is None:
            return "Je n'ai pas le nom de ton chien." if french else "I don't have your dog's name stored."
        return f"Ton chien s'appelle {fact.value}." if french else f"Your dog's name is {fact.value}."
    if re.search(r"\bmum\b|\bmom\b|\bmother\b|\bmère\b|\bmere\b|\bmaman\b", user, re.I):
        fact = slots.get("health:mum-diagnosis")
        if fact is None:
            episode = _episode_about(memories, ("mum", "mom", "mother", "mère", "maman", "diagnos"))
            if episode is not None:
                return episode.summary
            return "Je n'ai rien sur ta mère." if french else "I don't have a health note about your mum stored."
        if french:
            return f"Ta mère a été diagnostiquée d'un {fact.value}."
        return f"Your mum was diagnosed with {fact.value}."
    if re.search(r"\bwork\b|\bjob\b|\bemployer\b|\btravaille\b|\bposte\b|\bemploi\b", user, re.I):
        fact = slots.get("job:employer")
        if fact is None or fact.value.lower() == "none":
            episode = _episode_about(memories, ("job", "work", "hired", "poste", "travaille"))
            if episode is not None:
                return episode.summary
            return "Je n'ai pas de travail enregistré." if french else "I don't have a job stored."
        role = slots.get("job:role")
        if role is not None and not french:
            return f"You work at {fact.value}, as {role.value}."
        return f"Tu travailles chez {fact.value}." if french else f"You work at {fact.value}."
    if re.search(
        r"what do you remember|what do you know about me|que sais-tu|qu'est-ce que tu|tu te souviens",
        user,
        re.I,
    ):
        if not memories:
            return "Je n'ai rien enregistré." if french else "I don't have anything stored yet."
        lead = "Je me souviens : " if french else "I remember: "
        return lead + "; ".join(fact.summary for fact in memories) + "."
    person = _person_answer(user, slots, french)
    if person is not None:
        return person
    if re.search(r"\bmy name\b|je m'appelle|comment je m'appelle|who am i|qui suis-je", user, re.I):
        fact = slots.get("fact:user-name")
        if fact is None:
            return "Je n'ai pas ton nom." if french else "I don't have your name stored."
        return f"Tu t'appelles {fact.value}." if french else f"Your name is {fact.value}."
    if re.search(r"\ballergi", user, re.I):
        fact = slots.get("health:allergy")
        if fact is None:
            return "Je n'ai pas d'allergie enregistrée." if french else "I don't have an allergy stored."
        return f"Tu es allergique à {fact.value}." if french else f"You are allergic to {fact.value}."
    if re.search(r"\blive\b|\bhabite\b|\bcity\b|\bville\b", user, re.I):
        fact = slots.get("fact:home-city")
        if fact is None:
            return "Je ne sais pas où tu habites." if french else "I don't have a city stored."
        return f"Tu habites à {fact.value}." if french else f"You live in {fact.value}."
    if re.search(r"\bcodename\b", user, re.I):
        fact = slots.get("fact:test-codename")
        if fact is None:
            return "I don't have a codename stored."
        return f"Your test codename is {fact.value}."
    if memories:
        return memories[0].summary + "."
    return "I don't have that stored."


def _episode_about(memories: list[Fact], needles: tuple[str, ...]) -> Fact | None:
    for fact in memories:
        if fact.type != "episode":
            continue
        haystack = fact.summary.lower()
        if any(needle in haystack for needle in needles):
            return fact
    return None


_PERSON_ASK = re.compile(
    r"(?i)\b(sister|sœur|soeur|brother|frère|frere|mum|mom|mother|mère|mere|maman|"
    r"dad|father|père|pere|papa|wife|husband|femme|mari|son|fils|daughter|fille|"
    r"friend|ami|amie|aunt|tante|uncle|oncle|cousin|cousine)\b"
)
_PERSON_CANON = {
    "sister": "sister",
    "sœur": "sister",
    "soeur": "sister",
    "brother": "brother",
    "frère": "brother",
    "frere": "brother",
    "mum": "mum",
    "mom": "mum",
    "mother": "mum",
    "mère": "mum",
    "mere": "mum",
    "maman": "mum",
    "dad": "dad",
    "father": "dad",
    "père": "dad",
    "pere": "dad",
    "papa": "dad",
    "wife": "wife",
    "femme": "wife",
    "husband": "husband",
    "mari": "husband",
    "son": "son",
    "fils": "son",
    "daughter": "daughter",
    "fille": "daughter",
    "friend": "friend",
    "ami": "friend",
    "amie": "friend",
    "aunt": "aunt",
    "tante": "aunt",
    "uncle": "uncle",
    "oncle": "uncle",
    "cousin": "cousin",
    "cousine": "cousin",
}


def _person_answer(user: str, slots: dict[str, Fact], french: bool) -> str | None:
    match = _PERSON_ASK.search(user)
    if match is None:
        return None
    relation = _PERSON_CANON.get(match.group(1).lower())
    if not relation:
        return None
    fact = slots.get(f"person:{relation}")
    if fact is None:
        return None
    if french:
        label = {
            "sister": "Ta sœur",
            "brother": "Ton frère",
            "mum": "Ta mère",
            "dad": "Ton père",
            "wife": "Ta femme",
            "husband": "Ton mari",
            "son": "Ton fils",
            "daughter": "Ta fille",
            "friend": "Ton ami",
            "aunt": "Ta tante",
            "uncle": "Ton oncle",
            "cousin": "Ton/ta cousin(e)",
        }.get(relation, relation)
        return f"{label} s'appelle {fact.value}."
    return f"Your {relation} is called {fact.value}."
