"""Turn a user sentence into structured facts, without calling a model.

The 2026-11-05 misses were not disk bugs. The diagnosis was never sent
to the store, the job update was sent and came back empty, and the
rename never replaced the old slot. Extraction therefore lives in the
harness, runs on the user text itself, and treats a durable cue with
no structured parse as a verbatim episode so the words are still stored.

English and French share the same slots: ``ma mère`` and ``my mum``
both write ``health:mum-diagnosis``, and a rename in either language
supersedes ``fact:dog-name``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from hashlib import sha256

from hyper_ambient_maxi.memory.store import ProposedFact, ProposedThread

_NAME_STOP = {
    "now",
    "actually",
    "please",
    "today",
    "again",
    "the",
    "a",
    "an",
    "my",
    "me",
    "him",
    "her",
    "it",
    "that",
    "this",
    "just",
}

_PERSON = {
    "mom": "mum",
    "mother": "mum",
    "father": "dad",
    "mère": "mum",
    "mere": "mum",
    "maman": "mum",
    "père": "dad",
    "pere": "dad",
    "papa": "dad",
}

# Cues that mean "this sentence should leave a trace". A question does
# not, even if it mentions a dog or a job.
_CUES: tuple[tuple[str, str], ...] = (
    ("diagnosis", r"\bdiagnos"),
    ("job", r"\b(?:job|hired|employment)\b|\bwork at\b"),
    ("rename", r"\brenamed\b|\bname is now\b"),
    ("preference", r"\bi prefer\b"),
    ("remember", r"\bremember that\b"),
    ("named", r"\b(?:called|named)\b"),
    ("codename", r"\bcodename is\b"),
    ("diagnosis_fr", r"diagnostiqu"),
    ("job_fr", r"\b(?:poste|emploi|travaille|embauch)"),
    ("rename_fr", r"renomm|rebaptis"),
    ("preference_fr", r"je préfère|je prefere"),
    ("remember_fr", r"souviens-toi que|rappelle-toi que|n'oublie pas que"),
    ("name", r"\bmy name is\b|je m'appelle"),
    ("allergy", r"\ballergi"),
    ("home", r"\bi live in\b|j'habite"),
    ("appointment", r"\bappointment\b|rendez-vous"),
    ("named_fr", r"s'appelle"),
    ("constraint", r"^(?:please\s+)?(?:do not|don't|never|ne |n'|il ne faut pas)\b"),
)

_DOG = re.compile(
    r"\b(?:my dog(?:'s name)? is (?:called |named )?"
    r"|i have a dog named )"
    r"([A-Za-z][A-Za-z'-]{1,40})\b",
    re.I,
)
_RENAME = re.compile(
    r"\b(?:i renamed (?:him|her|my dog|the dog)(?:\s+to)?"
    r"|now called"
    r"|we call (?:him|her)"
    r"|(?:his|her) name is now"
    r"|(?:his|her) name is)"
    r"\s+([A-Za-z][A-Za-z'-]{1,40})\b",
    re.I,
)
_DIAG = re.compile(
    r"\bmy (mum|mom|mother|dad|father|sister|brother|partner|wife|husband)"
    r" was diagnosed with ([a-z0-9][a-z0-9 \-]{1,80}?)"
    r"(?=\s+last\b|\s*,|\s+and\b|\.|$)",
    re.I,
)
_JOB = re.compile(
    r"\b(?:i\s+)?(?:just\s+)?(?:got|started|have|accepted|landed)\s+"
    r"(?:a\s+|an\s+)?(?:new\s+)?(?:job|role|position)\s+at\s+"
    r"([A-Za-z][\w&'-]*)",
    re.I,
)
_HIRED = re.compile(r"\b([A-Za-z][\w&'-]*)\s+hired\s+me\b", re.I)
_WORK = re.compile(r"\bi\s+(?:now\s+)?work\s+at\s+([A-Za-z][\w&'-]*)", re.I)
_LEFT = re.compile(r"\bi\s+left\s+([A-Za-z][\w&'-]*)", re.I)
_ROLE = re.compile(
    r"\bas\s+an?\s+([A-Za-z][A-Za-z ]{2,40}?)(?=\s*(?:\.|,|$))",
    re.I,
)
_PREF = re.compile(r"\bi prefer\s+([^.?\n]+)", re.I)
_CODE = re.compile(
    r"\bmy (?:test )?codename is\s+([A-Za-z][A-Za-z0-9'-]{1,40})\b",
    re.I,
)
_REMEMBER = re.compile(r"\bremember that\s+(.+)$", re.I)
_QUESTION = re.compile(
    r"^(?:what|where|who|when|why|how|do you|did i|can you|could you|"
    r"comment|où|ou|pourquoi|est-ce|qu'est-ce|que sais)\b",
    re.I,
)
_CLOSE = re.compile(
    r"(?i)\b(?:done with|finished with|j'ai terminé(?:\s+avec)?|c'est réglé pour|c'est regle pour)\s+(.+)$"
)
_PROHIBITION = re.compile(
    r"(?i)^(?:please\s+)?(?:do not|don't|never|ne |n'|il ne faut pas)\b"
)


@dataclass
class ExtractResult:
    facts: list[ProposedFact] = field(default_factory=list)
    cues: list[str] = field(default_factory=list)
    durable_unparsed: bool = False
    rejected_pronoun: bool = False
    threads: list[ProposedThread] = field(default_factory=list)
    closes: list[str] = field(default_factory=list)


_CORRECTION_LEAD = re.compile(
    r"(?i)^(?:non(?:\s*,)?\s+)?en fait(?:\s*,)?\s+(.+)$"
)
_BLOCK_EPISODE = re.compile(
    r"(?i)(?:ne s'appelle pas|n'ai pas de rendez-vous|pas de rendez-vous|"
    r"n'aime pas|ne travaille (?:pas|plus))"
)


def extract(text: str) -> ExtractResult:
    raw = " ".join((text or "").split())
    if not raw:
        return ExtractResult()
    lead = _CORRECTION_LEAD.match(raw)
    if lead:
        raw = lead.group(1).strip()
    cues = _cues(raw)
    if _is_pronoun_remember(raw):
        return ExtractResult(cues=cues, rejected_pronoun=True)

    facts: list[ProposedFact] = []
    dog = _dog_fact(raw)
    if dog is not None:
        facts.append(dog)
    renamed = _rename_fact(raw)
    if renamed is not None:
        facts = [f for f in facts if f.key != "fact:dog-name"]
        facts.append(renamed)
    facts.extend(_health_facts(raw))
    facts.extend(_job_facts(raw))
    pref = _preference_fact(raw)
    if pref is not None:
        facts.append(pref)
    code = _codename_fact(raw)
    if code is not None:
        facts.append(code)
    if not facts:
        note = _remember_fact(raw)
        if note is not None:
            facts.append(note)
    for fact in _french_facts(raw):
        facts = [old for old in facts if old.key != fact.key]
        facts.append(fact)
    for fact in _life_facts(raw):
        facts = [old for old in facts if old.key != fact.key]
        facts.append(fact)
    if not any(fact.type == "constraint" for fact in facts):
        rule = _constraint_fact(raw)
        if rule is not None:
            facts.append(rule)

    question = _is_question(raw)
    unparsed = False
    if (
        cues
        and not facts
        and not question
        and not _PROHIBITION.match(raw)
        and _CLOSE.search(raw) is None
        and _BLOCK_EPISODE.search(raw) is None
    ):
        facts.append(_episode(raw))
        unparsed = True
    return _finish(raw, facts, cues, unparsed, False)


def _cues(text: str) -> list[str]:
    return [name for name, pat in _CUES if re.search(pat, text, re.I)]


def _is_question(text: str) -> bool:
    if "?" in text:
        return True
    return _QUESTION.match(text) is not None


def _is_pronoun_remember(text: str) -> bool:
    trimmed = text.strip().rstrip(".! ")
    return (
        re.fullmatch(
            r"(?i)(?:please\s+)?remember(?:\s+(?:that|it|this))?",
            trimmed,
        )
        is not None
        or re.fullmatch(
            r"(?i)(?:s'il te plaît\s+|s'il te plait\s+)?"
            r"(?:souviens-toi|rappelle-toi)(?:\s+de\s+(?:ça|ca|cela))?",
            trimmed,
        )
        is not None
    )


def _finish(raw: str, facts: list[ProposedFact], cues: list[str], unparsed: bool, rejected: bool) -> ExtractResult:
    close = _CLOSE.search(raw)
    closes = [close.group(1).strip(" .")] if close else []
    return ExtractResult(
        facts=facts,
        cues=cues,
        durable_unparsed=unparsed,
        rejected_pronoun=rejected,
        threads=_threads_for(facts),
        closes=closes,
    )


def _name(token: str) -> str:
    token = token.strip(" .,!?:;\"'")
    if not token or token.lower() in _NAME_STOP:
        return ""
    if token.islower():
        return token[0].upper() + token[1:]
    return token


def _phrase(token: str) -> str:
    return " ".join(token.strip(" .,!?:;\"'").split())


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "note"


def _dog_fact(text: str) -> ProposedFact | None:
    match = _DOG.search(text)
    if match is None:
        return None
    name = _name(match.group(1))
    if not name:
        return None
    return ProposedFact(
        key="fact:dog-name",
        type="personal_fact",
        summary=f"User's dog is called {name}",
        value=name,
        evidence=text,
        importance=0.8,
    )


def _rename_fact(text: str) -> ProposedFact | None:
    match = _RENAME.search(text)
    if match is None:
        return None
    name = _name(match.group(1))
    if not name:
        return None
    return ProposedFact(
        key="fact:dog-name",
        type="personal_fact",
        summary=f"User's dog is called {name}",
        value=name,
        evidence=text,
        importance=0.8,
    )


def _health_facts(text: str) -> list[ProposedFact]:
    match = _DIAG.search(text)
    if match is None:
        return []
    person = match.group(1).lower()
    person = _PERSON.get(person, person)
    condition = _phrase(match.group(2)).lower()
    if not condition:
        return []
    return [
        ProposedFact(
            key=f"health:{person}-diagnosis",
            type="health",
            summary=f"User's {person} was diagnosed with {condition}",
            value=condition,
            evidence=text,
            importance=0.95,
        )
    ]


def _job_facts(text: str) -> list[ProposedFact]:
    out: list[ProposedFact] = []
    employer = ""
    for pattern in (_WORK, _JOB, _HIRED):
        match = pattern.search(text)
        if match is None:
            continue
        employer = _name(match.group(1))
        if employer:
            break
    if employer:
        out.append(
            ProposedFact(
                key="job:employer",
                type="job",
                summary=f"User works at {employer}",
                value=employer,
                evidence=text,
                importance=0.9,
            )
        )
    elif (left := _LEFT.search(text)) is not None:
        old = _name(left.group(1))
        if old:
            out.append(
                ProposedFact(
                    key="job:employer",
                    type="job",
                    summary=f"User no longer works at {old}",
                    value="none",
                    evidence=text,
                    importance=0.9,
                )
            )
    role_match = _ROLE.search(text)
    if role_match is not None:
        role = _phrase(role_match.group(1)).lower()
        if role and role not in _NAME_STOP:
            out.append(
                ProposedFact(
                    key="job:role",
                    type="job",
                    summary=f"User's role is {role}",
                    value=role,
                    evidence=text,
                    importance=0.75,
                )
            )
    return out


def _preference_fact(text: str) -> ProposedFact | None:
    match = _PREF.search(text)
    if match is None:
        return None
    value = _phrase(match.group(1))
    if not value or value.lower() in _NAME_STOP:
        return None
    return ProposedFact(
        key=f"preference:{_slug(value)}",
        type="preference",
        summary=f"User prefers {value}",
        value=value,
        evidence=text,
        importance=0.6,
    )


def _codename_fact(text: str) -> ProposedFact | None:
    match = _CODE.search(text)
    if match is None:
        return None
    name = _name(match.group(1))
    if not name:
        return None
    return ProposedFact(
        key="fact:test-codename",
        type="personal_fact",
        summary=f"User's test codename is {name}",
        value=name,
        evidence=text,
        importance=0.85,
    )


def _remember_fact(text: str) -> ProposedFact | None:
    match = _REMEMBER.search(text)
    if match is None:
        return None
    clause = _phrase(match.group(1)).rstrip(".")
    if not clause or clause.lower() in _NAME_STOP | {"that", "it", "this"}:
        return None
    return ProposedFact(
        key=f"fact:note-{_slug(clause)[:40]}",
        type="personal_fact",
        summary=f"Remembered: {clause}",
        value=clause,
        evidence=text,
        importance=0.7,
    )


def _episode(text: str) -> ProposedFact:
    digest = sha256(_norm_episode(text).encode()).hexdigest()[:12]
    return ProposedFact(
        key=f"episode:{digest}",
        type="episode",
        summary=text.strip(),
        value=text.strip(),
        evidence=text,
        importance=0.7,
    )


def _norm_episode(text: str) -> str:
    return " ".join(text.lower().split())


_DOG_FR = re.compile(
    r"\b(?:mon chien s'appelle|mon chien est (?:appelé|appele|nommé|nomme)|"
    r"j'ai un chien (?:qui s'appelle|nommé|appelé))"
    r"\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'-]{1,40})\b",
    re.I,
)
_RENAME_FR = re.compile(
    r"\b(?:je l'ai renommé(?:e)?(?: en)?|je l'ai rebaptisé(?:e)?(?: en)?|"
    r"son nom est maintenant|il s'appelle maintenant|elle s'appelle maintenant|"
    r"on l['’]appelle maintenant|"
    r"il s'appelle(?!\s+maintenant)|elle s'appelle(?!\s+maintenant)|"
    r"on l['’]appelle(?!\s+maintenant))"
    r"\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'-]{1,40})\b",
    re.I,
)
_DIAG_FR = re.compile(
    r"\bma (mère|mere|maman) a été diagnostiquée "
    r"(?:d'un |d'une |de |avec |d')?"
    r"([a-zàâäéèêëïîôùûüç][a-zàâäéèêëïîôùûüç \-]{1,80}?)"
    r"(?=\s+la semaine|\s*,|\s+et\b|\.|$)",
    re.I,
)
_JOB_FR = re.compile(
    r"\bje (?:viens de )?(?:décrocher|decrocher|avoir|accepter) "
    r"(?:un |une )?(?:poste|travail|emploi|job) (?:chez|à|a) "
    r"([A-Za-zÀ-ÿ][\w&'-]*)",
    re.I,
)
_WORK_FR = re.compile(
    r"\bje travaille (?:maintenant |désormais |desormais )?chez "
    r"([A-Za-zÀ-ÿ][\w&'-]*)",
    re.I,
)
_LEFT_FR = re.compile(r"\bj'ai quitté ([A-Za-zÀ-ÿ][\w&'-]*)", re.I)
_STOP_WORK_FR = re.compile(
    r"\bje ne travaille (?:plus|pas) chez ([A-Za-zÀ-ÿ][\w&'-]*)",
    re.I,
)
_DISLIKE_FR = re.compile(r"\bje n'aime pas\s+([^.?\n]+)", re.I)
_APPT_NEG = re.compile(r"(?i)\b(?:n'ai pas de|pas de)\s+rendez-vous\b")
_PREF_FR = re.compile(r"\bje pr[ée]f[èe]re\s+([^.?\n]+)", re.I)
_CODE_FR = re.compile(
    r"\bmon nom de code est\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'-]{1,40})\b",
    re.I,
)
_REMEMBER_FR = re.compile(
    r"\b(?:souviens-toi|rappelle-toi|n'oublie pas) que\s+(.+)$",
    re.I,
)
_NAME_EN = re.compile(r"\bmy name is\s+([A-Za-zÀ-ÿ][\w'-]{1,40})\b", re.I)
_NAME_FR = re.compile(r"\bje m'appelle\s+([A-Za-zÀ-ÿ][\w'-]{1,40})\b", re.I)
_ALLERGY_EN = re.compile(r"\bi(?:'m| am) allergic to\s+([^.?\n]+)", re.I)
_ALLERGY_FR = re.compile(
    r"\bje suis allergique (?:à la|a la|à l'|a l'|aux|au|à|a)\s+([^.?\n]+)",
    re.I,
)
_HOME_EN = re.compile(
    r"\bi live in\s+([A-Za-zÀ-ÿ][\w'-]+(?:\s+[A-Za-zÀ-ÿ][\w'-]+)?)",
    re.I,
)
_HOME_FR = re.compile(
    r"\bj'habite (?:à|a|au|en)\s+([A-Za-zÀ-ÿ][\w'-]+(?:\s+[A-Za-zÀ-ÿ][\w'-]+)?)",
    re.I,
)
_APPT_EN = re.compile(r"\bi have (?:an|a) appointment\b(.*)$", re.I)
_APPT_FR = re.compile(r"\bj'ai (?:un |une )?rendez-vous\b(.*)$", re.I)


def _french_facts(text: str) -> list[ProposedFact]:
    found: list[ProposedFact] = []
    dog = _DOG_FR.search(text)
    if dog is not None:
        name = _name(dog.group(1))
        if name:
            found.append(_named("fact:dog-name", name, f"User's dog is called {name}", text, 0.8))
    renamed = _RENAME_FR.search(text)
    if renamed is not None:
        name = _name(renamed.group(1))
        if name:
            found = [fact for fact in found if fact.key != "fact:dog-name"]
            found.append(_named("fact:dog-name", name, f"User's dog is called {name}", text, 0.8))
    diagnosis = _DIAG_FR.search(text)
    if diagnosis is not None:
        person = _PERSON.get(diagnosis.group(1).lower(), diagnosis.group(1).lower())
        condition = _phrase(diagnosis.group(2)).lower()
        if condition:
            found.append(
                ProposedFact(
                    key=f"health:{person}-diagnosis",
                    type="health",
                    summary=f"User's {person} was diagnosed with {condition}",
                    value=condition,
                    evidence=text,
                    importance=0.95,
                )
            )
    found.extend(_french_jobs(text))
    dislike = _DISLIKE_FR.search(text)
    if dislike is not None:
        value = _phrase(dislike.group(1))
        if value and value.lower() not in _NAME_STOP | {"ça", "ca", "cela"}:
            found.append(
                ProposedFact(
                    key=f"preference:avoid-{_slug(value)}",
                    type="preference",
                    summary=f"User does not want {value}",
                    value=value,
                    evidence=text,
                    importance=0.6,
                )
            )
    preference = _PREF_FR.search(text)
    if preference is not None:
        value = _phrase(preference.group(1))
        if value:
            found.append(
                ProposedFact(
                    key=f"preference:{_slug(value)}",
                    type="preference",
                    summary=f"User prefers {value}",
                    value=value,
                    evidence=text,
                    importance=0.6,
                )
            )
    code = _CODE_FR.search(text)
    if code is not None:
        name = _name(code.group(1))
        if name:
            found.append(
                _named(
                    "fact:test-codename",
                    name,
                    f"User's test codename is {name}",
                    text,
                    0.85,
                )
            )
    if not found:
        remembered = _REMEMBER_FR.search(text)
        if remembered is not None:
            clause = _phrase(remembered.group(1)).rstrip(".")
            if clause and clause.lower() not in _NAME_STOP | {"ça", "ca", "cela", "that"}:
                found.append(
                    ProposedFact(
                        key=f"fact:note-{_slug(clause)[:40]}",
                        type="personal_fact",
                        summary=f"Remembered: {clause}",
                        value=clause,
                        evidence=text,
                        importance=0.7,
                    )
                )
    return found


def _french_jobs(text: str) -> list[ProposedFact]:
    stopped = _STOP_WORK_FR.search(text)
    if stopped is not None:
        old = _name(stopped.group(1))
        if old:
            return [
                ProposedFact(
                    key="job:employer",
                    type="job",
                    summary=f"User no longer works at {old}",
                    value="none",
                    evidence=text,
                    importance=0.9,
                )
            ]
    employer = ""
    for pattern in (_WORK_FR, _JOB_FR):
        match = pattern.search(text)
        if match is None:
            continue
        employer = _name(match.group(1))
        if employer:
            break
    if employer:
        return [
            ProposedFact(
                key="job:employer",
                type="job",
                summary=f"User works at {employer}",
                value=employer,
                evidence=text,
                importance=0.9,
            )
        ]
    left = _LEFT_FR.search(text)
    if left is None:
        return []
    old = _name(left.group(1))
    if not old:
        return []
    return [
        ProposedFact(
            key="job:employer",
            type="job",
            summary=f"User no longer works at {old}",
            value="none",
            evidence=text,
            importance=0.9,
        )
    ]


def _life_facts(text: str) -> list[ProposedFact]:
    found: list[ProposedFact] = []
    name = _NAME_FR.search(text) or _NAME_EN.search(text)
    if name is not None:
        value = _name(name.group(1))
        if value:
            found.append(
                _named("fact:user-name", value, f"User's name is {value}", text, 0.85)
            )
    allergy = _ALLERGY_FR.search(text) or _ALLERGY_EN.search(text)
    if allergy is not None:
        value = _phrase(allergy.group(1))
        if value:
            found.append(
                ProposedFact(
                    key="health:allergy",
                    type="health",
                    summary=f"User is allergic to {value}",
                    value=value,
                    evidence=text,
                    importance=0.9,
                )
            )
    home = _HOME_FR.search(text) or _HOME_EN.search(text)
    if home is not None:
        value = _phrase(home.group(1))
        if value:
            found.append(
                _named("fact:home-city", value, f"User lives in {value}", text, 0.7)
            )
    appointment = None if _APPT_NEG.search(text) else (_APPT_FR.search(text) or _APPT_EN.search(text))
    if appointment is not None:
        from hyper_ambient_maxi.memory.dates import resolve_fr_when

        detail = _phrase(appointment.group(1)).lstrip(" :-")
        summary = text.strip().rstrip(".")
        when = resolve_fr_when(text)
        value = detail or summary
        if when is not None:
            value = f"{value} [{when.isoformat()}]".strip()
            summary = f"{summary} ({when.isoformat()})"
        found.append(
            ProposedFact(
                key="fact:appointment",
                type="appointment",
                summary=summary,
                value=value,
                evidence=text,
                importance=0.75,
            )
        )
    return found


def _constraint_fact(text: str) -> ProposedFact | None:
    if _PROHIBITION.match(text.strip()) is None:
        return None
    summary = text.strip().rstrip(".")
    if len(summary) < 8:
        return None
    return ProposedFact(
        key=f"constraint:{_slug(summary)[:48]}",
        type="constraint",
        summary=summary,
        value=summary,
        evidence=text,
        importance=0.95,
        tags=["constraint"],
    )


def _named(key: str, value: str, summary: str, text: str, importance: float) -> ProposedFact:
    kind = "personal_fact"
    return ProposedFact(
        key=key,
        type=kind,
        summary=summary,
        value=value,
        evidence=text,
        importance=importance,
    )


def _threads_for(facts: list[ProposedFact]) -> list[ProposedThread]:
    found: dict[str, ProposedThread] = {}
    for fact in facts:
        thread: ProposedThread | None = None
        if fact.key.startswith("health:") and fact.key.endswith("-diagnosis"):
            person = fact.key[len("health:") : -len("-diagnosis")]
            thread = ProposedThread(
                id=f"thread:health-{person}",
                title=f"{person} diagnosis",
                summary=fact.summary,
                priority="high",
            )
        elif fact.key == "fact:appointment":
            thread = ProposedThread(
                id="thread:appointment",
                title="Appointment",
                summary=fact.summary,
                priority="high",
            )
        elif fact.key == "job:employer":
            thread = ProposedThread(
                id="thread:job",
                title="Work",
                summary=fact.summary,
                priority="normal",
            )
        elif fact.type == "constraint":
            thread = ProposedThread(
                id=f"thread:{fact.key}",
                title="Constraint",
                summary=fact.summary,
                priority="high",
            )
        if thread is not None:
            found[thread.id] = thread
    return list(found.values())
