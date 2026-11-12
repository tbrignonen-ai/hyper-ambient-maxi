"""One chat client for a local llama-server and any OpenAI-compatible URL.

Hyper Ambient uses the same split: ``llama-server`` speaks
``/v1/chat/completions``, so the local GGUF and a remote model are a
config change. This file is a new client. hyper-ambient's
``src/brain/openai_compat.py`` was not copied; that repository has no
license file.

A reasoning model on llama-server (LFM3-1.7B with the default template)
fills ``reasoning_content`` and leaves ``content`` empty, or wraps the
reply in ``<think>``. This client never speaks ``reasoning_content``.
An empty ``content`` is retried once with a larger token budget and
thinking disabled. Voice and llama-local profiles send that disable on
the first request (``think = false``), so the retry is the rare path.
The API key, if any, is an Authorization header and
is never written into the body or the memory journal.

``HAM_BRAIN_BASE_URL`` overrides the profile URL. ``--brain-url`` does
the same for one process.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from hyper_ambient_maxi.memory.store import Fact, Receipt

_STOPS = [
    "\nUser:",
    "\nuser:",
    "\nHuman:",
    "\nAssistant:",
    "<|user|>",
    "<|im_start|>",
]
_THINK = re.compile(r"(?is)<think>.*?</think>")
_THINK_TAG = re.compile(r"(?i)</?think>")
_LEADING_ROLE = re.compile(r"(?i)^\s*(?:assistant|user|human)\s*:\s*")
_CONTINUATION = re.compile(
    r"(?i)\n\s*(?:user|human|assistant)\s*:|<\|\s*(?:user|im_start)"
)
# A second answer glued on with no space: "involves.That’s".
# The letter before the punctuation keeps "U.S.Army" and "3.14" intact.
_GLUED = re.compile(r"[a-zà-ÿ][.!?](?=[A-ZÀ-Ý])")
_ABBREV = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "st",
    "vs",
    "etc",
    "prof",
    "jr",
    "sr",
}
_ABBREV_TAIL = re.compile(r"(?i)(?:^|[^A-Za-z])(?:e\.g|i\.e|u\.s|u\.k)$")
# A later sentence that is a fresh greeting or a stock congratulations,
# not the first sentence of the reply.
_RESTART = re.compile(
    r"(?ix)^['«\s]*"
    r"(?:hello|hi|hey|hi\ there|good\ morning|good\ afternoon|good\ evening"
    r"|salut|bonjour|bonsoir|coucou"
    r"|that['’]s\ (?:awesome|great|wonderful|amazing|fantastic)"
    r"|congrats|congratulations|félicitations|felicitations|bravo"
    r"|(?:awesome|great|wonderful|amazing)\ news"
    r"|c['’]est\ super"
    r"|well\ done|nice\ one)\b"
)

_ADDRESS = re.compile(
    r"(?i)^(hi|hello|hey|hi there|salut|bonjour|bonsoir|coucou)"
    r"(?:\s*,)?\s+maxi\b([,.!])?"
)


def system_prompt(user_name: str = "") -> str:
    """Maxi is the companion. The name slot is the user, when one is stored."""
    name = " ".join((user_name or "").split())
    if name:
        who = f"The user is {name}. You are not the user. Never call the user Maxi.\n"
    else:
        who = (
            "The user's name is not stored yet. "
            "Do not invent a name for the user, and do not call them Thomas "
            "unless that name is stored. Never call the user Maxi.\n"
        )
    return (
        "You are Maxi, a local voice companion. "
        + who
        + (
            "Speak in one to three short sentences. Plain words. No lists. No markdown. No emoji.\n"
            "Do not repeat the user's sentence as your reply. "
            "After a new fact is saved, confirm the stored value in your own words, "
            "naming that value. Prefer Okay or I'll remember that, not Got it.\n"
            "Speak about the user's life in the second person: say your dog, not my dog.\n"
            "If the user is only greeting you, greet them warmly and do not mention memory.\n"
            "Use only the memory block for facts about the user's life. "
            "If a fact is not in that block, say you do not have it stored. "
            "If the block or an earlier-values note has the answer, do not say it is missing.\n"
            "The harness saves memories itself. Say that something was saved "
            "only when the harness note says so.\n"
            "Never mention importance, relevance, or recency. There are no scores.\n"
            "Pets, family, and jobs belong to the user, not to you."
        )
    )


class OpenAICompatBrain:
    name = "openai-compat"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "local",
        api_key: str | None = None,
        timeout_s: float = 120.0,
        max_tokens: int = 512,
        think: bool = False,
        max_sentences: int = 3,
    ):
        self.base_url = resolve_base_url(base_url)
        self.model = model
        self.api_key = api_key if api_key is not None else os.environ.get(
            "HAM_BRAIN_API_KEY", ""
        )
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.think = think
        self.max_sentences = max_sentences

    def health(self) -> dict:
        try:
            payload = self._get("/models")
        except Exception as exc:  # network, HTTP, JSON
            return {"ok": False, "detail": str(exc)}
        return {"ok": True, "detail": "models", "raw_keys": list(payload)[:4]}

    def reply(
        self,
        user: str,
        context: str,
        receipt: Receipt,
        memories: list[Fact],
        notes: list[str] | None = None,
        warning: str | None = None,
        **extra,
    ) -> str:
        spoken_user = user
        if warning:
            user = user + "\n\nWorth raising, say this plainly: " + warning
        called = str(extra.get("user_name") or "").strip() or _user_name(memories, receipt)
        system = system_prompt(called)
        if context:
            system += "\n\n" + context
        if notes:
            earlier = ", ".join(str(note).strip() for note in notes if str(note).strip())
            if earlier:
                system += (
                    "\n\nEarlier values of a slot, use these if the user asks "
                    "what something used to be: " + earlier
                )
        if receipt.facts:
            saved = "; ".join(fact.summary for fact in receipt.facts)
            system += "\n\nThe harness already saved: " + saved
        elif receipt.stored == 0:
            system += "\n\nThe harness did not save a new fact on this turn."
        payload = self._post(
            "/chat/completions",
            self._body(system, user, self.max_tokens, disable_thinking=not self.think),
        )
        text = speakable_text(payload, max_sentences=self.max_sentences)
        if text:
            return polish_reply(
                text, spoken_user, context=context, notes=notes, receipt=receipt, memories=memories
            )
        # Empty content is not a license to read reasoning_content. One
        # retry spends the budget on the reply and asks the template to
        # stop thinking. A second empty content falls through to the
        # scripted brain instead of voicing the chain of thought.
        retry_tokens = min(4096, max(self.max_tokens * 4, 1024))
        try:
            payload = self._post(
                "/chat/completions",
                self._body(system, user, retry_tokens, disable_thinking=True),
            )
        except Exception as exc:
            raise RuntimeError("model returned no speakable content") from exc
        text = speakable_text(payload, max_sentences=self.max_sentences)
        if not text:
            raise RuntimeError("model returned no speakable content")
        return polish_reply(
            text, spoken_user, context=context, notes=notes, receipt=receipt, memories=memories
        )

    def _body(self, system: str, user: str, max_tokens: int, *, disable_thinking: bool) -> dict:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.4,
            "stop": list(_STOPS),
        }
        if disable_thinking:
            # Cover the llama.cpp / Qwen / LFM template flags we have seen.
            # LFM2.5-8B still spent hidden reasoning tokens with only
            # enable_thinking=False, so the request also sends think=false
            # and a zero reasoning budget when the server honors them.
            body["chat_template_kwargs"] = {
                "enable_thinking": False,
                "thinking": False,
            }
            body["reasoning_effort"] = "none"
            body["think"] = False
            body["reasoning_budget"] = 0
        return body

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get(self, path: str) -> dict:
        request = urllib.request.Request(
            self.base_url + path,
            headers=self._headers(),
            method="GET",
        )
        return self._read(request)

    def _post(self, path: str, body: dict) -> dict:
        if "api_key" in body or "authorization" in body:
            raise RuntimeError("refusing to put a credential in the request body")
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=self._headers(),
            method="POST",
        )
        return self._read(request)

    def _read(self, request: urllib.request.Request) -> dict:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def resolve_base_url(profile_url: str) -> str:
    """CLI and ``HAM_BRAIN_BASE_URL`` win over the profile URL."""
    override = os.environ.get("HAM_BRAIN_BASE_URL", "").strip()
    chosen = override or (profile_url or "http://127.0.0.1:8080/v1")
    return chosen.rstrip("/")


def speakable_text(payload: dict, max_sentences: int = 3) -> str:
    """The words to say. ``reasoning_content`` is ignored on purpose."""
    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ""
    raw = message.get("content") if isinstance(message, dict) else ""
    if isinstance(raw, list):
        raw = " ".join(
            str(part.get("text") or "")
            for part in raw
            if isinstance(part, dict)
        )
    if not isinstance(raw, str):
        raw = ""
    return clean_reply(raw, max_sentences=max_sentences)


def clean_reply(text: str, max_sentences: int = 3) -> str:
    """Drop think tags, a hallucinated next turn, and a second glued answer.

    Voice replies stay within ``max_sentences`` (default 3). A second
    answer stuck to the punctuation, as in ``involves.That’s``, is cut.
    So is a later greeting or a stock congratulations. The first
    sentence is kept even when it is itself a greeting.
    """
    cleaned = _THINK.sub("", text or "")
    cleaned = _THINK_TAG.sub("", cleaned)
    cleaned = _strip_markup(cleaned)
    cleaned = _LEADING_ROLE.sub("", cleaned).strip()
    cleaned = _drop_user_maxi(cleaned)
    match = _CONTINUATION.search(cleaned)
    if match is not None:
        cleaned = cleaned[: match.start()]
    cleaned = _drop_second_answer(cleaned)
    if max_sentences > 0:
        cleaned = " ".join(_split_sentences(cleaned)[:max_sentences])
    return " ".join(cleaned.split())


_REASONING_LEAK = re.compile(
    r"(?i)("
    r"\bthe user\b|"
    r"memory block|"
    r"harness already saved|"
    r"in memory from|"
    r"fact:[\w-]+|"
    r"i (?:just )?need to|"
    r"i(?:'ll| will) say|"
    r"say something like|"
    r"something like:|"
    r"in my own words|"
    r"without repeating|"
    r"without mentioning|"
    r"already stored|"
    r"existing memory|"
    r"they're essentially|"
    r"this matches what"
    r")"
)
_GOT_IT = re.compile(r"(?i)^got it\b(?:[,.!?]\s*|\s+)")
_GENERIC_HEDGE = re.compile(
    r"(?i)^(?:"
    r"i (?:do not|don't) have that stored"
    r"|i (?:do not|don't) have that in (?:my )?memory"
    r"|i (?:do not|don't) have it stored"
    r"|je n'ai pas (?:ça|cela)(?: en mémoire)?"
    r"|je ne (?:sais|connais) pas(?: ça| cela)?"
    r")\b"
)
_GREETING = re.compile(
    r"(?i)^(hi|hello|hey|hi there|salut|bonjour|bonsoir|coucou)\b"
)


def polish_reply(
    text: str,
    user: str,
    *,
    context: str = "",
    notes: list[str] | None = None,
    receipt: Receipt | None = None,
    memories: list | None = None,
) -> str:
    """Voice-side guard rails after the model has spoken.

    Piper often mishears a leading "Got it" as "Guy it" or "Cut it";
    "Okay" and "I'll remember that" survive a Piper to Whisper loop.
    An 8B local model sometimes echoed the user's sentence as the
    reply. A generic "I do not have that stored" is dropped when the
    memory block already has the answer.
    """
    cleaned = _rewrite_acks(text)
    cleaned = _drop_reasoning(cleaned)
    has_memory = _has_memory(context, notes, receipt, memories)
    cleaned = _drop_hedges(cleaned, has_memory=has_memory)
    cleaned = _drop_echo(cleaned, user, receipt)
    cleaned = _second_person(cleaned)
    cleaned = _drop_trailing_okay(cleaned)
    cleaned = _confirm_save(cleaned, user, receipt)
    cleaned = _drop_stray_save_ack(cleaned, receipt)
    cleaned = _drop_memory_talk(cleaned, user)
    cleaned = _warm_greeting(cleaned, user, context, notes, receipt, memories)
    cleaned = " ".join(cleaned.split())
    if cleaned:
        return cleaned
    return _fallback_ack(user, receipt, memories, notes)


def _rewrite_acks(text: str) -> str:
    return _GOT_IT.sub("Okay. ", text or "", count=1).strip()


def _drop_reasoning(text: str) -> str:
    """Planning or memory-bookkeeping must never be spoken."""
    sentences = _split_sentences(text)
    kept = [sentence for sentence in sentences if not _REASONING_LEAK.search(sentence)]
    if kept and len(kept) == len(sentences):
        return text
    return " ".join(kept)


def _has_memory(context: str, notes, receipt: Receipt | None, memories) -> bool:
    if notes:
        return True
    if memories:
        return True
    if receipt is not None and (getattr(receipt, "facts", None) or getattr(receipt, "already", None)):
        return True
    block = context or ""
    return "<memory>" in block or "Earlier values:" in block


def _drop_hedges(text: str, *, has_memory: bool) -> str:
    if not has_memory:
        return text
    sentences = _split_sentences(text)
    kept = [sentence for sentence in sentences if not _GENERIC_HEDGE.search(sentence.strip())]
    if kept:
        return " ".join(kept)
    return ""


def _drop_echo(text: str, user: str, receipt: Receipt | None) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return text
    if not _is_echo(user, sentences[0]):
        return text
    rest = " ".join(sentences[1:]).strip()
    if rest and not _is_echo(user, rest):
        return rest
    return ""


def _is_echo(user: str, reply: str) -> bool:
    left = _words(user)
    right = _words(reply)
    if not left or not right:
        return False
    if _GREETING.match(user or "") and _GREETING.match(reply or "") and len(left) <= 3:
        return False
    if left == right:
        return True
    if len(left) >= 4 and right[: len(left)] == left:
        return True
    # A question read back with only pronoun or tense changes
    # ("What did I use to call your dog?").
    if _question_shape(reply) and len(left) >= 5:
        a, b = _content(left), _content(right)
        if a and b and len(a & b) / len(a | b) >= 0.75:
            return True
    return False


def _question_shape(text: str) -> bool:
    stripped = (text or "").strip()
    if "?" in stripped:
        return True
    return bool(re.match(r"(?i)^(what|where|who|when|why|how|do|did|can|could|comment|où|pourquoi)\b", stripped))


def _content(words: list[str]) -> set[str]:
    skip = {"my", "your", "his", "her", "our", "the", "a", "an", "i", "you", "to"}
    found: set[str] = set()
    for word in words:
        if word in skip:
            continue
        found.add("use" if word in {"use", "used"} else word)
    return found


def _words(text: str) -> list[str]:
    return re.findall(r"[a-zà-ÿ0-9']+", (text or "").lower().replace("’", "'"))


_TRAILING_OKAY = re.compile(r"(?i)^(okay|ok)\.?$")
_CONFIRM_VERB = re.compile(
    r"(?i)\b("
    r"okay|ok|d'accord|remember|remembered|noted|updated|saved|it is|"
    r"i'll remember|je vais m'en souvenir|c'est|called|named|name is|"
    r"s'appelle|s'appelait|renamed"
    r")\b"
)
_STRAY_SAVE_ACK = re.compile(
    r"(?i)^(?:"
    r"(?:okay|ok|d'accord)[,.]?\s+(?:then(?:\s+the value)?,?\s+)?(?:c'est\s+)?.+?(?:\s+it is)?|"
    r"(?:okay|ok|d'accord)[,.]?\s+then it is|"
    r"it(?:'s| is) saved|"
    r"that(?:'s| is) saved|"
    r"i(?:'ve| have) saved (?:that|it)"
    r")[.!?]?$"
)
_META_ASK = re.compile(
    r"(?i)^you asked earlier (?:that |if |whether )?"
)


def _drop_trailing_okay(text: str) -> str:
    """A leftover closing Okay is not useful once the fact has been said."""
    sentences = _split_sentences(text)
    if len(sentences) >= 2 and _TRAILING_OKAY.match(sentences[-1].strip()):
        return " ".join(sentences[:-1])
    return text


def _drop_stray_save_ack(text: str, receipt: Receipt | None) -> str:
    """A leftover 'Okay, {value} it is' must not follow a recall answer."""
    if receipt is not None and getattr(receipt, "facts", None):
        return text
    sentences = _split_sentences(text)
    kept = [sentence for sentence in sentences if not _STRAY_SAVE_ACK.match(sentence.strip())]
    return _drop_meta_ask(" ".join(kept))


def _drop_memory_talk(text: str, user: str) -> str:
    """Drop leftover memory bookkeeping once the answer has been said."""
    if _GREETING.match(user or ""):
        return text
    sentences = _split_sentences(text)
    kept = [sentence for sentence in sentences if not _MEMORY_TALK.search(sentence)]
    if kept:
        return " ".join(kept)
    return text


def _drop_meta_ask(text: str) -> str:
    """A spoken answer should not narrate that the user just asked."""
    match = _META_ASK.match(text or "")
    if not match:
        return text
    rest = (text or "")[match.end():].lstrip()
    if not rest:
        return text
    return rest[0].upper() + rest[1:]


_LIFE_EN = (
    "dog|puppy|pet|cat|mum|mom|mother|dad|father|sister|brother|wife|husband|"
    "son|daughter|friend|family|job|employer|role|home|house|apartment|city|"
    "partner|aunt|uncle|cousin|grandma|grandpa|grandmother|grandfather"
)
_MY_LIFE = re.compile(rf"(?i)\bmy ({_LIFE_EN})\b")
_MON_LIFE = re.compile(
    r"(?i)\b(?:mon (chien|frère|frere|père|pere|papa|fils|ami|oncle|cousin|travail|poste|emploi)|"
    r"ma (sœur|soeur|mère|mere|maman|femme|fille|amie|tante|cousine|famille|ville)|"
    r"mes (chiens|parents|enfants|amis))\b"
)
_MEMORY_TALK = re.compile(
    r"(?i)\b(?:stored memories|memory block|in my memory|no stored|"
    r"don't have any stored|do not have any stored|no memories|"
    r"nothing stored|stored details|other stored|"
    r"aucun souvenir|pas de (?:mémoire|memoire|souvenir))\b"
)


def _second_person(text: str) -> str:
    """Maxi talks about the user's life as yours, not mine."""
    def en(match: re.Match[str]) -> str:
        noun = match.group(1)
        lead = "Your" if match.group(0)[0].isupper() else "your"
        return f"{lead} {noun}"

    cleaned = _MY_LIFE.sub(en, text or "")
    def fr(match: re.Match[str]) -> str:
        whole = match.group(0)
        lower = whole.lower()
        if lower.startswith("mon "):
            lead = "Ton" if whole[0].isupper() else "ton"
            return f"{lead} {match.group(1)}"
        if lower.startswith("ma "):
            lead = "Ta" if whole[0].isupper() else "ta"
            return f"{lead} {match.group(2)}"
        lead = "Tes" if whole[0].isupper() else "tes"
        return f"{lead} {match.group(3)}"

    return _MON_LIFE.sub(fr, cleaned)


def _confirm_save(text: str, user: str, receipt: Receipt | None) -> str:
    """A save ack must name the value that was actually written."""
    facts = list(getattr(receipt, "facts", None) or [])
    if not facts:
        return text
    if _mentions_saved(text, facts) and _CONFIRM_VERB.search(text or ""):
        return text
    return _save_ack(user, facts)


def _mentions_saved(text: str, facts: list) -> bool:
    lowered = (text or "").casefold()
    for fact in facts:
        value = str(getattr(fact, "value", "") or "").strip()
        if value and value.casefold() in lowered:
            return True
    return False


def _save_ack(user: str, facts: list) -> str:
    french = _looks_french(user)
    if len(facts) == 1 and _short_value(getattr(facts[0], "value", "")):
        kind = str(getattr(facts[0], "type", "") or "")
        if kind not in {"health", "constraint"}:
            value = str(facts[0].value).strip()
            if french:
                return f"D'accord, c'est {value}."
            return f"Okay, {value} it is."
    summaries = "; ".join(
        str(getattr(fact, "summary", "") or getattr(fact, "value", "")).strip()
        for fact in facts
        if str(getattr(fact, "summary", "") or getattr(fact, "value", "")).strip()
    )
    if not summaries:
        return "D'accord." if french else "Okay."
    if french:
        return f"Je vais m'en souvenir. {summaries}."
    return f"I'll remember that. {summaries}."


def _short_value(value: str) -> bool:
    tokens = (value or "").split()
    return 0 < len(tokens) <= 3 and len(value.strip()) <= 40


def _warm_greeting(
    text: str,
    user: str,
    context: str,
    notes,
    receipt: Receipt | None,
    memories,
) -> str:
    if not _GREETING.match(user or ""):
        return text
    if not _MEMORY_TALK.search(text or "") and " ".join((text or "").split()):
        return text
    french = _looks_french(user)
    match = _GREETING.match(text or "") or _GREETING.match(user or "")
    greet = match.group(0) if match else ("Bonjour" if french else "Hello")
    greet = greet[0].upper() + greet[1:]
    if french:
        return f"{greet}, je suis là."
    return f"{greet}. I'm here."


def _fallback_ack(user: str, receipt: Receipt | None, memories=None, notes=None) -> str:
    french = _looks_french(user)
    prior = [str(note).strip() for note in (notes or []) if str(note).strip()]
    if prior and _question_shape(user):
        value = prior[-1]
        if french:
            return f"Avant, c'était {value}."
        return f"That was previously {value}."
    facts = list(getattr(receipt, "facts", None) or [])
    if facts:
        return _save_ack(user, facts)
    return "Je suis là." if french else "Hello. I'm here." if _GREETING.match(user or "") else "Okay."


def _looks_french(text: str) -> bool:
    return re.search(
        r"(?i)\b(?:je|tu|il|elle|nous|vous|mon|ma|mes|ton|ta|chien|appelle|souvenir|"
        r"d'accord|bonjour|salut|bonsoir|coucou)\b",
        text or "",
    ) is not None


def _user_name(memories: list, receipt: Receipt) -> str:
    """The name slot, from this turn's write or the recalled facts."""
    found = list(getattr(receipt, "facts", None) or []) + list(memories or [])
    for fact in found:
        if getattr(fact, "key", "") == "fact:user-name" and str(fact.value).strip():
            return str(fact.value).strip()
    return ""


_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "\U0000FE00-\U0000FE0F"
    "\U0001F1E6-\U0001F1FF"
    "]+"
)
_BOLD = re.compile(r"\*{1,3}([^*]+)\*{1,3}")
_ITALIC = re.compile(r"(?<![A-Za-z0-9])_([^_]+)_(?![A-Za-z0-9])")
_CODE = re.compile(r"`+([^`]+)`+")
_HEADING = re.compile(r"(?m)^#+\s*")


def _strip_markup(text: str) -> str:
    """Drop markdown and emoji so Piper does not read them aloud."""
    cleaned = _CODE.sub(r"\1", text or "")
    cleaned = _BOLD.sub(r"\1", cleaned)
    cleaned = _ITALIC.sub(r"\1", cleaned)
    cleaned = _HEADING.sub("", cleaned)
    cleaned = _EMOJI.sub("", cleaned)
    return cleaned


def _drop_user_maxi(text: str) -> str:
    """A leading "Hi Maxi" is the model addressing the user as itself."""
    match = _ADDRESS.match(text)
    if match is None:
        return text
    greeting = match.group(1)
    punct = match.group(2) or ""
    rest = text[match.end() :].lstrip()
    if not rest:
        return greeting
    if punct == ",":
        return f"{greeting}, {rest}"
    if punct:
        return f"{greeting}{punct} {rest}"
    return f"{greeting} {rest}"


def _drop_second_answer(text: str) -> str:
    text = _cut_glued(text)
    sentences = _split_sentences(text)
    if not sentences:
        return ""
    kept = [sentences[0]]
    for sentence in sentences[1:]:
        if _RESTART.match(sentence):
            break
        kept.append(sentence)
    return " ".join(kept)


def _cut_glued(text: str) -> str:
    """Keep the first answer when the next one starts with no space."""
    for match in _GLUED.finditer(text):
        punct = match.start() + 1
        if _is_abbrev(text, punct):
            continue
        return text[: punct + 1]
    return text


def _is_abbrev(text: str, punct: int) -> bool:
    prefix = text[:punct]
    if _ABBREV_TAIL.search(prefix):
        return True
    word = re.search(r"([A-Za-z]+)$", prefix)
    return bool(word and word.group(1).lower() in _ABBREV)


def _split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char in ".!?" and index > 0 and text[index - 1].islower():
            if char == "." and _is_abbrev(text, index):
                index += 1
                continue
            nxt = text[index + 1] if index + 1 < len(text) else ""
            if nxt and not nxt.isspace() and not nxt.isupper():
                index += 1
                continue
            end = index + 1
            while end < len(text) and text[end] in ".!?":
                end += 1
            piece = text[start:end].strip()
            if piece:
                sentences.append(piece)
            start = end
            index = end
            continue
        index += 1
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences
