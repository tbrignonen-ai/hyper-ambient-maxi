"""Append-only memory journal with a snapshot and a read-after-write check.

On 2026-11-05 the AI Friend 1.8.3 three-harness run kept a dog's name and a
coffee preference, and lost three other writes:

- a family diagnosis, because the model answered and never called
  ``friend_remember``;
- a new job, because ``friend_remember`` ran and stored 0 facts;
- a pet rename, which never superseded the old name.

This store is the part that does not depend on the model. A commit lands
in the journal, is fsynced, is materialized to ``state.json``, and is
read back by a second open of the same directory before the caller is
told it stuck. A crash between the journal fsync and the snapshot rename
is repaired by replaying the journal tail on the next open.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

# Pronoun objects produced the junk memory "Remembered: that" in the
# 1.8.3 Linux report (finding 2). Refuse them at the store, not only
# in the extractor, so a model proposal cannot sneak one in later.
_PRONOUN_VALUES = {"that", "it", "this", "them", "those", "these", "ça", "ca", "cela"}
_PRONOUN_KEYS = {"fact:that", "fact:it", "fact:this"}


class MemoryWriteError(RuntimeError):
    """A commit could not be read back from disk."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _norm(value: str) -> str:
    return " ".join(value.lower().split())


def _best_thread(threads: list[Thread], hint: str) -> Thread | None:
    from hyper_ambient_maxi.memory.textutil import overlap_score, terms

    query = terms(hint)
    best: Thread | None = None
    best_score = 0.0
    for thread in threads:
        score = overlap_score(query, f"{thread.summary} {thread.title}")
        if score > best_score:
            best = thread
            best_score = score
    if best is None or best_score < 0.15:
        return None
    return best


def _put_ids(data: bytes) -> set[str]:
    """Ids the journal claims exist: facts and opened threads."""
    found: set[str] = set()
    for line in data.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if record.get("op") == "put":
            found.add(record["fact"]["id"])
        elif record.get("op") == "open_thread":
            found.add(record["thread"]["id"])
    return found


def _fact_id(key: str, value: str, ts: str) -> str:
    digest = sha256(f"{key}|{value}|{ts}".encode()).hexdigest()[:12]
    return f"mem_{digest}"


@dataclass
class Fact:
    id: str
    key: str
    type: str
    summary: str
    value: str
    status: str
    created_at: str
    source: str
    source_agent: str
    conversation_id: str
    evidence: str
    superseded_by: str | None = None
    # Kept for ranking. Never copied into the model prompt.
    importance: float = 0.8
    confidence: float = 0.85
    tags: list[str] = field(default_factory=list)
    thread_id: str | None = None

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> Fact:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ProposedFact:
    key: str
    type: str
    summary: str
    value: str
    evidence: str
    importance: float = 0.8
    tags: list[str] = field(default_factory=list)
    thread_id: str | None = None
    confidence: float = 0.85


@dataclass
class ProposedThread:
    id: str
    title: str
    summary: str
    priority: str = "normal"


@dataclass
class Thread:
    id: str
    title: str
    summary: str
    status: str
    priority: str
    created_at: str
    closed_at: str | None = None

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> Thread:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Correction:
    key: str
    old_value: str
    new_value: str


@dataclass
class Receipt:
    stored: int
    verified: bool
    facts: list[Fact] = field(default_factory=list)
    already: list[Fact] = field(default_factory=list)
    corrections: list[Correction] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    dropped_pronouns: int = 0
    threads_opened: list[str] = field(default_factory=list)
    threads_closed: list[str] = field(default_factory=list)


def _is_pronoun_junk(prop: ProposedFact) -> bool:
    if prop.key in _PRONOUN_KEYS:
        return True
    return _norm(prop.value) in _PRONOUN_VALUES


class MemoryStore:
    def __init__(self, home: Path | str):
        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.home / "journal.jsonl"
        self.snapshot_path = self.home / "state.json"
        self.facts: dict[str, Fact] = {}
        self.threads: dict[str, Thread] = {}
        self._journal_bytes = 0
        self._load()

    def _load(self) -> None:
        self.facts = {}
        self.threads = {}
        watermark = 0
        if self.snapshot_path.exists():
            try:
                raw = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                raw = {}
            watermark = int(raw.get("journal_bytes") or 0)
            for item in raw.get("memories") or []:
                try:
                    fact = Fact.from_json(item)
                except TypeError:
                    continue
                self.facts[fact.id] = fact
            for item in raw.get("threads") or []:
                try:
                    thread = Thread.from_json(item)
                except TypeError:
                    continue
                self.threads[thread.id] = thread
        data = b""
        if self.journal_path.exists():
            data = self.journal_path.read_bytes()
        self._journal_bytes = len(data)
        # The journal is the source of truth. A snapshot that claims to
        # be caught up but is missing a put, or a crash tail past the
        # watermark, is rebuilt from the whole journal.
        journal_ids = _put_ids(data)
        loaded = set(self.facts) | set(self.threads)
        caught_up = watermark <= len(data) and journal_ids <= loaded
        tail = b"" if watermark > len(data) else data[watermark:]
        if not caught_up or tail.strip():
            self.facts = {}
            self.threads = {}
            if data.strip():
                self._apply_journal_bytes(data)
            self._journal_bytes = len(data)
            self._write_snapshot()

    def _apply_journal_bytes(self, blob: bytes) -> None:
        for line in blob.decode("utf-8").splitlines():
            line = line.strip()
            if line:
                self._apply_record(json.loads(line))

    def _apply_record(self, rec: dict) -> None:
        op = rec.get("op")
        if op == "put":
            fact = Fact.from_json(rec["fact"])
            self.facts[fact.id] = fact
        elif op == "supersede":
            old = self.facts.get(rec["id"])
            if old is not None:
                old.status = "superseded"
                old.superseded_by = rec.get("by")
        elif op == "open_thread":
            thread = Thread.from_json(rec["thread"])
            previous = self.threads.get(thread.id)
            if previous is not None:
                thread.created_at = previous.created_at
            self.threads[thread.id] = thread
        elif op == "close_thread":
            thread = self.threads.get(rec["id"])
            if thread is not None:
                thread.status = "closed"
                thread.closed_at = rec.get("ts")
        else:
            raise MemoryWriteError(f"unknown journal op {op!r}")

    def active(self) -> list[Fact]:
        return [f for f in self.facts.values() if f.status == "active"]

    def slot(self, key: str) -> Fact | None:
        found = [f for f in self.active() if f.key == key]
        if not found:
            return None
        return max(found, key=lambda f: f.created_at)

    def history(self, key: str) -> list[Fact]:
        return sorted(
            (f for f in self.facts.values() if f.key == key),
            key=lambda f: f.created_at,
        )

    def open_threads(self) -> list[Thread]:
        return [thread for thread in self.threads.values() if thread.status == "open"]

    def commit(
        self,
        proposed: list[ProposedFact],
        *,
        source_agent: str,
        conversation_id: str,
        source: str = "steward-extract",
        threads: list[ProposedThread] | None = None,
        closes: list[str] | None = None,
    ) -> Receipt:
        dropped = 0
        kept: list[ProposedFact] = []
        for prop in proposed:
            if _is_pronoun_junk(prop):
                dropped += 1
                continue
            if not prop.value.strip() or not prop.key.strip():
                continue
            kept.append(prop)
        if not kept and not threads and not closes:
            return Receipt(stored=0, verified=True, dropped_pronouns=dropped)

        by_key: dict[str, ProposedFact] = {}
        for prop in kept:
            by_key[prop.key] = prop

        ts = _now()
        records: list[dict] = []
        created: list[Fact] = []
        already: list[Fact] = []
        corrections: list[Correction] = []

        for key, prop in by_key.items():
            current = self.slot(key)
            if current is not None and _norm(current.value) == _norm(prop.value):
                already.append(current)
                continue
            fact = Fact(
                id=_fact_id(key, prop.value, ts),
                key=key,
                type=prop.type,
                summary=prop.summary,
                value=prop.value.strip(),
                status="active",
                created_at=ts,
                source=source,
                source_agent=source_agent,
                conversation_id=conversation_id,
                evidence=prop.evidence[:500],
                importance=prop.importance,
                confidence=prop.confidence,
                tags=list(prop.tags),
                thread_id=prop.thread_id,
            )
            records.append({"op": "put", "fact": fact.to_json(), "ts": ts})
            if current is not None:
                records.append(
                    {"op": "supersede", "id": current.id, "by": fact.id, "ts": ts}
                )
                corrections.append(Correction(key, current.value, fact.value))
            created.append(fact)

        opened, closed, thread_records = self._thread_records(threads or [], closes or [], ts)
        records.extend(thread_records)

        if not records:
            self._verify(already)
            self._verify_threads(opened, closed)
            return Receipt(
                stored=0,
                verified=True,
                facts=[],
                already=already,
                ids=[f.id for f in already],
                dropped_pronouns=dropped,
                threads_opened=opened,
                threads_closed=closed,
            )

        self._append_journal(records)
        for rec in records:
            self._apply_record(rec)
        self._write_snapshot()
        try:
            self._verify(created)
            self._verify_threads(opened, closed)
        except MemoryWriteError:
            self._rebuild_from_journal()
            self._verify(created)
            self._verify_threads(opened, closed)
        return Receipt(
            stored=len(created),
            verified=True,
            facts=created,
            already=already,
            corrections=corrections,
            ids=[f.id for f in created],
            dropped_pronouns=dropped,
            threads_opened=opened,
            threads_closed=closed,
        )

    def _verify(self, facts: list[Fact]) -> None:
        journal = ""
        if self.journal_path.exists():
            journal = self.journal_path.read_text(encoding="utf-8")
        for fact in facts:
            if fact.id not in journal:
                raise MemoryWriteError(
                    f"journal missing {fact.key}={fact.value!r}"
                )
        disk = MemoryStore(self.home)
        for fact in facts:
            got = disk.facts.get(fact.id)
            if got is None or got.status != "active":
                raise MemoryWriteError(
                    f"read-after-write failed for {fact.key}={fact.value!r}"
                )
            active_rows = [f for f in disk.active() if f.key == fact.key]
            if len(active_rows) != 1 or active_rows[0].id != fact.id:
                raise MemoryWriteError(
                    f"slot {fact.key} did not become the single active row"
                )
            if _norm(active_rows[0].value) != _norm(fact.value):
                raise MemoryWriteError(
                    f"slot {fact.key} came back as {active_rows[0].value!r}"
                )

    def _thread_records(
        self,
        threads: list[ProposedThread],
        closes: list[str],
        ts: str,
    ) -> tuple[list[str], list[str], list[dict]]:
        records: list[dict] = []
        opened: list[str] = []
        by_id: dict[str, ProposedThread] = {}
        for thread in threads:
            by_id[thread.id] = thread
        for thread in by_id.values():
            current = self.threads.get(thread.id)
            if (
                current is not None
                and current.status == "open"
                and current.summary == thread.summary
                and current.priority == thread.priority
            ):
                continue
            created_at = current.created_at if current is not None else ts
            row = Thread(
                id=thread.id,
                title=thread.title,
                summary=thread.summary,
                status="open",
                priority=thread.priority,
                created_at=created_at,
            )
            records.append({"op": "open_thread", "thread": row.to_json(), "ts": ts})
            opened.append(thread.id)
        closed: list[str] = []
        for hint in closes:
            match = _best_thread(self.open_threads(), hint)
            if match is None or match.id in closed:
                continue
            records.append({"op": "close_thread", "id": match.id, "ts": ts})
            closed.append(match.id)
        return opened, closed, records

    def _verify_threads(self, opened: list[str], closed: list[str]) -> None:
        if not opened and not closed:
            return
        journal = ""
        if self.journal_path.exists():
            journal = self.journal_path.read_text(encoding="utf-8")
        disk = MemoryStore(self.home)
        for thread_id in opened:
            if thread_id not in journal:
                raise MemoryWriteError(f"journal missing thread {thread_id}")
            got = disk.threads.get(thread_id)
            if got is None or got.status != "open":
                raise MemoryWriteError(f"thread {thread_id} did not stay open")
        for thread_id in closed:
            got = disk.threads.get(thread_id)
            if got is None or got.status != "closed":
                raise MemoryWriteError(f"thread {thread_id} did not close")

    def _rebuild_from_journal(self) -> None:
        self.facts = {}
        self.threads = {}
        data = b""
        if self.journal_path.exists():
            data = self.journal_path.read_bytes()
        self._journal_bytes = 0
        if data:
            self._apply_journal_bytes(data)
        self._journal_bytes = len(data)
        self._write_snapshot()

    def _append_journal(self, records: list[dict]) -> None:
        blob = "".join(
            json.dumps(rec, ensure_ascii=False) + "\n" for rec in records
        ).encode("utf-8")
        fd = os.open(self.journal_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(fd, blob)
            os.fsync(fd)
        finally:
            os.close(fd)
        self._journal_bytes += len(blob)

    def _write_snapshot(self) -> None:
        payload = {
            "version": 1,
            "journal_bytes": self._journal_bytes,
            "memories": [
                f.to_json()
                for f in sorted(self.facts.values(), key=lambda f: (f.created_at, f.id))
            ],
            "threads": [
                thread.to_json()
                for thread in sorted(self.threads.values(), key=lambda item: item.id)
            ],
        }
        raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        tmp = self.snapshot_path.with_suffix(".json.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.write(fd, raw)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, self.snapshot_path)
        dirfd = os.open(self.home, os.O_RDONLY)
        try:
            try:
                os.fsync(dirfd)
            except OSError:
                # Directory fsync is not available on every platform.
                # The file fsync above still covers the bytes.
                pass
        finally:
            os.close(dirfd)
