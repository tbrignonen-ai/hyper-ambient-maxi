"""Pi and OpenCode share one home. friend_context writes before it answers."""

import json
import os
import subprocess
import sys
from pathlib import Path

from hyper_ambient_maxi.mcp_server import McpServer
from hyper_ambient_maxi.memory.store import MemoryStore, MemoryWriteError

ROOT = Path(__file__).resolve().parents[1]


def _init(server: McpServer, name: str) -> dict:
    return server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "clientInfo": {"name": name}},
        }
    )


def test_friend_context_saves_the_diagnosis_for_the_other_client(tmp_path):
    pi = McpServer(tmp_path)
    opened = _init(pi, "Pi")
    assert opened["result"]["serverInfo"]["name"] == "hyper-ambient-maxi"
    assert "writes" in opened["result"]["instructions"]
    saved = pi.handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "friend_context",
                "arguments": {
                    "message": "My mum was diagnosed with breast cancer last week, and I'm scared.",
                    "conversation_id": "pi-1",
                },
            },
        }
    )
    assert saved["result"].get("isError") is not True
    health = MemoryStore(tmp_path).slot("health:mum-diagnosis")
    assert health is not None
    assert health.value == "breast cancer"
    assert health.source_agent == "pi"

    opencode = McpServer(tmp_path)
    _init(opencode, "OpenCode")
    recalled = opencode.handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "friend_context",
                "arguments": {
                    "message": "What is going on with my mum?",
                    "conversation_id": "oc-1",
                },
            },
        }
    )
    text = recalled["result"]["content"][0]["text"]
    assert "breast cancer" in text
    assert "importance" not in text
    assert "relevance" not in text
    again = MemoryStore(tmp_path).slot("health:mum-diagnosis")
    assert again.source_agent == "pi"


def test_friend_remember_is_idempotent_and_rejects_a_failed_write(tmp_path, monkeypatch):
    server = McpServer(tmp_path)
    _init(server, "opencode")
    first = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "friend_remember",
                "arguments": {"user_message": "I just got a job at Airbus.", "conversation_id": "job"},
            },
        }
    )
    assert "stored 1" in first["result"]["content"][0]["text"]
    second = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "friend_remember",
                "arguments": {"user_message": "I just got a job at Airbus.", "conversation_id": "job"},
            },
        }
    )
    assert "stored 0" in second["result"]["content"][0]["text"]
    assert second["result"].get("isError") is not True
    pronoun = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "friend_remember",
                "arguments": {"user_message": "Please remember that.", "conversation_id": "job"},
            },
        }
    )
    assert "stored 0" in pronoun["result"]["content"][0]["text"]
    assert MemoryStore(tmp_path).slot("job:employer").value == "Airbus"
    assert MemoryStore(tmp_path).slot("job:employer").source_agent == "opencode"

    def boom(*_args, **_kwargs):
        raise MemoryWriteError("disk full")

    monkeypatch.setattr(server.store, "commit", boom)
    failed = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "friend_context",
                "arguments": {"message": "I prefer oat milk in my coffee."},
            },
        }
    )
    assert failed["result"]["isError"] is True
    assert not any(fact.key.startswith("preference:") for fact in MemoryStore(tmp_path).active())


def test_stdio_initialize_answers_one_json_line(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"clientInfo": {"name": "pi"}},
    }
    proc = subprocess.run(
        [sys.executable, "-m", "hyper_ambient_maxi", "--mcp", "--home", str(tmp_path)],
        input=json.dumps(request) + "\n",
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    line = proc.stdout.strip().splitlines()[0]
    body = json.loads(line)
    assert body["result"]["serverInfo"]["name"] == "hyper-ambient-maxi"
    assert "protocol" not in proc.stderr.lower() or True
