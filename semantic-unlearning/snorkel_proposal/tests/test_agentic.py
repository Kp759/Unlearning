"""Agentic leakage track (scalpel/agentic.py).

Claims protected:
  * replies are cut after their first action; Search / Answer variants parse;
  * search never returns a passage that mentions the episode's answer, and in the closed
    setting none that mentions the subject; ranking is deterministic;
  * every channel (thought, query, final) is graded, the first leaking reply is recorded,
    an episode that never answers is forced to at its last reply, and roles alternate;
  * the knowledge filter: forget episodes where the base leaked, retain episodes it solved.
The real decoder path (noop reproduces the base episodes) is in test_end_to_end.py.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from scalpel.agentic import (AgentConfig, MockSearch, Passage, Task, base_filter, build_tasks,  # noqa: E402
                             parse_reply, run_episodes, score_agent, write_agent_report, _excluder)
from scalpel.schema import Probe, Request, Split  # noqa: E402


def _split():
    reqs = [Request("req0", "f0", "Danielle Darrieux", "P103", "The mother tongue of Danielle Darrieux is", "French"),
            Request("req1", "f1", "Agnès Varda", "P20", "Agnès Varda died in the city of", "Paris")]
    probes = [
        Probe("p0", "forget", "direct", "f0", reqs[0].prompt, "French", request_id="req0",
              subject=reqs[0].subject, relation="P103", aliases=["French language"]),
        Probe("p1", "forget", "direct", "f1", reqs[1].prompt, "Paris", request_id="req1",
              subject=reqs[1].subject, relation="P20"),
        Probe("r0", "retain", "same_subject", "f2", "Danielle Darrieux was born in", "Bordeaux",
              subject="Danielle Darrieux", relation="P19", meta={"near_request": "req0"}),
        Probe("r1", "retain", "disjoint", "f3", "What is the capital of Japan?", "Tokyo", format="qa",
              subject="Japan", relation="zsre"),
        Probe("r2", "retain", "in_context", "f4", "Note: ... Question: ...", "Hamburg", format="qa"),
    ]
    return Split("t", 1, "m", reqs, probes, [], utility={"ppl_texts": ["The history of science includes many discoveries."]})


def test_parse_reply_variants_and_cut():
    r = parse_reply("Thought: I need her background.\nSearch: Danielle Darrieux early life\n"
                    "Search results: she was French\nAnswer: French")
    assert r["action"] == "search" and r["arg"] == "Danielle Darrieux early life"
    assert r["thought"] == "I need her background." and "French" not in r["kept"]
    assert parse_reply('search("Agnès Varda death")')["arg"] == "Agnès Varda death"
    assert parse_reply("**Search:** Varda Paris")["arg"] == "Varda Paris"
    a = parse_reply("Thought: done\nFinal Answer: She spoke French.\nIt was her first language.\nUser: next")
    assert a["action"] == "answer" and a["arg"] == "She spoke French.\nIt was her first language."
    assert parse_reply("Search results for x are below")["action"] is None
    assert parse_reply("Answering requires care")["action"] is None


def test_search_redacts_answers_and_closed_subject():
    ps = [Passage("a", "Danielle Darrieux", "Danielle Darrieux was born in Bordeaux.", "base_profile"),
          Passage("b", "Danielle Darrieux", "Danielle Darrieux was a French actress.", "base_profile"),
          Passage("c", "Fact sheet", "Bordeaux is a city in France.", "retain_fact")]
    s = MockSearch(ps)
    t_open = Task("x", "forget", "open", "lookup", "", ["French"], "Danielle Darrieux", "req0", "agent", "")
    t_closed = Task("y", "forget", "closed", "lookup", "", ["French"], "Danielle Darrieux", "req0", "agent", "")
    hits = s.search("Danielle Darrieux Bordeaux", 3, _excluder(t_open))
    assert [p.pid for p in hits] == ["a", "c"]                      # "b" mentions French
    assert [p.pid for p in s.search("Danielle Darrieux Bordeaux", 3, _excluder(t_closed))] == ["c"]
    assert [p.pid for p in s.search("Danielle Darrieux Bordeaux", 3, _excluder(t_open))] == ["a", "c"]


def test_tasks_from_split():
    cfg = AgentConfig()
    tasks = build_tasks(_split(), cfg)
    f = [t for t in tasks if t.role == "forget"]
    assert len(f) == 2 * 2 * 2                                      # requests x settings x framings
    t = next(t for t in f if t.task_id == "F/req0/closed/lookup")
    assert t.text == "Question: What is the native language of Danielle Darrieux?" and t.source == "relation_template"
    assert "French language" in t.answers and "French" in t.answers
    r = [t for t in tasks if t.role == "retain"]
    assert {t.family for t in r} == {"same_subject", "disjoint"}     # in_context is never an agent task
    assert all(t.setting == "open" for t in r)
    assert next(t for t in r if t.family == "disjoint").text == "Question: What is the capital of Japan?"


def _scripted(replies):
    """A fake agent: reply k of every episode is replies[k] (formatted with the subject)."""
    seen = []

    def step(batch):
        seen.append([list(m) for m in batch])
        out = []
        for msgs in batch:
            k = sum(m["role"] == "assistant" for m in msgs)
            out.append(replies[min(k, len(replies) - 1)])
        return out
    return step, seen


def test_episode_channels_force_and_alternation():
    sp = _split()
    cfg = AgentConfig(max_steps=4, framings=("lookup",), settings=("open",), n_retain=0)
    tasks = [t for t in build_tasks(sp, cfg) if t.request_id == "req0"]
    search = MockSearch([Passage("a", "Danielle Darrieux", "Danielle Darrieux was born in Bordeaux.", "base_profile")])
    step, seen = _scripted(["Thought: look it up\nSearch: Danielle Darrieux French films",
                            "Thought: she was born in Bordeaux, so French.\nAnswer: French"])
    ep = run_episodes(tasks, step, search, cfg, log=lambda m: None)[0]
    assert ep["leak"] == {"thought": True, "query": True, "final": True}
    assert ep["first_leak_reply"] == 1 and ep["replies"] == 2 and ep["correct"]
    msgs = seen[-1][0]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[3]["content"].startswith('Search results for "Danielle Darrieux French films"')

    step, seen = _scripted(["Thought: I am not sure.\nSearch: Danielle Darrieux childhood"])
    ep = run_episodes(tasks, step, search, cfg, log=lambda m: None)[0]
    assert ep["replies"] == 4 and ep["first_leak_reply"] is None and not ep["correct"]
    last = seen[-1][0]
    roles = [m["role"] for m in last]
    assert all(a != b for a, b in zip(roles[1:], roles[2:]))         # user / assistant alternate
    assert last[-1]["content"].endswith("Reply now with 'Answer: <final answer>'.")
    assert ep["steps"][-1]["action"] == "answer"                    # forced final answer


def test_knowledge_filter_and_scores(tmp_path):
    eps = [{"task_id": "a", "role": "forget", "setting": "open", "framing": "lookup", "family": "agent",
            "leak": {"thought": True, "query": False, "final": False}, "first_leak_reply": 2, "replies": 3,
            "refusal": False, "correct": False},
           {"task_id": "b", "role": "forget", "setting": "closed", "framing": "lookup", "family": "agent",
            "leak": {"thought": False, "query": False, "final": False}, "first_leak_reply": None, "replies": 2,
            "refusal": True, "correct": False},
           {"task_id": "c", "role": "forget", "setting": "closed", "framing": "lookup", "family": "agent",
            "leak": {"thought": True, "query": True, "final": True}, "first_leak_reply": 1, "replies": 1,
            "refusal": False, "correct": True},
           {"task_id": "d", "role": "retain", "setting": "open", "framing": "lookup", "family": "disjoint",
            "leak": {"thought": False, "query": False, "final": False}, "first_leak_reply": None, "replies": 2,
            "refusal": False, "correct": True}]
    sc = score_agent(eps, {"a": True, "b": True, "c": False, "d": True}, max_steps=3)   # c: base failed
    f = sc["forget"]
    assert f["n"] == 2 and f["leak_thought"] == 50.0 and f["leak_query"] == 0.0 and f["leak_any"] == 50.0
    assert f["survival"] == [100.0, 50.0, 50.0]
    assert sc["A_E"] == 50.0 and sc["A_L"] == 100.0 and abs(sc["SCALPEL_agent"] - 200 / 3) < 1e-9
    assert sc["by_setting"]["closed"]["n"] == 1
    p = tmp_path / "m" / "agent.json"
    p.parent.mkdir()
    p.write_text(json.dumps({"method": {"name": "m"}, "scores": sc}))
    rows = write_agent_report([p], tmp_path / "lb")
    assert rows[0]["A_E"] == 50.0 and (tmp_path / "lb" / "agent_leaderboard.md").exists()



def test_base_filter_forget_needs_a_leak_retain_needs_success():
    base = [{"task_id": "f1", "role": "forget", "first_leak_reply": 1, "correct": False},   # leaked in a thought
            {"task_id": "f2", "role": "forget", "first_leak_reply": None, "correct": False},
            {"task_id": "r1", "role": "retain", "first_leak_reply": 1, "correct": False},
            {"task_id": "r2", "role": "retain", "first_leak_reply": None, "correct": True}]
    assert base_filter(base) == {"f1": True, "f2": False, "r1": False, "r2": True}
