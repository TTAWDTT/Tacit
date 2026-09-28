"""Phase-aware protocol adapter for the pinned HiddenBench simulator.

This module deliberately reuses the pinned benchmark's task model, agent
state, prompt templates, and voting/scoring conventions. It adds only the
phase-specific discussion schedule that the upstream CLI does not expose.
"""
from __future__ import annotations

from collections import Counter
import random
from typing import Any

from hiddenbench.benchmark import BenchmarkTask
from hiddenbench.models import ModelClient
from hiddenbench.prompts import PromptSet, load_prompts, render_prompt
from hiddenbench.simulator import (
    Profile,
    collect_final_votes,
    instantiate_agents,
)


PROTOCOLS = {"natural_3", "exchange_decide", "reveal_all_3"}
ROUNDS = 3


def phase_instruction(protocol: str, round_index: int) -> str:
    """Return the frozen, one-indexed discussion instruction for a turn."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol: {protocol}")
    if round_index < 0 or round_index >= ROUNDS:
        raise ValueError(f"round_index must be in [0, {ROUNDS - 1}]")
    if protocol != "exchange_decide":
        return ""
    if round_index < 2:
        return (
            f"\n\nEXCHANGE PHASE — ROUND {round_index + 1} OF 2. "
            "Share one or two decision-relevant facts that you personally received. "
            "Identify the option that currently appears to lead, if one does, and give "
            "one evidence-based reason it could be wrong. Separate stated facts from "
            "your inference. Do not claim access to facts you have not received. "
            "Keep the message concise."
        )
    return (
        "\n\nDECIDE PHASE — FINAL DISCUSSION PASS. Summarize the strongest evidence "
        "for the leading option, the strongest counterevidence, and any uncertainty "
        "that could change the decision. Then state which option the evidence supports; "
        "you will cast a separate final vote after this pass."
    )


def _chat_prompt(
    *,
    prompt_set: PromptSet,
    round_index: int,
    previous_messages: list[str],
    initial_vote: dict[str, str] | None,
    protocol: str,
) -> str:
    phase = phase_instruction(protocol, round_index)
    if round_index == 0 and initial_vote is not None:
        prefix = (
            f"Your initial vote was: {initial_vote['vote']}\n"
            f"Your initial rationale was: {initial_vote['rationale']}\n\n"
        )
        if not previous_messages:
            return prefix + prompt_set.first_user_prompt + phase
        return prefix + render_prompt(
            prompt_set.user_prompt,
            {"messages": "\n".join(previous_messages), "extra": phase},
        )
    return render_prompt(
        prompt_set.user_prompt,
        {"messages": "\n".join(previous_messages), "extra": phase},
    )


def _append_reveal_all(response: str, facts: list[str]) -> str:
    """Mechanistically append every fact available to this sender."""
    lines = "\n".join(f"- {fact}" for fact in facts)
    return f"{response}\n\nInformation disclosed verbatim from this agent's profile:\n{lines}"


def run_scenario(
    task: BenchmarkTask,
    model_client: ModelClient,
    profile: Profile | str,
    protocol: str,
    *,
    prompts: PromptSet | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run one task with the same vote and sequential-round structure as upstream."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {sorted(PROTOCOLS)}")
    profile = Profile(profile)
    prompt_set = prompts or load_prompts()
    rng = random.Random(seed)
    agents, assignments = instantiate_agents(task, model_client, profile, prompt_set, rng)

    run_data: dict[str, Any] = {
        "task_id": task.id,
        "scenario": task.name,
        "profile": profile.value,
        "model": model_client.model_name,
        "protocol": protocol,
        "num_agents": len(agents),
        "num_special": 0,
        "correct_answer": task.correct_answer,
        "possible_answers": list(task.possible_answers),
        "fact_assignments": assignments,
        "initial_votes": [],
        "rounds": [],
        "discussion_transcript": [],
        "final_votes": [],
        "majority_vote": None,
        "seed": seed,
    }

    initial_vote_prompt = render_prompt(
        prompt_set.first_vote_prompt,
        {
            "group_discussion": "No discussion has occurred yet. This is your initial vote based solely on the information available to you.",
            "possible_answers": ", ".join(task.possible_answers),
        },
    )
    for agent in agents:
        vote = agent.vote(initial_vote_prompt, task.possible_answers)
        run_data["initial_votes"].append({"agent": agent.name, **vote})

    transcript: list[dict[str, Any]] = []
    for round_index in range(ROUNDS):
        round_data = {"round": round_index + 1, "messages": [], "votes": []}
        if round_index == 0:
            prior: list[str] = []
            for agent_index, agent in enumerate(agents):
                prompt = _chat_prompt(
                    prompt_set=prompt_set,
                    round_index=round_index,
                    previous_messages=prior,
                    initial_vote=run_data["initial_votes"][agent_index],
                    protocol=protocol,
                )
                response = agent.chat(prompt)
                if protocol == "reveal_all_3":
                    response = _append_reveal_all(response, assignments[agent_index]["visible_facts"])
                    agent.history[-1]["content"] = response
                message = {"agent": agent.name, "prompt": prompt, "response": response}
                round_data["messages"].append(message)
                transcript.append({"round": round_index + 1, "agent": agent.name, "response": response})
                prior.append(f"{agent.name}: {response}")
        else:
            for agent_index, agent in enumerate(agents):
                prior = [
                    f"{other.name}: {other.history[-1]['content']}"
                    for offset in range(1, len(agents))
                    for other in [agents[(agent_index + offset) % len(agents)]]
                ]
                prompt = _chat_prompt(
                    prompt_set=prompt_set,
                    round_index=round_index,
                    previous_messages=prior,
                    initial_vote=None,
                    protocol=protocol,
                )
                response = agent.chat(prompt)
                message = {"agent": agent.name, "prompt": prompt, "response": response}
                round_data["messages"].append(message)
                transcript.append({"round": round_index + 1, "agent": agent.name, "response": response})

        if round_index == ROUNDS - 1:
            round_data["votes"] = collect_final_votes(
                agents, prompt_set, task.possible_answers, round_data["messages"]
            )
        run_data["rounds"].append(round_data)

    run_data["discussion_transcript"] = transcript
    run_data["final_votes"] = run_data["rounds"][-1]["votes"]
    counts = Counter(vote["vote"] for vote in run_data["final_votes"])
    if counts:
        run_data["majority_vote"] = counts.most_common(1)[0][0]
    return run_data
