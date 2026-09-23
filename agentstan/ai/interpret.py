"""
AI-powered result interpretation.

Send simulation results to an LLM and get a natural language
explanation of what happened and why.
"""

import json
from typing import Dict, Any, Optional


from .llm import make_client, resolve_model


def interpret(
    results: Dict[str, Any],
    analysis: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    client=None,
) -> str:
    """
    Generate a natural language interpretation of simulation results.

    Args:
        results: Results dict from Simulation.run()
        analysis: Optional analysis dict from agentstan.analysis.analyze_population()
        model: Model name (default: $AGENTSTAN_MODEL, else gpt-5.5)
        api_key: API key (or set OPENAI_API_KEY)
        base_url, client: any OpenAI-compatible endpoint or client

    Returns:
        Natural language explanation string.
    """
    client = client or make_client(api_key, base_url)
    model = resolve_model(model)

    summary = results.get("summary", {})
    history = results.get("metrics", {}).get("history", [])

    # Build a compact representation for the LLM
    context = {
        "initial_counts": summary.get("initial_counts", {}),
        "final_counts": summary.get("final_counts", {}),
        "steps": results.get("final_step", 0),
        "duration_seconds": round(results.get("duration", 0), 2),
    }
    if results.get("globals"):
        context["final_globals"] = results["globals"]
    if results.get("stopped"):
        context["stopped_early"] = results["stopped"]
    description = (results.get("spec") or {}).get("metadata", {}).get("description")
    if description:
        context["model_description"] = description

    # Add population snapshots (sampled to keep token count low)
    if history:
        sample_points = _sample_history(history, max_points=10)
        context["snapshots"] = sample_points

    if analysis:
        context["analysis"] = analysis

    prompt = f"""You are analyzing the results of an agent-based model simulation.

Here are the results:
{json.dumps(context, indent=2)}

Write a clear, concise analysis (3-5 sentences) explaining:
1. What happened in the simulation (population trends, observables such as
   prices or money supply, key events)
2. Why it happened (based on the dynamics you can infer)
3. Whether the outcome is realistic or surprising

Be specific about numbers and steps. Don't hedge — make direct observations."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "text"},
    )

    return response.choices[0].message.content.strip()


def _snapshot(h: dict) -> dict:
    snap = {"step": h["step"], "counts": h["agent_counts"]}
    if h.get("observables"):
        snap["observables"] = h["observables"]
    if h.get("globals"):
        snap["globals"] = h["globals"]
    return snap


def _sample_history(history: list, max_points: int = 10) -> list:
    """Sample evenly-spaced points from history to keep tokens low."""
    if len(history) <= max_points:
        return [_snapshot(h) for h in history]

    step = len(history) / max_points
    indices = [int(i * step) for i in range(max_points)]
    indices.append(len(history) - 1)  # always include last

    return [_snapshot(history[i]) for i in sorted(set(indices))]
