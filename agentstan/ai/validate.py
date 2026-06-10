"""
AI-powered model validation.

Check whether a generated simulation spec actually matches
the user's original description.
"""

import json
from typing import Dict, Any, List, Optional


DEFAULT_MODEL = "gpt-5.5"


def validate(
    spec: Dict[str, Any],
    description: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate a simulation spec against a natural language description.

    Args:
        spec: Simulation specification dict.
        description: Original user description of what the model should do.
        model: OpenAI model name.
        api_key: OpenAI API key.

    Returns:
        Dict with 'valid' (bool), 'issues' (list of strings), 'suggestions' (list).
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key) if api_key else OpenAI()

    # Strip behavior_code to reduce tokens (just show function signatures)
    compact_spec = _compact_spec(spec)

    prompt = f"""You are validating an agent-based model specification against the user's description.

User wanted: "{description}"

Generated specification:
{json.dumps(compact_spec, indent=2)}

Check:
1. Does the spec include all agent types the user described?
2. Do the behaviors match what was described? (movement, interactions, lifecycle)
3. Are the parameters reasonable? (counts, energy, reproduction rates)
4. Is anything missing or wrong?

Return a JSON object with:
- "valid": true/false
- "issues": list of specific problems found (empty if valid)
- "suggestions": list of improvements (even if valid)"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    import json as _json
    try:
        result = _json.loads(response.choices[0].message.content)
    except _json.JSONDecodeError:
        result = {"valid": False, "issues": ["Failed to parse validation response"], "suggestions": []}

    return {
        "valid": result.get("valid", False),
        "issues": result.get("issues", []),
        "suggestions": result.get("suggestions", []),
    }


def _compact_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Strip long behavior_code to just function signatures for token efficiency."""
    compact = {
        "environment": spec.get("environment", {}),
        "agent_types": {},
    }
    if "metadata" in spec:
        compact["metadata"] = spec["metadata"]

    for name, config in spec.get("agent_types", {}).items():
        agent_info = {
            "initial_count": config.get("initial_count"),
            "initial_state": config.get("initial_state"),
        }
        if "behavior" in config:
            # Declarative rules are compact JSON — include them whole
            agent_info["behavior"] = config["behavior"]
        else:
            # Include just the first few lines of legacy behavior code
            code = config.get("behavior_code", "")
            lines = code.strip().split("\n")
            if len(lines) > 15:
                agent_info["behavior_code_preview"] = "\n".join(lines[:15]) + "\n    ..."
            else:
                agent_info["behavior_code"] = code
        compact["agent_types"][name] = agent_info

    return compact
