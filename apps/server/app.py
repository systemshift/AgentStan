"""
AgentStan reference HTTP server.

This is a demo/reference deployment shell, NOT part of the agentstan
library. It depends on the installed `agentstan` package; real production
servers (auth, multi-tenancy, persistence) should live in their own repo
and do the same.

Run with: python apps/server/app.py
Requires: pip install agentstan flask flask-cors python-dotenv
"""

import os
import sys
import uuid
import time
import copy
import json
import threading
import logging

from flask import Flask, request, jsonify
from flask_cors import CORS

from agentstan.core.simulation import Simulation
from agentstan.core.observer import Observer
from agentstan.core.intervention import InterventionEngine

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serialize import clean_for_json, results_to_dict, simulation_to_dict

logger = logging.getLogger("agentstan.server")

# --- State ---
sessions = {}       # session_id -> simulation session state
simulations = {}    # session_id -> Simulation object (for intervention)
SESSION_TTL = 600


def _cleanup():
    now = time.time()
    expired = [sid for sid, s in sessions.items() if now - s.get("created", 0) > SESSION_TTL]
    for sid in expired:
        sessions.pop(sid, None)
        simulations.pop(sid, None)


def create_app(cors_origins=None, api_key=None):
    """Flask application factory."""
    app = Flask(__name__)

    origins = cors_origins or os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
    CORS(app, origins=origins)

    # Store config on app
    app.config["OPENAI_API_KEY"] = api_key or os.getenv("OPENAI_API_KEY")
    app.config["OPENAI_MODEL"] = os.getenv("OPENAI_MODEL", "gpt-5.5")

    # --- Health ---

    @app.get("/api/health")
    def health():
        import agentstan
        return jsonify({
            "status": "ok",
            "version": agentstan.__version__,
        })

    # --- Generate spec from prompt ---

    @app.post("/api/generate")
    def generate():
        data = request.get_json()
        prompt = data.get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "prompt is required"}), 400

        model = data.get("model", app.config["OPENAI_MODEL"])
        key = data.get("api_key", app.config["OPENAI_API_KEY"])

        try:
            from agentstan.ai.generate import generate as generate_spec
            spec = generate_spec(prompt, model=model, api_key=key)
            sim = Simulation(spec)
            initial_counts = sim.agent_manager.get_counts()
            return jsonify({"spec": clean_for_json(spec), "initial_counts": initial_counts})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # --- Run simulation (streaming via polling) ---

    @app.post("/api/simulate")
    def simulate():
        data = request.get_json()
        spec = data.get("spec")
        steps = data.get("steps", 200)
        delay = data.get("delay", 0.05)

        if not spec:
            return jsonify({"error": "spec is required"}), 400

        _cleanup()
        session_id = str(uuid.uuid4())

        sessions[session_id] = {
            "status": "starting",
            "current_step": 0,
            "total_steps": steps,
            "history": [],
            "error": None,
            "created": time.time(),
        }

        def run():
            try:
                sim = Simulation(spec)
                simulations[session_id] = sim

                # Attach intervention engine so API can intervene
                engine = InterventionEngine(sim)
                sim.attach_intervention_engine(engine)

                sessions[session_id]["status"] = "running"

                for state in sim.run_stream(steps=steps, delay=delay):
                    if session_id not in sessions:
                        return

                    sessions[session_id]["current_step"] = state["step"]
                    sessions[session_id]["history"].append({
                        "step": state["step"],
                        "agent_counts": state.get("agent_counts", {}),
                        "total_agents": state.get("total_agents", 0),
                    })

                if session_id in sessions:
                    sessions[session_id]["status"] = "completed"

            except Exception as e:
                if session_id in sessions:
                    sessions[session_id]["status"] = "error"
                    sessions[session_id]["error"] = str(e)

        threading.Thread(target=run, daemon=True).start()
        return jsonify({"session_id": session_id})

    @app.get("/api/simulate/<session_id>")
    def simulate_status(session_id):
        session = sessions.get(session_id)
        if not session:
            return jsonify({"error": "Session not found"}), 404
        return jsonify(session)

    # --- Intervene in running simulation ---

    @app.post("/api/simulate/<session_id>/intervene")
    def intervene(session_id):
        """Apply interventions to a running simulation."""
        sim = simulations.get(session_id)
        if not sim:
            return jsonify({"error": "Simulation not found or not running"}), 404

        if not sim.intervention_engine:
            return jsonify({"error": "No intervention engine attached"}), 500

        data = request.get_json()
        interventions = data.get("interventions", [])
        applied = []

        for i in interventions:
            action = i.get("action")
            try:
                if action == "add_agents":
                    agent_type = i.get("agent_type")
                    count = min(i.get("count", 1), 50)
                    type_spec = sim.spec.get("agent_types", {}).get(agent_type, {})
                    state = copy.deepcopy(type_spec.get("initial_state", {"energy": 20}))
                    code = type_spec.get("behavior_code", "")
                    for _ in range(count):
                        sim.intervention_engine.add_agent(agent_type, state, behavior_code=code, source="api")
                    applied.append(i)

                elif action == "remove_agents":
                    agent_type = i.get("agent_type")
                    count = min(i.get("count", 1), 50)
                    import random
                    agents = sim.agent_manager.get_agents_by_type(agent_type)
                    for a in random.sample(agents, min(count, len(agents))):
                        sim.intervention_engine.remove_agent(a.id, source="api")
                    applied.append(i)

                elif action == "modify_environment":
                    sim.intervention_engine.modify_environment(
                        i.get("property"), i.get("value"), source="api"
                    )
                    applied.append(i)

            except Exception as e:
                logger.error(f"Intervention failed: {e}")

        return jsonify({"applied": applied, "count": len(applied)})

    # --- Batch run ---

    @app.post("/api/batch")
    def batch():
        """Run a batch experiment."""
        data = request.get_json()
        spec = data.get("spec")
        n_runs = min(data.get("n_runs", 10), 100)
        steps = min(data.get("steps", 200), 1000)
        vary = data.get("vary")

        if not spec:
            return jsonify({"error": "spec is required"}), 400

        try:
            from agentstan.experiment.batch import batch_run
            results = batch_run(spec, n_runs=n_runs, steps=steps, vary=vary, max_workers=4)
            return jsonify({"results": clean_for_json(results), "count": len(results)})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # --- Analyze results ---

    @app.post("/api/analyze")
    def analyze():
        """Analyze simulation results."""
        data = request.get_json()
        results = data.get("results")
        if not results:
            return jsonify({"error": "results is required"}), 400

        try:
            from agentstan.analysis.population import analyze as analyze_population
            from agentstan.analysis.events import analyze_events
            report = {
                "population": analyze_population(results),
                "events": analyze_events(results),
            }
            return jsonify(clean_for_json(report))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # --- AI interpret ---

    @app.post("/api/interpret")
    def interpret_results():
        """AI-powered result interpretation."""
        data = request.get_json()
        results = data.get("results")
        if not results:
            return jsonify({"error": "results is required"}), 400

        key = data.get("api_key", app.config["OPENAI_API_KEY"])
        model = data.get("model", app.config["OPENAI_MODEL"])

        try:
            from agentstan.ai.interpret import interpret
            explanation = interpret(results, model=model, api_key=key)
            return jsonify({"explanation": explanation})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # --- Save / Load ---

    @app.post("/api/simulate/<session_id>/save")
    def save_simulation(session_id):
        """Get simulation checkpoint as JSON."""
        sim = simulations.get(session_id)
        if not sim:
            return jsonify({"error": "Simulation not found"}), 404

        return jsonify(simulation_to_dict(sim))

    @app.post("/api/simulate/load")
    def load_simulation():
        """Create a simulation session from a checkpoint."""
        data = request.get_json()
        checkpoint = data.get("checkpoint")
        if not checkpoint:
            return jsonify({"error": "checkpoint is required"}), 400

        try:
            # Save to temp file, load via Simulation.load
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(checkpoint, f)
                path = f.name

            sim = Simulation.load(path)
            os.unlink(path)

            session_id = str(uuid.uuid4())
            simulations[session_id] = sim
            sessions[session_id] = {
                "status": "loaded",
                "current_step": sim.step,
                "total_steps": 0,
                "history": [],
                "error": None,
                "created": time.time(),
            }

            return jsonify({"session_id": session_id, "step": sim.step})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


def main():
    """CLI entry point for agentstan-server."""
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(prog="agentstan-server", description="AgentStan API server")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 5000)))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--cors", default=None, help="Comma-separated CORS origins")
    args = parser.parse_args()

    cors = args.cors.split(",") if args.cors else None
    app = create_app(cors_origins=cors)

    print(f"AgentStan server starting on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
