"""
Event logging system for comprehensive simulation tracking

Tracks all agent actions, interactions, state changes, and system events
for detailed post-simulation analysis.
"""

from typing import Dict, List, Any, Optional


LOG_LEVELS = ("minimal", "normal", "detailed")


class EventLogger:
    """
    Centralized event logging for simulations

    Tracks everything that happens in the simulation for analysis and debugging.
    """

    def __init__(self, enabled: bool = True, log_level: str = "normal"):
        """
        Initialize event logger

        Args:
            enabled: Whether logging is active
            log_level: "minimal" (births and deaths), "normal" (plus
                interactions and global/environment changes; the default),
                or "detailed" (plus every agent action and state change —
                large: several events per agent per step)
        """
        if log_level not in LOG_LEVELS:
            raise ValueError(f"log_level must be one of {LOG_LEVELS}, got {log_level!r}")
        self.enabled = enabled
        self.log_level = log_level
        self.events: List[Dict[str, Any]] = []

    def log_event(self, step: int, event_type: str, **kwargs):
        """
        Log a simulation event

        Args:
            step: Current simulation step
            event_type: Type of event (agent_action, interaction, state_change, etc.)
            **kwargs: Event-specific data
        """
        if not self.enabled:
            return

        event = {
            "step": step,
            "type": event_type,
            **kwargs
        }

        self.events.append(event)

    def log_agent_action(self, step: int, agent_id: int, agent_type: str,
                        action_type: str, details: Optional[Dict] = None):
        """Log an agent performing an action"""
        if self.log_level != "detailed":
            return

        self.log_event(
            step=step,
            event_type="agent_action",
            agent_id=agent_id,
            agent_type=agent_type,
            action=action_type,
            details=details or {}
        )

    def log_interaction(self, step: int, agent_ids: List[int],
                       interaction_type: str, outcome: str,
                       details: Optional[Dict] = None):
        """Log an interaction between agents"""
        if self.log_level == "minimal":
            return
        self.log_event(
            step=step,
            event_type="interaction",
            agents=agent_ids,
            interaction_type=interaction_type,
            outcome=outcome,
            details=details or {}
        )

    def log_state_change(self, step: int, agent_id: int, attribute: str,
                        old_value: Any, new_value: Any, cause: str = ""):
        """Log a change in agent state"""
        if self.log_level != "detailed":
            return

        self.log_event(
            step=step,
            event_type="state_change",
            agent_id=agent_id,
            attribute=attribute,
            old_value=old_value,
            new_value=new_value,
            cause=cause
        )

    def log_agent_birth(self, step: int, parent_id: Optional[int],
                       child_id: int, agent_type: str):
        """Log creation of a new agent"""
        self.log_event(
            step=step,
            event_type="agent_birth",
            parent_id=parent_id,
            child_id=child_id,
            agent_type=agent_type
        )

    def log_agent_death(self, step: int, agent_id: int, agent_type: str,
                       cause: str = ""):
        """Log death of an agent"""
        self.log_event(
            step=step,
            event_type="agent_death",
            agent_id=agent_id,
            agent_type=agent_type,
            cause=cause
        )

    def log_global_change(self, step: int, name: str,
                          old_value: Any, new_value: Any):
        """Log a change to a shared global"""
        if self.log_level == "minimal":
            return

        self.log_event(
            step=step,
            event_type="global_change",
            name=name,
            old_value=old_value,
            new_value=new_value
        )

    def log_environment_change(self, step: int, property_name: str,
                              old_value: Any, new_value: Any):
        """Log a change in environment state"""
        if self.log_level == "minimal":
            return

        self.log_event(
            step=step,
            event_type="environment_change",
            property=property_name,
            old_value=old_value,
            new_value=new_value
        )

    def get_events(self, step: Optional[int] = None,
                  event_type: Optional[str] = None) -> List[Dict]:
        """
        Retrieve logged events with optional filtering

        Args:
            step: Filter by specific step
            event_type: Filter by event type

        Returns:
            List of event dictionaries
        """
        events = self.events

        if step is not None:
            events = [e for e in events if e["step"] == step]

        if event_type is not None:
            events = [e for e in events if e["type"] == event_type]

        return events

    def get_agent_timeline(self, agent_id: int) -> List[Dict]:
        """Get all events related to a specific agent"""
        return [
            e for e in self.events
            if (e.get("agent_id") == agent_id or
                agent_id in e.get("agents", []))
        ]

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of logged events"""
        if not self.events:
            return {"total_events": 0}

        event_types = {}
        for event in self.events:
            event_type = event["type"]
            event_types[event_type] = event_types.get(event_type, 0) + 1

        return {
            "total_events": len(self.events),
            "event_types": event_types,
            "steps_logged": max(e["step"] for e in self.events) if self.events else 0
        }

    def clear(self):
        """Clear all logged events"""
        self.events = []

    def export_json(self) -> List[Dict]:
        """Export events as JSON-serializable list"""
        return self.events.copy()

    def export_csv_data(self) -> List[Dict]:
        """
        Export events in flattened format suitable for CSV

        Returns:
            List of flat dictionaries (one per event)
        """
        csv_data = []

        for event in self.events:
            flat_event = {
                "step": event["step"],
                "event_type": event["type"]
            }

            # Add event-specific fields
            for key, value in event.items():
                if key not in ["step", "type"]:
                    # Convert complex types to strings
                    if isinstance(value, (list, dict)):
                        flat_event[key] = str(value)
                    else:
                        flat_event[key] = value

            csv_data.append(flat_event)

        return csv_data
