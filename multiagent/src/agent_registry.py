from pathlib import Path
from typing import Any, Dict, Union

import yaml


class AgentRegistry:
    """Load and validate the CSS v3 agent configuration."""

    REQUIRED_FIELDS = {"type", "gpu"}

    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Agent configuration not found: {self.config_path}"
            )

        with self.config_path.open() as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict) or "agents" not in config:
            raise ValueError("Configuration must contain an 'agents' section.")

        self.agents: Dict[str, Dict[str, Any]] = config["agents"]
        self._validate()

    def _validate(self) -> None:
        if not self.agents:
            raise ValueError("No agents configured.")

        for name, agent in self.agents.items():
            missing = self.REQUIRED_FIELDS - agent.keys()
            if missing:
                raise ValueError(
                    f"{name}: missing required fields: {sorted(missing)}"
                )

            agent_type = agent["type"]

            if agent_type == "live":
                if not agent.get("model"):
                    raise ValueError(f"{name}: live agent requires a model.")

                if agent.get("gpu") is None:
                    raise ValueError(f"{name}: live agent requires a GPU.")

            elif agent_type == "sigma":
                if not agent.get("pre_generated", False):
                    raise ValueError(
                        "Sigma must be marked as pre_generated."
                    )

            else:
                raise ValueError(
                    f"{name}: unsupported agent type '{agent_type}'."
                )

    def get(self, name: str) -> Dict[str, Any]:
        if name not in self.agents:
            raise KeyError(f"Unknown agent: {name}")
        return self.agents[name]

    def live_agents(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: agent
            for name, agent in self.agents.items()
            if agent["type"] == "live"
        }

    def sigma(self) -> Dict[str, Any]:
        return self.get("SG")

    def gpu_agents(self, gpu_id: int) -> Dict[str, Dict[str, Any]]:
        return {
            name: agent
            for name, agent in self.live_agents().items()
            if agent["gpu"] == gpu_id
        }

    def __len__(self) -> int:
        return len(self.agents)
