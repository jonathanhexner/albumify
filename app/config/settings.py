"""Application settings and configuration."""

from dataclasses import dataclass
from typing import Dict
from .constants import AgentType


@dataclass
class AgentConfig:
    """Configuration for an agent type."""
    name: str
    description: str
    requires_api: bool
    available: bool


@dataclass
class AppConfig:
    """Main application configuration."""
    title: str = "📸 AI Photo Organization"
    subtitle: str = "Organize your photos using natural language"
    layout: str = "wide"
    page_icon: str = "📸"


class Settings:
    """Application settings singleton."""

    # App configuration
    APP = AppConfig()

    # Agent configurations
    AGENTS: Dict[AgentType, AgentConfig] = {
        AgentType.TEMPLATE: AgentConfig(
            name="Template Agent",
            description="Uses keyword matching for pre-built workflows. No API key needed.",
            requires_api=False,
            available=True
        ),
        AgentType.WORKFLOW: AgentConfig(
            name="Workflow Agent",
            description="LLM plans custom workflows. Requires API key.",
            requires_api=True,
            available=False
        ),
        AgentType.CONVERSATIONAL: AgentConfig(
            name="Conversational Agent",
            description="Full LLM conversation. Requires API key.",
            requires_api=True,
            available=False
        )
    }

    @classmethod
    def get_available_agents(cls) -> Dict[AgentType, AgentConfig]:
        """Get only available agent types."""
        return {
            agent_type: config
            for agent_type, config in cls.AGENTS.items()
            if config.available
        }
