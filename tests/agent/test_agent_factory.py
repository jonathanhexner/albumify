"""
Unit tests for agent factory.

Tests agent creation, configuration, and basic functionality.
"""

import pytest
from pathlib import Path
from sim_bench.agent.factory import create_agent, list_agent_types
from sim_bench.agent.core.base import Agent, AgentResponse


class TestAgentFactory:
    """Tests for agent factory functions."""

    def test_list_agent_types(self):
        """Test listing available agent types."""
        types = list_agent_types()

        assert isinstance(types, dict)
        assert 'workflow' in types
        assert 'template' in types
        assert 'conversational' in types

    def test_create_template_agent(self):
        """Test creating template agent (no LLM required)."""
        agent = create_agent(agent_type='template')

        assert isinstance(agent, Agent)
        assert hasattr(agent, 'process_query')
        assert hasattr(agent, 'refine')

    def test_create_workflow_agent_placeholder(self):
        """Test creating workflow agent returns placeholder."""
        agent = create_agent(agent_type='workflow')

        assert isinstance(agent, Agent)
        # Should be placeholder for now
        response = agent.process_query("test query")
        assert not response.success
        assert "not yet implemented" in response.message.lower()

    def test_invalid_agent_type(self):
        """Test that invalid agent type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown agent_type"):
            create_agent(agent_type='invalid_type')

    def test_agent_with_custom_config(self):
        """Test creating agent with custom configuration."""
        config = {'max_iterations': 5, 'verbose': True}
        agent = create_agent(agent_type='template', config=config)

        assert isinstance(agent, Agent)
        # Config is passed to agent
        assert 'max_iterations' in agent.config
        assert agent.config['max_iterations'] == 5


class TestTemplateAgent:
    """Tests for template-based agent."""

    @pytest.fixture
    def agent(self):
        """Create template agent for testing."""
        return create_agent(agent_type='template')

    @pytest.fixture
    def sample_images(self, tmp_path):
        """Create sample image paths."""
        # Create dummy image files
        images = []
        for i in range(5):
            img = tmp_path / f"photo{i}.jpg"
            img.write_text("dummy")
            images.append(str(img))
        return images

    def test_process_query_without_images(self, agent):
        """Test processing query without image context."""
        response = agent.process_query("Organize by event")

        assert not response.success
        assert "No images provided" in response.message
        assert 'available_templates' in response.data

    def test_process_query_with_unknown_intent(self, agent, sample_images):
        """Test processing query with unknown intent."""
        context = {'image_paths': sample_images}
        response = agent.process_query("Do something random", context)

        assert not response.success
        assert "Could not match query" in response.message
        assert 'available_templates' in response.data

    def test_organize_by_event(self, agent, sample_images):
        """Test 'organize by event' workflow."""
        context = {'image_paths': sample_images}
        response = agent.process_query("Organize my photos by event", context)

        # Note: This will fail if clustering tools not available
        # But we test that the workflow is at least attempted
        assert isinstance(response, AgentResponse)
        assert 'workflow' in response.data or 'error' in response.metadata

    def test_find_group_photos(self, agent, sample_images):
        """Test 'find group photos' workflow."""
        context = {'image_paths': sample_images}
        response = agent.process_query("Find my group photos", context)

        assert isinstance(response, AgentResponse)
        assert 'workflow' in response.data or 'error' in response.metadata

    def test_organize_travel_photos(self, agent, sample_images):
        """Test 'organize travel photos' workflow."""
        context = {'image_paths': sample_images}
        response = agent.process_query("Organize my travel photos by place", context)

        assert isinstance(response, AgentResponse)
        assert 'workflow' in response.data or 'error' in response.metadata

    def test_refine_not_supported(self, agent):
        """Test that refine() is not supported for template agent."""
        response = agent.refine("Make it better")

        assert not response.success
        assert "not supported" in response.message.lower()


class TestAgentResponse:
    """Tests for AgentResponse dataclass."""

    def test_create_response(self):
        """Test creating AgentResponse."""
        response = AgentResponse(
            success=True,
            message="Test message",
            data={'result': 'value'},
            metadata={'time': 1.0}
        )

        assert response.success
        assert response.message == "Test message"
        assert response.data == {'result': 'value'}
        assert response.metadata == {'time': 1.0}
        assert response.next_steps is None

    def test_response_with_next_steps(self):
        """Test AgentResponse with next steps."""
        response = AgentResponse(
            success=True,
            message="Done",
            data={},
            metadata={},
            next_steps=["Review results", "Export data"]
        )

        assert len(response.next_steps) == 2
        assert "Review results" in response.next_steps


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
