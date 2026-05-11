"""Tests for output models and AgentArtifact contract (these should fail before implementation)."""


def test_image_output_exists():
    """ImageOutput should exist and be importable."""
    from src.meta_agent.api_models import ImageOutput

    output = ImageOutput(
        url="http://api.example.com/artifacts/chart-123.png",
        caption="Revenue Trend",
        alt_text="Bar chart showing revenue by quarter",
    )
    assert output.type == "image"
    assert output.url == "http://api.example.com/artifacts/chart-123.png"
    assert output.caption == "Revenue Trend"


def test_agent_artifact_model_exists():
    """AgentArtifact should be importable and have required fields."""
    from src.meta_agent.api_models import AgentArtifact

    artifact = AgentArtifact(
        id="chart-001",
        kind="chart",
        path="/app/charts/chart_20260502_120000_000000.png",
        filename="chart_20260502_120000_000000.png",
        mime_type="image/png",
        caption="Analysis Results",
    )
    assert artifact.id == "chart-001"
    assert artifact.kind == "chart"
    assert artifact.mime_type == "image/png"


def test_meta_agent_state_has_outputs_field():
    """MetaAgentState should have outputs and artifacts fields with reducers."""
    from src.meta_agent.utils.state import MetaAgentState

    state = MetaAgentState(
        question="What is the trend?",
        outputs=[],
        artifacts=[],
    )
    assert hasattr(state, "outputs")
    assert hasattr(state, "artifacts")
    assert state.outputs == []
    assert state.artifacts == []


def test_outputs_reducer_appends():
    """The outputs reducer should append new outputs to the state."""
    from src.meta_agent.utils.state import MetaAgentState
    from src.meta_agent.api_models import TextOutput, ImageOutput

    state = MetaAgentState(
        question="test",
        outputs=[TextOutput(text="Initial")],
    )

    # Simulate what a reducer should do: append new outputs
    new_outputs = state.outputs + [ImageOutput(url="http://example.com/chart.png")]
    assert len(new_outputs) == 2
    assert new_outputs[0].type == "text"
    assert new_outputs[1].type == "image"


def test_artifacts_reducer_appends():
    """The artifacts reducer should append new artifacts to the state."""
    from src.meta_agent.utils.state import MetaAgentState
    from src.meta_agent.api_models import AgentArtifact

    artifact1 = AgentArtifact(
        id="art-1",
        kind="chart",
        path="/charts/c1.png",
        filename="c1.png",
        mime_type="image/png",
    )

    state = MetaAgentState(
        question="test",
        artifacts=[artifact1],
    )

    artifact2 = AgentArtifact(
        id="art-2",
        kind="csv",
        path="/artifacts/data.csv",
        filename="data.csv",
        mime_type="text/csv",
    )

    new_artifacts = state.artifacts + [artifact2]
    assert len(new_artifacts) == 2
    assert new_artifacts[0].kind == "chart"
    assert new_artifacts[1].kind == "csv"


def test_meta_agent_result_has_outputs():
    """MetaAgentResult should return thread_id and outputs, not answer."""
    from src.meta_agent.graph import MetaAgentResult
    from src.meta_agent.api_models import TextOutput

    result = MetaAgentResult(
        thread_id="thread-123",
        outputs=[TextOutput(text="Analysis complete")],
    )
    assert result.thread_id == "thread-123"
    assert len(result.outputs) == 1
    assert result.outputs[0].type == "text"


def test_ask_endpoint_returns_outputs():
    """The /ask endpoint should return MetaAgentApiResponse with outputs list, not wrapped answer."""
    from src.meta_agent.api_models import MetaAgentApiResponse, TextOutput

    response = MetaAgentApiResponse(
        thread_id="t-1",
        outputs=[
            TextOutput(text="Analysis results"),
        ],
    )
    assert response.thread_id == "t-1"
    assert len(response.outputs) >= 1
    assert all(hasattr(o, "type") for o in response.outputs)
