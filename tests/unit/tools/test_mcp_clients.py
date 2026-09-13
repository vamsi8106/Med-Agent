from unittest.mock import AsyncMock

from medagent.core.config import Settings
from medagent.tools.mcp.healthcare import HealthcareMCPClient
from medagent.tools.mcp.medical import MedicalMCPClient
from medagent.tools.mcp.research import ResearchMCPClient


def _settings() -> Settings:
    return Settings(_env_file=None)


async def test_medical_client_search_drugs_calls_correct_tool() -> None:
    client = MedicalMCPClient(settings=_settings())
    client.call_tool = AsyncMock(return_value="result")

    result = await client.search_drugs("metformin")

    client.call_tool.assert_awaited_once_with("search-drugs", {"query": "metformin"})
    assert result == "result"


async def test_healthcare_client_clinical_trials_search() -> None:
    client = HealthcareMCPClient(settings=_settings())
    client.call_tool = AsyncMock(return_value="trials")

    result = await client.clinical_trials_search("diabetes", phase="phase_2")

    client.call_tool.assert_awaited_once_with(
        "clinical_trials_search", {"condition": "diabetes", "phase": "phase_2"}
    )
    assert result == "trials"


async def test_research_client_drug_safety_profile() -> None:
    client = ResearchMCPClient(settings=_settings())
    client.call_tool = AsyncMock(return_value="profile")

    result = await client.drug_safety_profile("metformin")

    client.call_tool.assert_awaited_once_with(
        "research_drug_safety_profile", {"drugName": "metformin"}
    )
    assert result == "profile"


async def test_research_client_comprehensive_analysis() -> None:
    client = ResearchMCPClient(settings=_settings())
    client.call_tool = AsyncMock(return_value="analysis")

    result = await client.comprehensive_analysis("Metformin", "Glimepiride interaction")

    client.call_tool.assert_awaited_once_with(
        "research_comprehensive_analysis",
        {"drugName": "Metformin", "condition": "Glimepiride interaction"},
    )
    assert result == "analysis"
