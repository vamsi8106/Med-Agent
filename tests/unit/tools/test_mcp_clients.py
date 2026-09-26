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

    result = await client.clinical_trials_search("diabetes", status="recruiting")

    client.call_tool.assert_awaited_once_with(
        "clinical_trials_search", {"condition": "diabetes", "status": "recruiting"}
    )
    assert result == "trials"


async def test_research_client_drug_safety_profile() -> None:
    client = ResearchMCPClient(settings=_settings())
    client.request = AsyncMock(return_value={"success": True, "data": "profile"})

    result = await client.drug_safety_profile("metformin")

    client.request.assert_awaited_once_with(
        "POST", "/api/analysis/safety", json={"drugName": "metformin"}
    )
    assert result == {"success": True, "data": "profile"}


async def test_research_client_comprehensive_analysis() -> None:
    client = ResearchMCPClient(settings=_settings())
    client.request = AsyncMock(return_value={"success": True, "data": "analysis"})

    result = await client.comprehensive_analysis("Metformin", "Glimepiride interaction")

    client.request.assert_awaited_once_with(
        "POST",
        "/api/analysis/comprehensive",
        json={"drugName": "Metformin", "condition": "Glimepiride interaction"},
    )
    assert result == {"success": True, "data": "analysis"}
