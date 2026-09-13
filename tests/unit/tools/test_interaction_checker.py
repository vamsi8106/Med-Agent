from unittest.mock import AsyncMock

from medagent.core.models import Medication
from medagent.core.types import InteractionSeverity
from medagent.tools.custom.interaction_checker import InteractionCheckerTool


class _FakeResearchClient:
    def __init__(self, response_text: str) -> None:
        self.comprehensive_analysis = AsyncMock(return_value=[_Block(response_text)])

    async def __aenter__(self) -> "_FakeResearchClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None


class _Block:
    def __init__(self, text: str) -> None:
        self.text = text


async def test_interaction_checker_flags_major_severity() -> None:
    fake_client = _FakeResearchClient("This is a major interaction risk.")
    checker = InteractionCheckerTool(research_client=fake_client)  # type: ignore[arg-type]

    result = await checker.run([Medication(name="Metformin"), Medication(name="Glimepiride")])

    assert result.success is True
    interactions = result.data
    assert len(interactions) == 1
    assert interactions[0].severity == InteractionSeverity.MAJOR
    assert interactions[0].drug_a == "Metformin"
    assert interactions[0].drug_b == "Glimepiride"


async def test_interaction_checker_no_pairs_for_single_drug() -> None:
    fake_client = _FakeResearchClient("n/a")
    checker = InteractionCheckerTool(research_client=fake_client)  # type: ignore[arg-type]

    result = await checker.run([Medication(name="Metformin")])

    assert result.success is True
    assert result.data == []
