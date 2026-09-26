from medagent.core.models import PatientContext
from medagent.tools.custom.patient_context import PatientContextTool


async def test_get_missing_patient_returns_none() -> None:
    tool = PatientContextTool()
    result = await tool.run(action="get", patient_id="P-TEST-001")
    assert result.success is True
    assert result.data is None


async def test_save_then_get_round_trips() -> None:
    tool = PatientContextTool()
    context = PatientContext(id="P-TEST-001", name="Patient Alpha", age=54, sex="F")

    save_result = await tool.run(action="save", patient_id=context.id, context=context)
    assert save_result.success is True
    assert save_result.data.created_at is not None

    get_result = await tool.run(action="get", patient_id=context.id)
    assert get_result.data.id == "P-TEST-001"


async def test_unknown_action_returns_failure() -> None:
    tool = PatientContextTool()
    result = await tool.run(action="delete", patient_id="P-TEST-001")
    assert result.success is False
    assert "Unknown" in (result.error or "")
