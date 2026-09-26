from medagent.tools.decorators import tool


@tool()
async def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def test_tool_derives_name_and_description() -> None:
    assert add_numbers.name == "add_numbers"
    assert add_numbers.description == "Add two numbers."


def test_tool_derives_json_schema() -> None:
    schema = add_numbers.parameters
    assert schema["type"] == "object"
    assert schema["properties"]["a"] == {"type": "integer"}
    assert set(schema["required"]) == {"a", "b"}


async def test_tool_run_returns_success_result() -> None:
    result = await add_numbers.run(a=2, b=3)
    assert result.success is True
    assert result.data == 5


async def test_tool_run_captures_exception() -> None:
    @tool()
    async def always_fails() -> None:
        raise ValueError("boom")

    result = await always_fails.run()
    assert result.success is False
    assert "boom" in (result.error or "")
