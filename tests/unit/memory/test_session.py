from medagent.core.models import Message
from medagent.memory.session import SessionMemory


def test_add_and_get_messages() -> None:
    session = SessionMemory()
    session.add_message(Message(role="user", content="hi"))
    assert [m.content for m in session.get_messages()] == ["hi"]


def test_sliding_window_trims_oldest() -> None:
    session = SessionMemory(max_messages=2)
    session.add_message(Message(role="user", content="1"))
    session.add_message(Message(role="user", content="2"))
    session.add_message(Message(role="user", content="3"))
    assert [m.content for m in session.get_messages()] == ["2", "3"]


def test_clear_empties_session() -> None:
    session = SessionMemory()
    session.add_message(Message(role="user", content="hi"))
    session.clear()
    assert session.get_messages() == []
