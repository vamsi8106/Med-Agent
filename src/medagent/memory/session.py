"""Sliding-window conversation buffer for a single agent session."""

from medagent.core.models import Message


class SessionMemory:
    def __init__(self, max_messages: int = 20) -> None:
        self._max_messages = max_messages
        self._messages: list[Message] = []

    def add_message(self, message: Message) -> None:
        self._messages.append(message)
        if len(self._messages) > self._max_messages:
            self._messages = self._messages[-self._max_messages :]

    def get_messages(self) -> list[Message]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages = []
