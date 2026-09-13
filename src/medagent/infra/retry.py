"""Async retry decorator with exponential backoff and jitter."""

import asyncio
import functools
import random
from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


def retry(
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            attempt = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except exceptions:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    delay += random.uniform(0, delay * 0.1)
                    await asyncio.sleep(delay)

        return wrapper

    return decorator
