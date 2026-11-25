"""Run a DataPilot session's async code from synchronous UIs (Streamlit reruns, CLIs).

AutoGen teams and HTTP clients are bound to the event loop they first ran on, so each
session owns ONE long-lived loop in a background thread; the UI submits work to it and
consumes streamed items through a thread-safe queue.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any, AsyncIterator, Awaitable, Iterator, TypeVar

T = TypeVar("T")
_DONE = object()


class LoopThread:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True, name="datapilot-loop")
        self._thread.start()

    def run(self, coro: Awaitable[T], timeout: float | None = None) -> T:
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def iterate(self, agen: AsyncIterator[Any]) -> Iterator[Any]:
        """Turn an async generator running on the loop into a sync iterator (re-raises errors)."""
        q: queue.Queue = queue.Queue()

        async def pump() -> None:
            try:
                async for item in agen:
                    q.put(("item", item))
            except BaseException as e:  # noqa: BLE001 - forwarded to the consumer
                q.put(("error", e))
            finally:
                q.put(("done", _DONE))

        asyncio.run_coroutine_threadsafe(pump(), self.loop)
        while True:
            kind, value = q.get()
            if kind == "item":
                yield value
            elif kind == "error":
                raise value
            else:
                return

    def close(self) -> None:
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)
