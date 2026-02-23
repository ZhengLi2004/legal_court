"""Event history/stream service extracted from API manager facade."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional

from mas.session.event_stream import (
    get_event_history as get_session_event_history,
)
from mas.session.event_stream import (
    register_event_subscriber as register_session_event_subscriber,
)
from mas.session.event_stream import (
    unregister_event_subscriber as unregister_session_event_subscriber,
)


class EventService:
    """Encapsulate event query and subscriber operations per session."""

    def __init__(self, *, get_session: Callable[[str], Any]):
        """Create event service dependencies."""
        self._get_session = get_session

    def get_event_history(
        self,
        session_id: str,
        limit: int = 100,
        from_seq: Optional[int] = None,
        to_seq: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return filtered event history for one session."""
        session = self._get_session(session_id)

        return get_session_event_history(
            session.events,
            limit=limit,
            from_seq=from_seq,
            to_seq=to_seq,
        )

    def register_event_subscriber(
        self, session_id: str, max_queue_size: int = 200
    ) -> asyncio.Queue:
        """Register one live event queue subscriber."""
        session = self._get_session(session_id)

        return register_session_event_subscriber(
            session.event_subscribers,
            max_queue_size=max_queue_size,
        )

    def unregister_event_subscriber(
        self, session_id: str, queue: asyncio.Queue
    ) -> None:
        """Remove one previously registered queue subscriber."""
        session = self._get_session(session_id)
        unregister_session_event_subscriber(session.event_subscribers, queue)
