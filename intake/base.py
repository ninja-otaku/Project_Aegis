from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np


class BaseVideoIntake(ABC):
    """Abstract base for all video intake sources."""

    @abstractmethod
    async def start(self) -> None:
        """Initialise hardware / connections and begin capturing frames."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Tear down hardware / connections and release all resources."""
        ...

    @abstractmethod
    def get_latest_frame(self) -> np.ndarray | None:
        """Return the most-recently captured frame, or None if none yet.

        Callers receive a copy — mutating the returned array is safe.
        """
        ...

    def get_frame_timestamp(self) -> datetime | None:
        """Return the UTC timestamp of the latest frame, or None if none yet.

        Subclasses should override this to expose timing information.
        """
        return None
