from abc import ABC, abstractmethod

import numpy as np


class BaseAIProvider(ABC):
    """Abstract base for AI analysis providers.

    Concrete implementations receive a preprocessed frame and return a
    structured analysis string (or JSON).  The intake pipeline is
    intentionally decoupled from the provider so intake mode and AI
    backend can be swapped independently.
    """

    @abstractmethod
    async def analyze_frame(self, frame: np.ndarray) -> str:
        """Analyse a single frame and return a human-readable or JSON result.

        Args:
            frame: A numpy ndarray — either grayscale (H, W) from
                   PhoneBrowserIntake or BGR (H, W, 3) from
                   CaptureCardIntake.

        Returns:
            A string describing the game state inferred from the frame.
        """
        ...
