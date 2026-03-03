from abc import ABC, abstractmethod

import numpy as np


class BaseAIProvider(ABC):
    """Abstract base for AI analysis providers.

    Concrete implementations receive a preprocessed frame and return a
    structured analysis dict.  The intake pipeline is intentionally
    decoupled from the provider so intake mode and AI backend can be
    swapped independently.
    """

    @abstractmethod
    async def analyze_frame(self, frame: np.ndarray) -> dict:
        """Analyse a single frame and return a structured result dict.

        Args:
            frame: A numpy ndarray — either grayscale (H, W) from
                   PhoneBrowserIntake or BGR (H, W, 3) from
                   CaptureCardIntake.

        Returns:
            A dict with at minimum the following keys:
              game_state    (str)       — brief description of current state
              threats       (list[str]) — immediate threats or opportunities
              recommendation (str)      — single recommended action
              confidence    (str)       — "low" | "medium" | "high"
        """
        ...
