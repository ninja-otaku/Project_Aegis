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
    async def analyze_frame(
        self,
        frame: np.ndarray,
        system_prompt: str | None = None,
    ) -> dict:
        """Analyse a single frame and return a structured result dict.

        Args:
            frame: A numpy ndarray — either grayscale (H, W) from
                   PhoneBrowserIntake or BGR (H, W, 3) from
                   CaptureCardIntake.
            system_prompt: Optional per-call system prompt override.
                   When provided (e.g. from a game profile), implementations
                   should use it in place of their stored default prompt.

        Returns:
            A dict with at minimum the following keys:
              game_state    (str)       — brief description of current state
              threats       (list[str]) — immediate threats or opportunities
              recommendation (str)      — single recommended action
              confidence    (str)       — "low" | "medium" | "high"
        """
        ...
