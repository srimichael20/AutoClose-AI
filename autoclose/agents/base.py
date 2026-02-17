"""Base agent interface and utilities."""

from abc import ABC, abstractmethod
from typing import Any

from autoclose.schemas import WorkflowState


class BaseAgent(ABC):
    """Abstract base for all workflow agents."""

    name: str = "base"

    @abstractmethod
    async def process(self, state: WorkflowState) -> dict[str, Any]:
        """
        Process the current state and return state updates.
        Implementations must return a dict of fields to merge into state.
        """
        pass
