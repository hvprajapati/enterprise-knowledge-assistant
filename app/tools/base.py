"""Abstract base class for all tools in the agent framework.

Every tool — whether it's a calculator, web search, or database
query — must inherit from ``BaseTool``.  This ensures a uniform
interface that ``ToolRegistry`` and ``ToolExecutor`` can rely on
without knowing the specifics of each tool.

Design principles
-----------------
1. **Single-method interface.**  ``invoke()`` is the only entry point.
   The executor never calls anything else on a tool instance.
2. **Self-describing.**  ``name``, ``description``, and ``schema``
   let the planner and LLM-based selectors know what the tool does
   without importing it.
3. **Schema is a JSON Schema dict.**  This decouples tools from
   Pydantic — future tools can define their input contract in plain
   JSON Schema and the executor validates against it.
4. **Stateless.**  Tools should not hold mutable state between
   invocations.  If state is needed, pass it via arguments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Abstract tool that every concrete tool must implement.

    Subclass and provide:

    - ``name`` — unique, kebab-case identifier (e.g. ``"calculator"``)
    - ``description`` — one-sentence explanation for the LLM/planner
    - ``schema`` — JSON Schema dict describing the expected arguments
    - ``invoke(**kwargs)`` — execute the tool and return a dict result
    """

    # ------------------------------------------------------------------
    # subclasses MUST define these
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier (kebab-case, e.g. ``"document-search"``)."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for LLM-based tool selection.

        Should answer: "What does this tool do and when should I use it?"
        """
        ...

    @property
    @abstractmethod
    def schema(self) -> dict[str, Any]:
        """JSON Schema for the tool's input arguments.

        Example::

            {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate."
                    }
                },
                "required": ["expression"],
            }
        """
        ...

    # ------------------------------------------------------------------
    # subclasses MUST implement this
    # ------------------------------------------------------------------

    @abstractmethod
    def invoke(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the tool and return the result as a dict.

        Parameters
        ----------
        **kwargs
            Keyword arguments matching the tool's ``schema``.

        Returns
        -------
        dict
            Must contain at least a ``"result"`` key with the primary
            output.  May include additional metadata keys.

        Raises
        ------
        ValueError
            When required arguments are missing or invalid.
        RuntimeError
            When the tool encounters an execution error.
        """
        ...
