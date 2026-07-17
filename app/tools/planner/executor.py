"""Sequential Tool Executor — runs tool plans one invocation at a time.

Executes the ``ToolExecutionPlan`` produced by ``MultiToolPlanner``,
collecting ``ToolResult`` objects for each invocation.

Design
------
- **Sequential.**  Tools run one at a time in plan order.  Future
  support for parallel execution is designed for (``depends_on`` field
  in ``ToolInvocation``) but not yet implemented.
- **Fail-fast on required.**  If a required tool fails, the executor
  stops and returns partial results.  Optional tools that fail are
  skipped with a warning.
- **Measured.**  Each invocation and the total plan are timed.
- **Safe.**  The executor never raises — every failure is captured in
  a ``ToolResult(success=False)``.
"""

from __future__ import annotations

import logging
import time

from app.tools.executor import ToolExecutor
from app.tools.models import ToolResult
from app.tools.planner.models import ToolExecutionPlan

logger = logging.getLogger(__name__)


class SequentialToolExecutor:
    """Execute a ``ToolExecutionPlan`` one tool at a time.

    Usage::

        plan = planner.plan_tools(question)
        executor = SequentialToolExecutor(tool_executor=tool_exec)
        results = executor.execute(plan)
        for r in results:
            print(r.tool_name, r.success)
    """

    def __init__(self, *, tool_executor: ToolExecutor) -> None:
        self._executor = tool_executor

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def execute(self, plan: ToolExecutionPlan) -> list[ToolResult]:
        """Execute all invocations in *plan* sequentially.

        Parameters
        ----------
        plan:
            The tool execution plan to run.

        Returns
        -------
        list[ToolResult]
            One result per invocation that was attempted.  If a
            required tool fails, the list may be shorter than
            ``plan.tool_count``.
        """
        if plan.is_empty():
            logger.info("SequentialToolExecutor — empty plan, nothing to execute.")
            return []

        t_start = time.monotonic()
        logger.info(
            "SequentialToolExecutor — starting %d tool(s): %s",
            plan.tool_count,
            [t.tool_name for t in plan.tools],
        )

        results: list[ToolResult] = []

        for i, invocation in enumerate(plan.tools):
            logger.info(
                "  [%d/%d] Executing %s (optional=%s)",
                i + 1,
                plan.tool_count,
                invocation.tool_name,
                invocation.optional,
            )

            result = self._executor.execute(
                tool_name=invocation.tool_name,
                arguments=invocation.arguments,
            )

            results.append(result)

            if result.success:
                logger.info(
                    "  [%d/%d] %s ✓  latency=%.0fms",
                    i + 1,
                    plan.tool_count,
                    invocation.tool_name,
                    result.execution_time_ms,
                )
            elif invocation.optional:
                logger.warning(
                    "  [%d/%d] %s ✗ (optional, continuing)  error=%s",
                    i + 1,
                    plan.tool_count,
                    invocation.tool_name,
                    result.error,
                )
            else:
                logger.error(
                    "  [%d/%d] %s ✗ (REQUIRED, aborting plan)  error=%s",
                    i + 1,
                    plan.tool_count,
                    invocation.tool_name,
                    result.error,
                )
                break

        elapsed = (time.monotonic() - t_start) * 1000
        succeeded = sum(1 for r in results if r.success)
        logger.info(
            "SequentialToolExecutor — done: %d/%d succeeded  latency=%.0fms",
            succeeded,
            len(results),
            elapsed,
        )

        return results
