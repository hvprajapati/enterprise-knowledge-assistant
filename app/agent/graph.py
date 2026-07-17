"""Build the LangGraph ``StateGraph`` with conditional routing.

The graph topology::

                    START
                      │
               supervisor_node       ← initialises multi-agent tracking
                      │
                planner_node         ← delegate to PlannerAgent
                      │          ┌──────────────────────────┐
              route_after_planner    │                          │
         ┌───────────┼───────────┐   │  FULL_PIPELINE retry     │
         ▼           ▼            ▼   │  (back to planner)       │
  tool_planner  rewrite_node  retrieve│                          │
         │           │            │   │                          │
  tool_executor     │            │   │                          │
         │           │            │   │                          │
  route_after_tools  │            │   │                          │
         │           │            │   │                          │
         └─────┬─────┘            │   │                          │
               ▼                  │   │                          │
        retrieve_node ◄───────────┘   │  delegate to RetrievalAgent
               │                  │   │                          │
       route_after_retrieve       │   │                          │
          ┌────────┴────────┐     │   │                          │
          ▼                  ▼     │   │                          │
   generate_node ◄──────────┼─────┼───┤  delegate to GenerationAgent
          │                  │     │   │                          │
          ▼                  │     │   │                          │
   reflection_node           │     │   │  delegate to ReflectionAgent
          │                  │     │   │                          │
          ▼                  │     │   │                          │
   validation_node           │     │   │  delegate to ValidationAgent
          │                  │     │   │                          │
          ▼                  │     │   │                          │
     retry_node ─────────────┴─────┴───┘                          │
          │                                                       │
   route_after_retry                                              │
          │                                                       │
     ┌────┴────┐                                                  │
     ▼         ▼                                                  │
(loop back)    END

"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from app.agent.nodes import (
    generate_node,
    planner_node,
    reflection_node,
    retrieve_node,
    retry_node,
    rewrite_node,
    supervisor_node,
    tool_executor_node,
    tool_planner_node,
    validation_node,
)
from app.agent.router import (
    route_after_planner,
    route_after_retrieve,
    route_after_retry,
    route_after_rewrite,
    route_after_tool_executor,
)
from app.agent.state import AgentState

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """Return a compiled ``StateGraph`` ready for invocation."""
    graph = StateGraph(AgentState)

    # -- nodes -----------------------------------------------------------
    graph.add_node("supervisor_node", supervisor_node)
    graph.add_node("planner_node", planner_node)
    graph.add_node("tool_planner_node", tool_planner_node)
    graph.add_node("tool_executor_node", tool_executor_node)
    graph.add_node("rewrite_node", rewrite_node)
    graph.add_node("retrieve_node", retrieve_node)
    graph.add_node("generate_node", generate_node)
    graph.add_node("reflection_node", reflection_node)
    graph.add_node("validation_node", validation_node)
    graph.add_node("retry_node", retry_node)

    # -- edges -----------------------------------------------------------
    # START -> supervisor_node (always)
    graph.set_entry_point("supervisor_node")

    # supervisor_node -> planner_node
    graph.add_edge("supervisor_node", "planner_node")

    # planner_node → conditional (tool_planner → rewrite → retrieve)
    graph.add_conditional_edges(
        "planner_node",
        route_after_planner,
        {
            "tool_planner_node": "tool_planner_node",
            "rewrite_node": "rewrite_node",
            "retrieve_node": "retrieve_node",
        },
    )

    # tool_planner_node → tool_executor_node (always)
    graph.add_edge("tool_planner_node", "tool_executor_node")

    # tool_executor_node → conditional (rewrite or retrieve)
    graph.add_conditional_edges(
        "tool_executor_node",
        route_after_tool_executor,
        {"rewrite_node": "rewrite_node", "retrieve_node": "retrieve_node"},
    )

    # rewrite_node -> conditional (always -> retrieve_node)
    graph.add_conditional_edges(
        "rewrite_node",
        route_after_rewrite,
        {"retrieve_node": "retrieve_node"},
    )

    # retrieve_node -> conditional (generate or END)
    graph.add_conditional_edges(
        "retrieve_node",
        route_after_retrieve,
        {"generate_node": "generate_node", END: END},
    )

    # generate_node -> reflection_node
    graph.add_edge("generate_node", "reflection_node")

    # reflection_node -> validation_node
    graph.add_edge("reflection_node", "validation_node")

    # validation_node -> retry_node
    graph.add_edge("validation_node", "retry_node")

    # retry_node -> conditional (loop back or END)
    graph.add_conditional_edges(
        "retry_node",
        route_after_retry,
        {
            "planner_node": "planner_node",
            "retrieve_node": "retrieve_node",
            "generate_node": "generate_node",
            END: END,
        },
    )

    return graph
