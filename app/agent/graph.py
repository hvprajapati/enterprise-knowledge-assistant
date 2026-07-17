"""Build the LangGraph ``StateGraph`` with conditional routing.

The graph topology::

                    START
                      │
                planner_node          ← analyses question, creates plan
                      │          ┌──────────────────────────┐
              route_after_start   │                          │
                 ┌────────┴───────┤  FULL_PIPELINE retry     │
                 ▼                 │  (back to planner)       │
          rewrite_node             │                          │
                 │                 │                          │
          route_after_rewrite      │                          │
                 │                 │                          │
                 └──────┬──────────┤                          │
                        ▼          │                          │
                 retrieve_node ◄───┤ RETRIEVE_AGAIN retry     │
                        │          │                          │
                route_after_retrieve                          │
                   ┌────────┴────────┐                        │
                   ▼                  ▼                        │
            generate_node ◄──────────┼── GENERATE_AGAIN retry │
                   │                 │                        │
                   ▼                 │                        │
            reflection_node          │                        │
                   │                 │                        │
                   ▼                 │                        │
            validation_node          │                        │
                   │                 │                        │
                   ▼                 │                        │
              retry_node ────────────┴────────────────────────┘
                   │
            route_after_retry
                   │
              ┌────┴────┐
              ▼         ▼
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
    validation_node,
)
from app.agent.router import (
    route_after_retrieve,
    route_after_retry,
    route_after_rewrite,
    route_after_start,
)
from app.agent.state import AgentState

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """Return a compiled ``StateGraph`` ready for invocation."""
    graph = StateGraph(AgentState)

    # -- nodes -----------------------------------------------------------
    graph.add_node("planner_node", planner_node)
    graph.add_node("rewrite_node", rewrite_node)
    graph.add_node("retrieve_node", retrieve_node)
    graph.add_node("generate_node", generate_node)
    graph.add_node("reflection_node", reflection_node)
    graph.add_node("validation_node", validation_node)
    graph.add_node("retry_node", retry_node)

    # -- edges -----------------------------------------------------------
    # START -> planner_node (always)
    graph.set_entry_point("planner_node")

    # planner_node -> conditional (rewrite or skip to retrieve)
    graph.add_conditional_edges(
        "planner_node",
        route_after_start,
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
