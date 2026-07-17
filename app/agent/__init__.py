"""LangGraph-based Enterprise Knowledge Agent.

Conditional routing dynamically decides which pipeline stages to
execute based on the question and retrieval results.
"""

from app.agent.agent import EnterpriseKnowledgeAgent

__all__ = ["EnterpriseKnowledgeAgent"]
