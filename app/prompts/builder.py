from app.ingestion.models import SearchResult


class PromptBuilder:
    """Builds grounded prompts for the LLM."""

    SYSTEM_PROMPT = """
You are an Enterprise Knowledge Assistant.

Answer ONLY using the provided context.

If the answer cannot be found in the context,
respond with:

"I could not find the answer in the provided documents."

Always cite the source document and page number.
""".strip()

    def build(
        self,
        query: str,
        results: list[SearchResult],
    ) -> str:

        context = []

        for result in results:
            chunk = result.chunk

            context.append(
                f"""Source: {chunk.metadata.filename}
Page: {chunk.page_number}

{chunk.text}
"""
            )

        context_text = "\n\n----------------------\n\n".join(context)

        return f"""
{self.SYSTEM_PROMPT}

Context
========

{context_text}

Question
========

{query}

Answer
========
""".strip()
