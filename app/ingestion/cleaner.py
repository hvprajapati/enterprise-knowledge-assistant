import re
import unicodedata


class TextCleaner:
    """Cleans extracted document text before chunking."""

    @staticmethod
    def clean(text: str) -> str:
        # Normalize Unicode characters
        text = unicodedata.normalize("NFKC", text)

        # Remove control characters (except newlines and tabs)
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

        # Replace multiple spaces/tabs with a single space
        text = re.sub(r"[ \t]+", " ", text)

        # Collapse 3+ newlines into exactly 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()