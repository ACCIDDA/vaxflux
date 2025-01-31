__all__: tuple[str] = ()


import re


def _clean_text(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", " ", text).title().replace(" ", "")
