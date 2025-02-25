__all__: tuple[str, ...] = ()


import re


def _clean_text(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", " ", text).title().replace(" ", "")


def _pm_name(*args: str | None) -> str:
    return "".join(map(_clean_text, filter(None, args)))
