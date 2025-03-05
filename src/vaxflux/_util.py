__all__: tuple[str, ...] = ()


import re
from typing import Callable


_CLEAN_TEXT_REGEX = re.compile(r"[^a-zA-Z0-9]")


def _clean_text(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", " ", text).title().replace(" ", "")


def _clean_name(
    *args: str | None, joiner: str = "", transform: Callable[[str], str] = lambda x: x
) -> str:
    return joiner.join(
        map(
            lambda x: transform(_CLEAN_TEXT_REGEX.sub(" ", x)).replace(" ", joiner),
            filter(None, args),
        )
    )


def _pm_name(*args: str | None) -> str:
    return _clean_name(*args, transform=lambda x: x.title())


def _coord_name(*args: str | None) -> str:
    return _clean_name(*args, joiner="_", transform=lambda x: x.lower())
