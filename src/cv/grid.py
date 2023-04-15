from typing import Any, Iterator, Iterable


def grid(**kwargs: Iterable[Any]) -> Iterator[dict[str, Any]]:
    if len(kwargs) == 0:
        yield {}
    else:
        key, values = kwargs.popitem()
        for rest in grid(**kwargs):
            for value in values:
                yield {key: value, **rest}
