from typing import Any, Iterator, Iterable

def grid(**kwargs: Iterable[Any]) -> Iterator[dict[str, Any]]:
    """Generate all combinations of hyperparameters.

    Generate all combinations of hyperparameters.

    Args:
        Iterable[Any]: hyperparameter lists

    Returns:
        Iterator[dict[str, Any]]: sets of hyperparameters

    """
    if len(kwargs) == 0:
        yield {}
    else:
        key, values = kwargs.popitem()
        for rest in grid(**kwargs):
            for value in values:
                yield {key: value, **rest}
