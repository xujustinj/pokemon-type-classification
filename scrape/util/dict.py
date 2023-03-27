from typing import Generic, TypeVar
from difflib import get_close_matches


K = TypeVar('K')
V = TypeVar('V')


class FuzzyDict(Generic[V]):
    def __init__(self, items: dict[str, V]):
        assert len(items) > 0

        self._items = items
        self._n_items = len(self._items)
        self._keys = list(self._items.keys())

    def get(self, key: str) -> V:
        if (key in self._items):
            return self._items[key]
        closest, = get_close_matches(
            word=key,
            possibilities=self._keys,
            n=1,
            cutoff=0.0,
        )
        print(f"'{key}' not found, falling back to '{closest}'")
        return self._items[closest]

    def __repr__(self) -> str:
        return repr(self._items)


def safe_update(d: dict[K, V], key: K, value: V) -> None:
    if key in d:
        assert d[key] == value, f'Key mismatch: expected {value} at {key} but got {d[key]}'
    else:
        d[key] = value
