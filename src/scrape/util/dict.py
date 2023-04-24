from typing import Generic, TypeVar
from difflib import get_close_matches


V = TypeVar('V')


class FuzzyDict(Generic[V]):
    """A string-keyed dictionary of values V with fuzzy matching for keys.

    Args:
        items (dict[str, V]): A normal string-keyed dictionary with values V.
    """
    def __init__(self, items: dict[str, V]):
        assert len(items) > 0

        self._items = items
        self._n_items = len(self._items)
        self._keys = list(self._items.keys())

    def get(self, key: str) -> V:
        """Retrieve a value stored at or near the given key.

        Args:
            key (str): The key.

        Returns:
            value (V): The value.
        """
        if (key in self._items):
            return self._items[key]
        # hack to get around Darmanitan mapping to Galarian
        elif key.startswith("Darmanitan "):
            return self.get(key[len("Darmanitan "):])
        closest, = get_close_matches(
            word=key,
            possibilities=self._keys,
            n=1,
            cutoff=0.0,
        )
        print(f"'{key}' not found, falling back to '{closest}'")
        return self._items[closest]
