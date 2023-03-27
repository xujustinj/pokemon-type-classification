from os import path, mkdir
from difflib import get_close_matches
from typing import Any, Callable, Generic, TypeVar

from bs4 import BeautifulSoup
import requests

POKEDEX_URL = "https://pokemondb.net/pokedex/"
ALL_URL = path.join(POKEDEX_URL, "all")

SCRIPT_PATH = path.realpath(__file__)

BASE_CACHE_DIR = path.join(path.dirname(SCRIPT_PATH), ".cache")
if not path.exists(BASE_CACHE_DIR):
    mkdir(BASE_CACHE_DIR)

POKEMON_CACHE_DIR = path.join(BASE_CACHE_DIR, "pokemon")
if not path.exists(POKEMON_CACHE_DIR):
    mkdir(POKEMON_CACHE_DIR)

VARIANT_CACHE_DIR = path.join(BASE_CACHE_DIR, "variant")
if not path.exists(VARIANT_CACHE_DIR):
    mkdir(VARIANT_CACHE_DIR)


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


def load_soup(load_path: str) -> BeautifulSoup:
    with open(load_path, "r") as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    return soup


def save_soup(soup: BeautifulSoup, save_path: str):
    with open(save_path, "w") as f:
        f.write(str(soup))


def fetch_soup(url: str, cache_path: str | None = None) -> BeautifulSoup:
    if path.exists(cache_path):
        print(f"Resource {url} found in cache at {cache_path}")
        soup = load_soup(cache_path)
    else:
        print(f"Resource {url} not in cache, fetching...")
        req = requests.get(url)
        soup = BeautifulSoup(req.text, 'html.parser')
        print(f"Saving resource {url} to cache at {cache_path}")
        save_soup(soup, cache_path)
    return soup


REGIONAL_PREFIXES = ["Galarian", "Alolan"]


def get_full_name(base_name: str, variant_name: str) -> str:
    if base_name in variant_name:
        return variant_name
    for prefix in REGIONAL_PREFIXES:
        if variant_name.startswith(prefix):
            variant_name_rest = variant_name[len(prefix):].strip()
            return f'{prefix} {base_name} {variant_name_rest}'
    return f'{base_name} {variant_name}'


def name_to_pokemon() -> dict[str, str]:
    page = fetch_soup(
        url=path.join(POKEDEX_URL, "all"),
        cache_path=path.join(BASE_CACHE_DIR, "all.html"),
    )
    cells = page.find_all("td", "cell-name")

    mapping = {}
    for cell in cells:
        name_a = cell.find("a", "ent-name")
        pokemon = path.basename(name_a["href"])

        base_name = name_a.text.strip()
        safe_update(mapping, key=base_name, value=pokemon)

        variant_small = cell.find("small", "text-muted")
        if variant_small is None:
            continue

        variant_name = variant_small.text.strip()
        safe_update(
            mapping,
            key=get_full_name(base_name, variant_name),
            value=pokemon
        )

    return FuzzyDict(mapping)


NAME_TO_POKEMON = name_to_pokemon()


def get_pokemon(pokemon: str) -> BeautifulSoup:
    return fetch_soup(
        url=path.join(POKEDEX_URL, pokemon),
        cache_path=path.join(POKEMON_CACHE_DIR, f"{pokemon}.html"),
    )


def get_variant(variant: str) -> BeautifulSoup:
    cache_path = path.join(VARIANT_CACHE_DIR, f"{variant}.html")
    if path.exists(cache_path):
        print(f"Variant {variant} found in cache at {cache_path}")
        soup = load_soup(cache_path)
    else:
        print(f"Variant {variant} not in cache, fetching...")
        pokemon = get_pokemon(NAME_TO_POKEMON.get(variant))

        tabs = pokemon.find("div", "sv-tabs-tab-list").find_all("a")
        variant_to_id = FuzzyDict({
            tab.text.strip(): tab["href"].lstrip("#")
            for tab in tabs
        })
        closest_id = variant_to_id.get(variant)
        soup = pokemon.find("div", id=closest_id)
        print(f"Saving variant {variant} to cache at {cache_path}")
        save_soup(soup, cache_path)
    return soup


def get_catch_rate(variant: BeautifulSoup) -> int:
    td = variant.find("th", string="Catch rate").find_next_sibling("td")
    return int(td.text.strip().split()[0])


def get_base_friendship(variant: BeautifulSoup) -> int:
    td = variant \
        .find("a", href="/glossary#def-friendship") \
        .find_parent("th") \
        .find_next_sibling("td")
    return int(td.text.strip().split()[0])


def get_base_experience(variant: BeautifulSoup) -> int:
    td = variant.find("th", string="Base Exp.").find_next_sibling("td")
    return int(td.text.strip())


for variant_name in [
    "Galarian Meowth",
    "Orbeetle",
    "Toxtricity Low Key Form",
    "Wormadam Plant Cloak",
    "Galarian Darmanitan Standard Mode",
    "Zacian Hero of Many Battles",
]:
    print(variant_name)
    variant = get_variant(variant_name)
    print(get_catch_rate(variant))
    print(get_base_friendship(variant))
    print(get_base_experience(variant))
