from functools import cached_property
from os import path, mkdir

from bs4 import BeautifulSoup

from .util.dict import FuzzyDict, safe_update
from .util.soup import (
    fetch_soup,
    load_soup,
    parse_int,
    parse_percent,
    parse_str,
    save_soup
)


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

REGIONAL_PREFIXES = ["Galarian", "Alolan"]


def get_full_name(base_name: str, variant_name: str) -> str:
    if base_name in variant_name:
        return variant_name
    for prefix in REGIONAL_PREFIXES:
        if variant_name.startswith(prefix):
            variant_name_rest = variant_name[len(prefix):].strip()
            return f'{prefix} {base_name} {variant_name_rest}'
    return f'{base_name} {variant_name}'


def name_to_dex_path() -> dict[str, str]:
    page = fetch_soup(
        url=path.join(POKEDEX_URL, "all"),
        cache_path=path.join(BASE_CACHE_DIR, "all.html"),
    )
    cells = page.find_all("td", "cell-name")

    mapping = {}
    for cell in cells:
        name_a = cell.find("a", "ent-name")
        pokemon = path.basename(name_a["href"])

        base_name = parse_str(name_a)
        safe_update(mapping, key=base_name, value=pokemon)

        variant_small = cell.find("small", "text-muted")
        if variant_small is None:
            continue

        variant_name = parse_str(variant_small)
        safe_update(
            mapping,
            key=get_full_name(base_name, variant_name),
            value=pokemon
        )

    return FuzzyDict(mapping)


def fetch_pokemon_soup(dex_path: str) -> BeautifulSoup:
    return fetch_soup(
        url=path.join(POKEDEX_URL, dex_path),
        cache_path=path.join(POKEMON_CACHE_DIR, f"{dex_path}.html"),
    )


def fetch_variant_soup(dex_path: str, variant_name: str) -> BeautifulSoup:
    cache_path = path.join(VARIANT_CACHE_DIR, f"{variant_name}.html")
    if path.exists(cache_path):
        # print(f"Variant {variant} found in cache at {cache_path}")
        variant_soup = load_soup(cache_path)
    else:
        # print(f"Variant {variant} not in cache, fetching...")
        pokemon_soup = fetch_pokemon_soup(dex_path)

        tabs = pokemon_soup.find("div", "sv-tabs-tab-list").find_all("a")
        variant_to_id = FuzzyDict({
            parse_str(tab): tab["href"].lstrip("#")
            for tab in tabs
        })
        closest_id = variant_to_id.get(variant_name)
        variant_soup = pokemon_soup.find("div", id=closest_id)
        # print(f"Saving variant {variant} to cache at {cache_path}")
        save_soup(variant_soup, cache_path)
    return variant_soup


class Variant:
    _NAME_TO_DEX_PATH = name_to_dex_path()
    PROPERTIES = [
        "base_experience",
        "base_friendship",
        "catch_rate",
        "egg_cycles",
        "growth_rate",
        "percentage_male",
    ]

    def __init__(self, name: str, soup: BeautifulSoup):
        self.name = name
        self._soup = soup

    @classmethod
    def fetch(cls, variant_name: str) -> "Variant":
        dex_path = cls._NAME_TO_DEX_PATH.get(variant_name)
        soup = fetch_variant_soup(dex_path=dex_path, variant_name=variant_name)
        return Variant(name=variant_name, soup=soup)

    @cached_property
    def catch_rate(self) -> int:
        cell = self._soup \
            .find("th", string="Catch rate") \
            .find_next_sibling("td")
        return parse_int(cell)

    @cached_property
    def base_friendship(self) -> int:
        cell = self._soup \
            .find("a", href="/glossary#def-friendship") \
            .find_parent("th") \
            .find_next_sibling("td")
        return parse_int(cell)

    @cached_property
    def base_experience(self) -> int:
        cell = self._soup \
            .find("th", string="Base Exp.") \
            .find_next_sibling("td")
        return parse_int(cell)

    @cached_property
    def growth_rate(self) -> str:
        cell = self._soup \
            .find("th", string="Growth Rate") \
            .find_next_sibling("td")
        return parse_str(cell)

    @cached_property
    def percentage_male(self) -> float | None:
        cell = self._soup \
            .find("th", string="Gender") \
            .find_next_sibling("td")
        s = parse_str(cell)
        return None if s == "Genderless" else parse_percent(cell)

    @cached_property
    def egg_cycles(self) -> int:
        cell = self._soup \
            .find("th", string="Egg cycles") \
            .find_next_sibling("td")
        return parse_int(cell)

    def as_dict(self) -> dict[str, int | str | None]:
        return {
            attr: getattr(self, attr)
            for attr in Variant.PROPERTIES
        }
