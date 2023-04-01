from functools import cached_property
from math import sqrt
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
from .util.sprite import Sprite


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

SPRITE_CACHE_DIR = path.join(BASE_CACHE_DIR, "sprite")
if not path.exists(SPRITE_CACHE_DIR):
    mkdir(SPRITE_CACHE_DIR)

REGIONAL_PREFIXES = ["Galarian", "Alolan"]


def get_full_name(base_name: str, variant_name: str) -> str:
    if base_name in variant_name:
        return variant_name
    for prefix in REGIONAL_PREFIXES:
        if variant_name.startswith(prefix):
            variant_name_rest = variant_name[len(prefix):].strip()
            return f'{prefix} {base_name} {variant_name_rest}'
    return f'{base_name} {variant_name}'


def name_to_dex_path() -> FuzzyDict[str]:
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
            value=pokemon,
        )

    return FuzzyDict(mapping)


def name_to_sprite_url() -> FuzzyDict[str]:
    page = fetch_soup(
        url=path.join(POKEDEX_URL, "all"),
        cache_path=path.join(BASE_CACHE_DIR, "all.html"),
    )
    cells = page.find_all("td", "cell-name")

    mapping = {}
    for cell in cells:
        name_a = cell.find("a", "ent-name")
        base_name = parse_str(name_a)

        row = cell.find_parent("tr")
        sprite = row.find("img", "img-fixed icon-pkmn")
        sprite_url = sprite["src"]

        variant_small = cell.find("small", "text-muted")

        # hack to get around Burmy mapping to Buneary
        if base_name == "Burmy" and "Burmy" not in mapping:
            safe_update(mapping, key=base_name, value=sprite_url)

        if variant_small is None:
            safe_update(mapping, key=base_name, value=sprite_url)
        else:
            variant_name = parse_str(variant_small)
            safe_update(
                mapping,
                key=get_full_name(base_name, variant_name),
                value=sprite_url,
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


def fetch_variant_sprite(url: str, variant_name: str) -> Sprite:
    cache_path = path.join(SPRITE_CACHE_DIR, f"{variant_name}.png")
    return Sprite.fetch(url, cache_path)


class Variant:
    _NAME_TO_DEX_PATH = name_to_dex_path()
    _NAME_TO_SPRITE_URL = name_to_sprite_url()
    PROPERTIES = [
        "base_experience",
        "base_friendship",
        "catch_rate",
        "egg_cycles",
        "growth_rate",
        "percentage_male",
        "sprite_size",
        "sprite_perimeter",
        "sprite_perimeter_to_size_ratio",
        "sprite_red_mean",
        "sprite_green_mean",
        "sprite_blue_mean",
        "sprite_brightness_mean",
        "sprite_red_sd",
        "sprite_green_sd",
        "sprite_blue_sd",
        "sprite_brightness_sd",
        "sprite_overflow_vertical",
        "sprite_overflow_horizontal",
    ]

    def __init__(self, name: str, soup: BeautifulSoup, sprite: Sprite):
        self.name = name
        self._soup = soup
        self._sprite = sprite

    @classmethod
    def fetch(cls, variant_name: str) -> "Variant":
        dex_path = cls._NAME_TO_DEX_PATH.get(variant_name)
        sprite_url = cls._NAME_TO_SPRITE_URL.get(variant_name)
        soup = fetch_variant_soup(
            dex_path=dex_path,
            variant_name=variant_name,
        )
        sprite = fetch_variant_sprite(
            url=sprite_url,
            variant_name=variant_name,
        )
        return Variant(name=variant_name, soup=soup, sprite=sprite)

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

    @cached_property
    def sprite_size(self) -> float:
        return (self._sprite.alpha != 0).sum()

    @cached_property
    def sprite_perimeter(self) -> float:
        return (self._sprite.perimeter).sum()

    @cached_property
    def sprite_perimeter_to_size_ratio(self) -> float:
        return self.sprite_perimeter / self.sprite_size

    @cached_property
    def sprite_red_mean(self) -> float:
        return (self._sprite.alpha * self._sprite.red).sum() / self.sprite_size

    @cached_property
    def sprite_green_mean(self) -> float:
        return (self._sprite.alpha * self._sprite.green).sum() / self.sprite_size

    @cached_property
    def sprite_blue_mean(self) -> float:
        return (self._sprite.alpha * self._sprite.blue).sum() / self.sprite_size

    @cached_property
    def sprite_brightness_mean(self) -> float:
        return (self._sprite.alpha * self._sprite.brightness).sum() / self.sprite_size

    @cached_property
    def sprite_red_sd(self) -> float:
        return sqrt(
            (self._sprite.alpha * (self._sprite.red - self.sprite_red_mean)**2).sum()
            / self.sprite_size
        )

    @cached_property
    def sprite_green_sd(self) -> float:
        return sqrt(
            (self._sprite.alpha * (self._sprite.green - self.sprite_green_mean)**2).sum()
            / self.sprite_size
        )

    @cached_property
    def sprite_blue_sd(self) -> float:
        return sqrt(
            (self._sprite.alpha * (self._sprite.blue - self.sprite_blue_mean)**2).sum()
            / self.sprite_size
        )

    @cached_property
    def sprite_brightness_sd(self) -> float:
        return sqrt(
            (self._sprite.alpha * (self._sprite.brightness -
             self.sprite_brightness_mean)**2).sum()
            / self.sprite_size
        )

    @cached_property
    def sprite_overflow_vertical(self) -> float:
        return self._sprite.alpha[[0, -1], :].mean()

    @cached_property
    def sprite_overflow_horizontal(self) -> float:
        return self._sprite.alpha[:, [0, -1]].mean()

    def as_dict(self) -> dict[str, int | str | None]:
        return {
            attr: getattr(self, attr)
            for attr in Variant.PROPERTIES
        }
