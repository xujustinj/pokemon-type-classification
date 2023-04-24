from os import path, mkdir

from bs4 import BeautifulSoup
from typing import Iterator

from .util.soup import fetch_soup, load_soup, parse_str, save_soup
from .util.sprite import Sprite
from .variant import Variant


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


def safe_name(pokemon_name: str, variant_name: str | None = None) -> str:
    """Get a normalized name for the Pokemon.

    Spaces, punctuation, and special symbols are removed.

    Args:
        pokemon_name (str): The Pokemon's name.
        variant_name (str): The Pokemon's variant's name.

    Return:
        safe_name (str): The normalized name.
    """
    name = (
        pokemon_name if variant_name is None
        else f'{pokemon_name}+{variant_name}'
    )
    name = name.replace(" ", "_")  # flatten all spaces
    name = name.replace(".", "").replace("'", "")  # remove punctuation
    name = name.replace("♀", "-f").replace("♂", "-m")  # unroll genders
    name = name.lower()  # lowercase
    return name


def fetch_pokemon_soup(dex_path: str) -> BeautifulSoup:
    """Fetch the Pokemon's data.

    Args:
        dex_path (str): The path to the Pokemon's data in PokemonDB.

    Return:
        soup (BeautifulSoup): The HTML web data of the Pokemon.
    """
    url = path.join(POKEDEX_URL, dex_path)
    cache_path = path.join(POKEMON_CACHE_DIR, f"{dex_path}.html")
    return fetch_soup(url=url, cache_path=cache_path)


def fetch_variant_sprite(
    url: str,
    pokemon_name: str,
    variant_name: str | None,
) -> Sprite:
    """Fetch the Pokemon's variant's sprite.

    Args:
        url (str): The website URL.
        pokemon_name (str): The Pokemon's name
        variant_name (str): The Pokemon's variant's name

    Return:
        sprite (Sprite): The Pokemon's variant's small picture representation.
    """
    cache_path = path.join(
        SPRITE_CACHE_DIR,
        f"{safe_name(pokemon_name, variant_name)}.png"
    )
    return Sprite.fetch(url, cache_path)


def get_variant_soup(
    pokemon_soup: BeautifulSoup,
    pokemon_name: str,
    variant_name: str,
) -> BeautifulSoup:
    """Extract a variant from a Pokemon's web page.

    Args:
        pokemon_soup (BeautifulSoup): The Pokemon's data.
        pokemon_name (str): The Pokemon's name.
        variant_name (str): The Pokemon's variant's name.

    Return:
        soup (BeautifulSoup): A web element containing the Pokemon's variant's
            data.
    """
    cache_path = path.join(
        VARIANT_CACHE_DIR,
        f"{safe_name(pokemon_name, variant_name)}.html"
    )
    if path.exists(cache_path):
        variant_soup = load_soup(cache_path)
    else:
        tabs = pokemon_soup.find("div", "sv-tabs-tab-list").find_all("a")
        variant_to_id = {
            parse_str(tab): tab["href"].lstrip("#")
            for tab in tabs
        }
        closest_id = variant_to_id[variant_name]
        variant_soup = pokemon_soup.find("div", id=closest_id)
        save_soup(variant_soup, cache_path)
    return variant_soup


def all_variants() -> Iterator[Variant]:
    """Return all Pokemons' variants' data.

    Return:
        data (Iterator[Variant]): All Pokemons' variant's data.
    """
    page = fetch_soup(
        url=path.join(POKEDEX_URL, "all"),
        cache_path=path.join(BASE_CACHE_DIR, "all.html"),
    )
    rows = page.find_all("tr")

    for i, row in enumerate(rows[1:]):
        name_td = row.find("td", "cell-name")
        pokemon_name_a = name_td.find("a", "ent-name")
        pokemon_name = parse_str(pokemon_name_a)
        variant_name_small = name_td.find("small", "text-muted")
        if variant_name_small is None:
            variant_name = None
        else:
            variant_name = parse_str(variant_name_small)

        dex_path = path.basename(pokemon_name_a["href"])
        pokemon_soup = fetch_pokemon_soup(dex_path)
        variant_soup = get_variant_soup(
            pokemon_soup=pokemon_soup,
            pokemon_name=pokemon_name,
            variant_name=pokemon_name if variant_name is None else variant_name,
        )

        sprite_img = row.find("img", "img-fixed icon-pkmn")
        sprite_url: str = sprite_img["src"]
        variant_sprite = fetch_variant_sprite(
            url=sprite_url,
            pokemon_name=pokemon_name,
            variant_name=variant_name,
        )

        if variant_name is None:
            print(f"{i+1:04d}\t{pokemon_name}")
        else:
            print(f"{i+1:04d}\t{pokemon_name}: {variant_name}")

        yield Variant(
            pokemon_name=pokemon_name,
            variant_name=variant_name,
            soup=variant_soup,
            sprite=variant_sprite,
        )
