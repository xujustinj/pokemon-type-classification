from functools import cached_property
from math import sqrt

from bs4 import BeautifulSoup
from .util.sprite import Sprite

from .util.soup import parse_int, parse_percent, parse_str


class Variant:
    PROPERTIES = [
        "abilities_number",
        "total_points",
        "hp",
        "attack",
        "defense",
        "sp_attack",
        "sp_defense",
        "speed",
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

    def __init__(
        self,
        pokemon_name: str,
        variant_name: str,
        soup: BeautifulSoup,
        sprite: Sprite
    ):
        self.pokemon_name = pokemon_name
        self.variant_name = variant_name
        self._soup = soup
        self._sprite = sprite

    def _get_cell(self, label: str) -> BeautifulSoup:
        return self._soup.find("th", string=label).find_next_sibling("td")

    @cached_property
    def abilities_number(self) -> int:
        return len(self._get_cell("Abilities").find_all("br"))

    @cached_property
    def hp(self) -> int:
        return parse_int(self._get_cell("HP"))

    @cached_property
    def attack(self) -> int:
        return parse_int(self._get_cell("Attack"))

    @cached_property
    def defense(self) -> int:
        return parse_int(self._get_cell("Defense"))

    @cached_property
    def sp_attack(self) -> int:
        return parse_int(self._get_cell("Sp. Atk"))

    @cached_property
    def sp_defense(self) -> int:
        return parse_int(self._get_cell("Sp. Def"))

    @cached_property
    def speed(self) -> int:
        return parse_int(self._get_cell("Speed"))

    @cached_property
    def total_points(self) -> int:
        total = parse_int(self._get_cell("Total"))
        assert total == (
            self.hp
            + self.attack
            + self.defense
            + self.sp_attack
            + self.sp_defense
            + self.speed
        )
        return total

    @cached_property
    def catch_rate(self) -> int:
        return parse_int(self._get_cell("Catch rate"))

    @cached_property
    def base_friendship(self) -> int:
        cell = self._soup \
            .find("a", href="/glossary#def-friendship") \
            .find_parent("th") \
            .find_next_sibling("td")
        return parse_int(cell)

    @cached_property
    def base_experience(self) -> int:
        return parse_int(self._get_cell("Base Exp."))

    @cached_property
    def growth_rate(self) -> str:
        return parse_str(self._get_cell("Growth Rate"))

    @cached_property
    def percentage_male(self) -> float | None:
        cell = self._get_cell("Gender")
        s = parse_str(cell)
        return None if s == "Genderless" else parse_percent(cell)

    @cached_property
    def egg_cycles(self) -> int:
        return parse_int(self._get_cell("Egg cycles"))

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
