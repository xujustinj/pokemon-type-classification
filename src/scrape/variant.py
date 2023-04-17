from functools import cached_property
from math import sqrt

from bs4 import BeautifulSoup
from .util.sprite import Sprite

from .util.soup import parse_float, parse_int, parse_percent, parse_str


REGIONAL_PREFIXES = ["Galarian", "Alolan"]


class Variant:
    PROPERTIES = [
        "type_number",
        "type_1",
        "type_2",
        "height_m",
        "weight_kg",
        "abilities_number",
        "total_points",
        "hp",
        "attack",
        "defense",
        "sp_attack",
        "sp_defense",
        "speed",
        "catch_rate",
        "base_friendship",
        "base_experience",
        "maximum_experience",
        "egg_type_number",
        "has_gender",
        "proportion_male",
        "egg_cycles",
        "damage_from_normal",
        "damage_from_fire",
        "damage_from_water",
        "damage_from_electric",
        "damage_from_grass",
        "damage_from_ice",
        "damage_from_fighting",
        "damage_from_poison",
        "damage_from_ground",
        "damage_from_flying",
        "damage_from_psychic",
        "damage_from_bug",
        "damage_from_rock",
        "damage_from_ghost",
        "damage_from_dragon",
        "damage_from_dark",
        "damage_from_steel",
        "damage_from_fairy",
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

    _POSITIONS = {
        "Nor": (0, 0)
    }

    def __init__(
        self,
        pokemon_name: str,
        variant_name: str | None,
        soup: BeautifulSoup,
        sprite: Sprite
    ):
        self.pokemon_name = pokemon_name
        self.variant_name = variant_name
        self._soup = soup
        self._sprite = sprite

    @cached_property
    def full_name(self) -> str:
        if self.variant_name is None:
            return self.pokemon_name
        if self.pokemon_name in self.variant_name:
            return self.variant_name
        for prefix in REGIONAL_PREFIXES:
            if self.variant_name.startswith(prefix):
                variant_name_rest = self.variant_name[len(prefix):].strip()
                return f'{prefix} {self.pokemon_name} {variant_name_rest}'
        return f'{self.pokemon_name} {self.variant_name}'

    def _get_cell(self, label: str) -> BeautifulSoup:
        return self._soup.find("th", string=label).find_next_sibling("td")

    @cached_property
    def pokedex_number(self) -> int:
        return parse_int(self._get_cell("National №"))

    @cached_property
    def type_number(self) -> int:
        return len(self._get_cell("Type").find_all("a"))

    @cached_property
    def type_1(self) -> str:
        return parse_str(self._get_cell("Type").find_all("a")[0])

    @cached_property
    def type_2(self) -> str:
        if self.type_number >= 2:
            return parse_str(self._get_cell("Type").find_all("a")[1])
        return "None"

    @ cached_property
    def height_m(self) -> float:
        return parse_float(self._get_cell("Height"))

    @ cached_property
    def weight_kg(self) -> float | None:
        cell = self._get_cell("Weight")
        if parse_str(cell) == "—":
            return None
        return parse_float(cell)

    @ cached_property
    def abilities_number(self) -> int:
        return len(self._get_cell("Abilities").find_all("br"))

    @ cached_property
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

    @ cached_property
    def hp(self) -> int:
        return parse_int(self._get_cell("HP"))

    @ cached_property
    def attack(self) -> int:
        return parse_int(self._get_cell("Attack"))

    @ cached_property
    def defense(self) -> int:
        return parse_int(self._get_cell("Defense"))

    @ cached_property
    def sp_attack(self) -> int:
        return parse_int(self._get_cell("Sp. Atk"))

    @ cached_property
    def sp_defense(self) -> int:
        return parse_int(self._get_cell("Sp. Def"))

    @ cached_property
    def speed(self) -> int:
        return parse_int(self._get_cell("Speed"))

    @ cached_property
    def catch_rate(self) -> int | None:
        cell = self._get_cell("Catch rate")
        if parse_str(cell) == "—":
            return None
        return parse_int(self._get_cell("Catch rate"))

    @ cached_property
    def base_friendship(self) -> int:
        cell = self._soup \
            .find("a", href="/glossary#def-friendship") \
            .find_parent("th") \
            .find_next_sibling("td")
        return parse_int(cell)

    @ cached_property
    def base_experience(self) -> int:
        return parse_int(self._get_cell("Base Exp."))

    @ cached_property
    def _growth_rate(self) -> str:
        return parse_str(self._get_cell("Growth Rate"))

    _MAX_EXP = {
        "Erratic": 600_000,
        "Fast": 800_000,
        "Medium Fast": 1_000_000,
        "Medium Slow": 1_059_860,
        "Slow": 1_250_000,
        "Fluctuating": 1_640_000,
    }

    @ cached_property
    def maximum_experience(self) -> int:
        return self._MAX_EXP[self._growth_rate]

    @ cached_property
    def egg_type_number(self) -> bool:
        return len(self._get_cell("Egg Groups").find_all("a"))

    @ cached_property
    def has_gender(self) -> bool:
        cell = self._get_cell("Gender")
        s = parse_str(cell)
        return s not in ("Genderless", "—")

    @ cached_property
    def proportion_male(self) -> float:
        cell = self._get_cell("Gender")
        s = parse_str(cell)
        return parse_percent(cell) / 100. if self.has_gender else 0.5

    @ cached_property
    def egg_cycles(self) -> int:
        return parse_int(self._get_cell("Egg cycles"))

    def _get_damage_from(self, r: int, c: int) -> float:
        section = self._soup \
            .find("h2", string="Type defenses") \
            .find_parent("div")
        cell = section \
            .find_all("table")[r] \
            .find_all("tr")[-1] \
            .find_all("td")[c]
        s = parse_str(cell)
        return {
            "4": 4.,
            "3": 3.,
            "2": 2.,
            "1½": 1.5,
            "1.25": 1.25,
            "": 1.,
            "½": 1./2.,
            "¼": 1./4.,
            "⅛": 1./8.,
            "0": 0.,
        }[s]

    @cached_property
    def damage_from_normal(self) -> int:
        return self._get_damage_from(r=0, c=0)

    @cached_property
    def damage_from_fire(self) -> int:
        return self._get_damage_from(r=0, c=1)

    @cached_property
    def damage_from_water(self) -> int:
        return self._get_damage_from(r=0, c=2)

    @cached_property
    def damage_from_electric(self) -> int:
        return self._get_damage_from(r=0, c=3)

    @cached_property
    def damage_from_grass(self) -> int:
        return self._get_damage_from(r=0, c=4)

    @cached_property
    def damage_from_ice(self) -> int:
        return self._get_damage_from(r=0, c=5)

    @cached_property
    def damage_from_fighting(self) -> int:
        return self._get_damage_from(r=0, c=6)

    @cached_property
    def damage_from_poison(self) -> int:
        return self._get_damage_from(r=0, c=7)

    @cached_property
    def damage_from_ground(self) -> int:
        return self._get_damage_from(r=0, c=8)

    @cached_property
    def damage_from_flying(self) -> int:
        return self._get_damage_from(r=1, c=0)

    @cached_property
    def damage_from_psychic(self) -> int:
        return self._get_damage_from(r=1, c=1)

    @cached_property
    def damage_from_bug(self) -> int:
        return self._get_damage_from(r=1, c=2)

    @cached_property
    def damage_from_rock(self) -> int:
        return self._get_damage_from(r=1, c=3)

    @cached_property
    def damage_from_ghost(self) -> int:
        return self._get_damage_from(r=1, c=4)

    @cached_property
    def damage_from_dragon(self) -> int:
        return self._get_damage_from(r=1, c=5)

    @cached_property
    def damage_from_dark(self) -> int:
        return self._get_damage_from(r=1, c=6)

    @cached_property
    def damage_from_steel(self) -> int:
        return self._get_damage_from(r=1, c=7)

    @cached_property
    def damage_from_fairy(self) -> int:
        return self._get_damage_from(r=1, c=8)

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