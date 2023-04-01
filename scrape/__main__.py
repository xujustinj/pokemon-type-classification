from pprint import pprint

from . import Variant

for variant_name in [
    "Galarian Meowth",
    "Orbeetle",
    "Toxtricity Low Key Form",
    "Wormadam Plant Cloak",
    "Galarian Darmanitan Standard Mode",
    "Zacian Hero of Many Battles",
    "Eternatus Eternamax",
]:
    print(variant_name)
    variant = Variant.fetch(variant_name)
    pprint(variant.as_dict())
