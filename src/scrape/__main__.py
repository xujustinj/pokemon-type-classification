from itertools import islice
from pprint import pprint

from . import all_variants

for variant in islice(all_variants(), 5):
    pprint(variant.pokemon_name)
    pprint(variant.variant_name)
    pprint(variant.pokedex_number)
    pprint(variant.as_dict())
