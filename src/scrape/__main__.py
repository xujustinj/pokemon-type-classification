from itertools import islice
from pprint import pprint

from . import all_variants

for variant in islice(all_variants(), 5):
    pprint(variant.as_dict())
