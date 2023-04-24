## Scrape Data

It turns out that most columns in the original dataset are inconsistent with [pokemondb.net](https://pokemondb.net/pokedex). We will rescrape the data with our own more accurate scraper.

We will also scrape all Pokémon's sprites and compute 13 numerical features.

This notebook uses our custom scraper to retrieve missing values from the website.


### Imports

```python
from collections import defaultdict

import pandas as pd

from scrape import Variant, all_variants
from scrape.util.dict import FuzzyDict
```

### Original Data

Our scraper handles everything but generation and status (which are correct). We will use the original dataset to provide those values.

```python
IN_PATH = "../data/raw.csv"
raw = pd.read_csv(IN_PATH)

info_by_pokedex = defaultdict(dict)

raw = raw[["pokedex_number", "name", "generation", "status"]]
for i, name, generation, status in raw.itertuples(index=False):
    info_by_pokedex[i][name] = (generation, status)

info_by_pokedex = { i: FuzzyDict(d) for i, d in info_by_pokedex.items() }
```

### Scraping

```python
variant_dicts = []
for variant in all_variants():
    # skip new Hisuian variants because they're Generation 9
    if variant.variant_name is not None and "Hisuian" in variant.variant_name:
        continue
    if variant.pokedex_number in info_by_pokedex:
        (generation, status) = info_by_pokedex[variant.pokedex_number].get(variant.full_name)
        variant_dict = variant.as_dict()
        variant_dict["generation"] = generation
        variant_dict["status"] = status
        variant_dicts.append(variant_dict)
```

```python
data = pd.DataFrame.from_dict(variant_dicts)
data = data[["generation", "status", *Variant.PROPERTIES]]
data.head()
```

### Missing Values

There are only two `None` values in the entire data set, and they both belong to [Eternatus Eternamax](https://bulbapedia.bulbagarden.net/wiki/Eternatus_(Pok%C3%A9mon)).

Eternatus Eternamax is 5 times the height of its normal form, so if we assume that Eternatus Eternamax is about 5 times as large in every dimension, its volume (thus its mass, assuming the same density) is 125 times as much (placing it at 118,750 kg).

Moreover, Eternatus Eternamax cannot be caught, so we can assume its catch rate is zero.

```python
data["weight_kg"] = data["weight_kg"].fillna(118750.0)
data["catch_rate"] = data["catch_rate"].fillna(0.)
```

### Data Types

Convert all columns to either `float` (continuous), `bool` (binary), or `str` (categorical). In practice, this just means converting all `int` columns to `float`.

```python
int_columns = data.columns[data.dtypes == "int64"]
data[int_columns] = data[int_columns].astype(float)
data = data.convert_dtypes(
    convert_string=True,
    convert_integer=False,
    convert_boolean=True,
    convert_floating=True,
)
```

### Save Results

```python
OUT_PATH = "../data/scraped.csv"
data.to_csv(OUT_PATH, index=False)
data.head()
```
