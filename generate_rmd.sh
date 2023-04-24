jupytext --to markdown --update-metadata '{"jupytext": {"notebook_metadata_filter":"-all"}}' src/*.ipynb
mv src/*.md rmd/converted_ipynb

printf '## `model/__init__.py`\n\n```{py}\n%s\n```\n' "`cat src/model/__init__.py`" > rmd/converted_py/model/__init__.md
printf '## `model/duplication.py`\n\n```{py}\n%s\n```\n' "`cat src/model/duplication.py`" > rmd/converted_py/model/duplication.md
printf '## `model/model.py`\n\n```{py}\n%s\n```\n' "`cat src/model/model.py`" > rmd/converted_py/model/model.md
printf '## `model/sk_model.py`\n\n```{py}\n%s\n```\n' "`cat src/model/sk_model.py`" > rmd/converted_py/model/sk_model.md
printf '## `util/dict.py`\n\n```{py}\n%s\n```\n' "`cat src/scrape/util/dict.py`" > rmd/converted_py/scrape/util/dict.md
printf '## `util/soup.py`\n\n```{py}\n%s\n```\n' "`cat src/scrape/util/soup.py`" > rmd/converted_py/scrape/util/soup.md
printf '## `util/sprite.py`\n\n```{py}\n%s\n```\n' "`cat src/scrape/util/sprite.py`" > rmd/converted_py/scrape/util/sprite.md
printf '## `scrape/__init__.py`\n\n```{py}\n%s\n```\n' "`cat src/scrape/__init__.py`" > rmd/converted_py/scrape/__init__.md
printf '## `scrape/scrape.py`\n\n```{py}\n%s\n```\n' "`cat src/scrape/scrape.py`" > rmd/converted_py/scrape/scrape.md
printf '## `scrape/variant.py`\n\n```{py}\n%s\n```\n' "`cat src/scrape/variant.py`" > rmd/converted_py/scrape/variant.md
printf '## `tune/__init__.py`\n\n```{py}\n%s\n```\n' "`cat src/tune/__init__.py`" > rmd/converted_py/tune/__init__.md
printf '## `tune/dimension.py`\n\n```{py}\n%s\n```\n' "`cat src/tune/dimension.py`" > rmd/converted_py/tune/dimension.md
printf '## `tune/outer_cv.py`\n\n```{py}\n%s\n```\n' "`cat src/tune/outer_cv.py`" > rmd/converted_py/tune/outer_cv.md
printf '## `tune/sk_bayes.py`\n\n```{py}\n%s\n```\n' "`cat src/tune/sk_bayes.py`" > rmd/converted_py/tune/sk_bayes.md
printf '## `tune/tuner.py`\n\n```{py}\n%s\n```\n' "`cat src/tune/tuner.py`" > rmd/converted_py/tune/tuner.md
printf '## `util/__init__.py`\n\n```{py}\n%s\n```\n' "`cat src/util/__init__.py`" > rmd/converted_py/util/__init__.md
printf '## `util/accuracy.py`\n\n```{py}\n%s\n```\n' "`cat src/util/accuracy.py`" > rmd/converted_py/util/accuracy.md
printf '## `util/confusion.py`\n\n```{py}\n%s\n```\n' "`cat src/util/confusion.py`" > rmd/converted_py/util/confusion.md
printf '## `util/data.py`\n\n```{py}\n%s\n```\n' "`cat src/util/data.py`" > rmd/converted_py/util/data.md
printf '## `util/duplication.py`\n\n```{py}\n%s\n```\n' "`cat src/util/duplication.py`" > rmd/converted_py/util/duplication.md
printf '## `util/parallel.py`\n\n```{py}\n%s\n```\n' "`cat src/util/parallel.py`" > rmd/converted_py/util/parallel.md
