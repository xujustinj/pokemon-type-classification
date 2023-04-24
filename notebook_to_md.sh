jupytext --to markdown --update-metadata '{"jupytext": {"notebook_metadata_filter":"-all"}}' src/*.ipynb
mv src/*.md rmd/converted_ipynb
