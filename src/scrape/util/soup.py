from os import path

from bs4 import BeautifulSoup
import requests


def load_soup(load_path: str) -> BeautifulSoup:
    with open(load_path, "r") as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    return soup


def save_soup(soup: BeautifulSoup, save_path: str) -> None:
    with open(save_path, "w") as f:
        f.write(str(soup))


def fetch_soup(url: str, cache_path: str) -> BeautifulSoup:
    if path.exists(cache_path):
        # print(f"Resource {url} found in cache at {cache_path}")
        soup = load_soup(cache_path)
    else:
        # print(f"Resource {url} not in cache, fetching...")
        req = requests.get(url)
        soup = BeautifulSoup(req.text, 'html.parser')
        # print(f"Saving resource {url} to cache at {cache_path}")
        save_soup(soup, cache_path)
    return soup


def parse_str(soup: BeautifulSoup) -> str:
    return soup.text.strip()


def parse_int(soup: BeautifulSoup) -> int:
    return int(parse_str(soup).split()[0])


def parse_percent(soup: BeautifulSoup) -> float:
    raw = parse_str(soup).split()[0]
    assert raw.endswith("%")
    return float(raw[:-1])