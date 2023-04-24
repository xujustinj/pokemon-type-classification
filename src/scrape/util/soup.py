from os import path

from bs4 import BeautifulSoup
import requests


def load_soup(load_path: str) -> BeautifulSoup:
    """Load a web element from a local file.

    Args:
        load_path (str): The path to the local file.

    Returns:
        soup (BeautifulSoup): The web element.
    """
    with open(load_path, "r") as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    return soup


def save_soup(soup: BeautifulSoup, save_path: str) -> None:
    """Save a web element to a local file.

    Args:
        soup (BeautifulSoup): The web element.
        save_path (str): The path to the local file.
    """
    with open(save_path, "w") as f:
        f.write(str(soup))


def fetch_soup(url: str, cache_path: str) -> BeautifulSoup:
    """Retrieve a webpage.

    Args:
        url (str): The URL of the webpage.
        cache_path (str): The path to a local file where the page is cached.

    Returns:
        soup (BeautifulSoup): The webpage.
    """
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
    """Parse a webpage element that has a string.

    Args:
        soup (BeautifulSoup): The webpage element.

    Returns:
        value (str): The string, with whitespace trimmed.
    """

    return soup.text.strip()


def parse_int(soup: BeautifulSoup) -> int:
    """Parse a webpage element that has an integer.

    For example, an element containing 441 (and possibly some text after) is
    parsed as 441.

    Args:
        soup (BeautifulSoup): The webpage element.

    Returns:
        value (integer): The value of the integer.
    """

    return int(parse_str(soup).split()[0])


def parse_float(soup: BeautifulSoup) -> int:
    """Parse a webpage element that has a decimal number.

    For example, an element containing 44.1 (and possibly some text after) is
    parsed as 44.1.

    Args:
        soup (BeautifulSoup): The webpage element.

    Returns:
        value (float): The value of the decimal number.
    """
    return float(parse_str(soup).split()[0])


def parse_percent(soup: BeautifulSoup) -> float:
    """Parse a webpage element that has a percentage.

    For example, an element containing 44.1% (and possibly some text after) is
    parsed as 44.1.

    Args:
        soup (BeautifulSoup): The webpage element.

    Returns:
        value (float): The value of the percentage.
    """
    raw = parse_str(soup).split()[0]
    assert raw.endswith("%")
    return float(raw[:-1])
