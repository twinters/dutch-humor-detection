from typing import List

import requests
from bs4 import BeautifulSoup, Tag, NavigableString
from cachier import cachier
import json

agent = {"User-Agent": "Joke Data Collection Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36"}


@cachier(cache_dir="../../.cachier")
def scrape_lachjekrom_page_raw(rel_url):
    url = "https://www.lachjekrom.com" + rel_url
    response = requests.get(url, headers=agent)
    return response.text


@cachier(cache_dir="../../.cachier")
def scrape_lachjekrom_main_page_raw():
    url = "https://www.lachjekrom.com/moppen/"
    response = requests.get(url, headers=agent)
    return response.text


def get_categories():
    main_page = BeautifulSoup(scrape_lachjekrom_main_page_raw(), "html.parser")
    group_item_list = main_page.find_all("div", class_="itemsummaryblock")
    elements = [li for g in group_item_list for li in g.find_all("li")]
    categories = []
    for el in elements:
        if isinstance(el, NavigableString):
            continue
        link = el.find("a").get("href")
        categories.append(link)
    return categories


def extract_joke(li: Tag):
    return li.text.replace("\r\n", "\n").strip()


def scrape_jokes_from_page(category_url):
    page = BeautifulSoup(scrape_lachjekrom_page_raw(category_url), "html.parser")
    joke_containers = page.find_all("ul", class_="moppenlist")
    jokes = []
    for container in joke_containers:
        jokes_lis = container.find_all("li")
        jokes.extend([extract_joke(li) for li in jokes_lis])
    return [joke for joke in jokes if joke is not None and len(joke.strip()) > 0]


def scrape_all_jokes():
    all_jokes = []

    categories = get_categories()

    for cat in categories:
        all_jokes.extend(scrape_jokes_from_page(cat))
    return all_jokes


file_name = "../../data/raw/lachjekrom.json"
with open(file_name, "w+", encoding="utf-8") as f:
    f.write("")
    all_jokes = scrape_all_jokes()
    json.dump(all_jokes, f, ensure_ascii=False, indent=4)
