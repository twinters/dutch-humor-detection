from typing import List

import requests
from bs4 import BeautifulSoup, Tag, NavigableString
from cachier import cachier
import json

agent = {
    "User-Agent": "Joke Data Collection Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36"
}


@cachier(cache_dir="../../.cachier")
def scrape_raw(url):
    response = requests.get(url, headers=agent)
    return response.text


@cachier(cache_dir="../../.cachier")
def scrape_debestemoppen_page_raw(category, page_nr):
    url = "https://www.debestemoppen.nl/category/{}/page/{}".format(
        category, str(page_nr)
    )
    response = requests.get(url, headers=agent)
    return response.text


@cachier(cache_dir="../../.cachier")
def scrape_debestemoppen_main_page_raw():
    url = "https://www.debestemoppen.nl/"
    response = requests.get(url, headers=agent)
    return response.text


category_page_start = "https://www.debestemoppen.nl/category/"


def get_categories():
    main_page = BeautifulSoup(scrape_main_page_raw(), "html.parser")
    sidebar = main_page.find("ul", class_="sidebarmenu")
    elements = sidebar.find_all("li", class_="cat-item")
    categories = []
    for el in elements:
        if isinstance(el, NavigableString):
            continue
        link = el.find("a").get("href")
        category = link[len(category_page_start) :].replace("/", "")
        categories.append(category)
    return categories


def contains_image(div):
    return div.find("img")


def scrape_jokes_from_page(category, page_nr):
    page = BeautifulSoup(scrape_page_raw(category, page_nr), "html.parser")
    content = page.find("div", id="main")
    jokes_divs = content.find_all("div", class_="postbox")
    jokes = [extract_joke(div) for div in jokes_divs if not contains_image(div)]
    return [joke for joke in jokes if joke is not None and len(joke.strip()) > 0]


def scrape_joke_page(url):
    joke_page = BeautifulSoup(scrape_raw(url), "html.parser")
    box = joke_page.find("div", class_="postbox")
    all_paragraphs = box.find_all("p")
    joke = "\n".join([para.text for para in all_paragraphs])
    print("Scraped ", url, "for", joke)
    return joke


def extract_joke(div: Tag):
    joke_content_div = div.find(class_="postcontent")
    if joke_content_div:
        joke = joke_content_div.text

        if "Lees verder" in joke:
            joke = scrape_joke_page(
                joke_content_div.find("a", class_="more-link").get("href")
            )

        joke = joke.replace("\nLaat het antwoord zien", "").replace("\xa0", " ")
        return joke.strip()


# Ban English jokes
banned_categories = ["jokes"]


def scrape_all_jokes():
    all_jokes = []

    categories = [cat for cat in get_categories() if cat not in banned_categories]

    for cat in categories:
        page_nr = 0
        stop = False
        while not stop:
            new_jokes = scrape_jokes_from_page(category=cat, page_nr=page_nr)
            page_nr += 1

            if len(new_jokes) == 0:
                stop = True
                break

            all_jokes += new_jokes
    return all_jokes


file_name = "../../data/raw/debestemoppen.json"
with open(file_name, "w+", encoding="utf-8") as f:
    f.write("")
    all_jokes = scrape_all_jokes()
    json.dump(all_jokes, f, ensure_ascii=False, indent=4)
