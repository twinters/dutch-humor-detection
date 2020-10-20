import requests
from bs4 import BeautifulSoup
from cachier import cachier
import json


@cachier(cache_dir="../../.cachier")
def scrape_kidsweek_page_raw(page_nr):
    url = "https://www.kidsweek.nl/moppen?page=" + str(page_nr)
    response = requests.get(url)
    return response.text


def is_advertisement(div):
    return "ADVERTENTIE" in div.text


def scrape_kidsweek_jokes_from_page(page_nr):
    page = BeautifulSoup(scrape_page_raw(page_nr), "html.parser")
    content = page.find("div", class_="view-content")
    jokes_divs = content.find_all("div", class_="views-row")
    jokes = [extract_joke(div) for div in jokes_divs if not is_advertisement(div)]
    print("Pagina", page_nr, jokes)
    return [joke for joke in jokes if len(joke.strip()) > 0]


def extract_joke(div):
    joke_text_parts_divs = div.find_all("div", class_="field-items")
    joke_text_parts = [el.text for el in joke_text_parts_divs]
    return "\n".join(joke_text_parts).replace("\xa0", " ")


def scrape_all_jokes():
    all_jokes = []
    page_nr = 0
    stop = False
    while not stop:
        new_jokes = scrape_jokes_from_page(page_nr=page_nr)
        page_nr += 1

        if len(new_jokes) == 0:
            stop = True
            break

        all_jokes += new_jokes
    return all_jokes


file_name = "../../data/raw/kidsweek.json"
with open(file_name, "w+", encoding="utf-8") as f:
    f.write("")
    all_jokes = scrape_all_jokes()
    json.dump(all_jokes, f, ensure_ascii=False, indent=4)
