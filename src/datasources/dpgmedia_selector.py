# Download dpgmedia.jsonl from https://github.com/dpgmedia/partisan-news2019 (or directly from https://partisan-news2019.s3-eu-west-1.amazonaws.com/dpgMedia2019-articles-bypublisher.jsonl)
# The file should be about 300 MB.
# Rename the downloaded file to dpgmedia.jsonl
import json
import random
import jsonlines

random.seed(42)

titles = []
number_of_headlines_required = 3235

with jsonlines.open("../../data/raw/dpgmedia.jsonl") as f:
    for line in f.iter():
        if line["title"]:
            titles.append(line["title"])

unique_titles = list(set(titles))

random_headlines = random.sample(unique_titles, number_of_headlines_required)

assert len(set(random_headlines)) == len(random_headlines)

file_name = "../../data/processed/news.json"

with open(file_name, "w+", encoding="utf-8") as f:
    json.dump(random_headlines, f, ensure_ascii=False, indent=4)