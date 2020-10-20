# This class merely combines all the other joke datasets into one json
import json

all_jokes_jsons = ["debestemoppen.json", "kidsweek.json", "lachjekrom.json"]
file_name = "../../data/processed/jokes.json"
with open(file_name, "w+", encoding="utf-8") as f:
    f.write("")
    all_jokes = []

    for joke_json in all_jokes_jsons:
        with open("../data/raw/" + joke_json, encoding="utf-8") as json_file:
            all_jokes.extend(json.load(json_file))

    # make unique
    all_jokes = list(set(all_jokes))

    json.dump(all_jokes, f, ensure_ascii=False, indent=4)
