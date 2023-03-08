# Doesn't work since they have a lot of links to
# Rosstat and it is not allowing to download .csv's without tokens
# freely

import re

import cfscrape

TOKEN = "7560135ad22cf68ea64bb48b6d0b7c31"
scraper = cfscrape.create_scraper()


def get_data_record():
    url = f"https://data.gov.ru/api/dataset/?access_token={TOKEN}"
    return scraper.get(url).json()


def get_dataset_version(id):
    url = f"https://data.gov.ru/api/dataset/{id}/?access_token={TOKEN}"
    return scraper.get(url).json()["modified"]


def get_dataset_source_link(id, version):
    url = f"https://data.gov.ru/api/dataset/{id}/version/{version}/?access_token={TOKEN}"
    return scraper.get(url).json()[0]["source"]


data_records = get_data_record()

for record in data_records:
    id = record["identifier"]
    version = get_dataset_version(id)
    try:
        file_url = get_dataset_source_link(
            id, version) + f"/?access_token={TOKEN}"

        print(file_url)
    except Exception as _:
        continue

    if re.match("https:\/\/data\.gov\.ru.*", file_url):
        dataset_file = scraper.get(file_url).content

        with open(f"../govRu/{id}", "wb") as file:
            file.write(dataset_file)
