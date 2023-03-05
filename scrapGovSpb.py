import re
import os
import zipfile

import requests


PAGES_NUMBER = 37
DOMAIN = "https://classif.gov.spb.ru"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36"
}

for page in range(15, PAGES_NUMBER):
    print("Processing page №", page + 1)

    url = DOMAIN + f"/irsi/?page={page + 1}"
    page_html = requests.get(url, headers).text
    csv_links = re.findall(r"\/.*export_data\/", page_html)

    for csv_link in csv_links:
        print("Getting", csv_link)

        csv_zip = requests.get(DOMAIN + csv_link, headers).content

        try:
            with open("../govSpb/tmp.zip", "wb") as file:
                file.write(csv_zip)

            with zipfile.ZipFile("../govSpb/tmp.zip", "r") as zip:
                zip.extractall("../govSpb/")

            os.remove("../govSpb/tmp.zip")

        except Exception as e:
            print("Cannot download csv file", e)
