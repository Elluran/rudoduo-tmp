from pathlib import Path
import tarfile
import logging
import gzip
import json
import shutil
import os
from multiprocessing import Pool
import hashlib
import pandas as pd
import chardet
import numpy as np


def main():
    logging.root.setLevel(logging.INFO)
    archives_paths = Path("./data/wdc").glob("*.gz")
    shutil.rmtree("./data/extracted_ru_wdc_tables/", ignore_errors=True)
    os.mkdir("./data/extracted_ru_wdc_tables/")
    
    for archive_path in archives_paths:
        with tarfile.open(archive_path, "r:gz") as tar:
            logging.info(f"Extracting: {archive_path}")
            tar.extractall("./tmp/wdc")
            folder_name = Path(archive_path.stem).stem
            jsons_paths = list(Path("./tmp/wdc").glob(f"./*/warc/*.gz"))
            logging.info(f"Reading json files")
            for idx, json_path in enumerate(jsons_paths):
                logging.info(f"{idx}/{len(jsons_paths)}")
                with gzip.open(json_path, "rt") as file:
                    json_strings = file.read().split("\n")
                    for string in json_strings:
                        try:
                            sample = json.loads(string)
                        except json.decoder.JSONDecodeError:
                            continue

                        not_relation = sample["tableType"] != "RELATION"
                        bad_header_pos = sample["headerPosition"] != "FIRST_ROW"
                        bad_orientation = sample["tableOrientation"] != "HORIZONTAL"
                        if not_relation or bad_header_pos or bad_orientation:
                            continue

                        text_after = sample["textAfterTable"]
                        text_before = sample["textBeforeTable"]
                        relation = np.array(sample["relation"])
                        relation_string_repr = " ".join(relation.flatten())
                        combined_text = relation_string_repr + text_after + " " + text_before
                        try:
                            lang_detection = chardet.detect(
                                combined_text.encode('cp1251'))
                            if lang_detection["language"] == "Russian" and lang_detection["confidence"] > 0.92:

                                header = relation[:, 0]
                                data = relation[:, 1:].T
                                df = pd.DataFrame(data, columns=header)
                                filename = hashlib.md5(
                                    relation_string_repr.encode('utf-8')).hexdigest()
                                df.to_csv(
                                    "./data/extracted_ru_wdc_tables/" + filename + ".csv", index=False)
                        except:
                            continue
            
            logging.info(f"Removing: ./tmp/wdc/" + folder_name)
            shutil.rmtree("./tmp/wdc/" + folder_name, ignore_errors=True)


if __name__ == "__main__":
    main()
