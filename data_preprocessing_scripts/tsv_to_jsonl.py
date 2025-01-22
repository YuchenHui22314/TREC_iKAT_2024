import csv
import json
import os
from argparse import ArgumentParser

from tqdm import tqdm

INPUT_FILE = ""
OUTPUT_FILE = ""
LOG_FILE = "" 


def main(input_file, output_file):
    with open(output_file, 'w') as output:
        for line in tqdm(open(input_file, "r")):
            try:
                ## processing of different cases
                parts = line.split('\t')

                pid = ""
                passage = ""
                url = ""

                if len(parts) < 3:
                    raise Exception("less than 2 tab in a line")
                elif len(parts) == 3:
                    pid, passage, url = parts 
                elif len(parts) > 3:
                    pid = parts[0]
                    url = parts[-1]
                    passage = '    '.join(parts[1:-1])
                else:
                    raise Exception("not possible to reach here!")

                if not url.startswith("http"):
                    raise Exception("part[-1] is seemingly not an url, please check.")

                ## write to output file
                obj = {"id": str(pid), "contents": passage}
                output.write(json.dumps(obj, ensure_ascii=False) + '\n')
            except Exception as e:
                # write to log
                with open(LOG_FILE, 'a') as log:
                    log.write("=====================\n")
                    log.write(f"Error in line: {line} \n")
                    log.write(f"Error: {e} \n")
                    log.write("=====================\n")
                continue

if __name__ == "__main__":
    main(INPUT_FILE, OUTPUT_FILE)
