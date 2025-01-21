import json
import csv
from tqdm import tqdm

INPUT_FILE = "input.jsonl"  # Replace with your input JSONL file
OUTPUT_FILE = "output.tsv"  # Replace with your desired output TSV file name
LOG_FILE = "conversion_log.txt"


def reverse_conversion(input_file, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')

        for line in tqdm(open(input_file, "r", encoding='utf-8')):
            try:
                data = json.loads(line)
                pid = data.get("id", "")
                passage = data.get("contents", "")
                # We don't have the URL in the JSONL, so we'll add a placeholder
                url = "http://example.com/" + str(pid)  

                writer.writerow([pid, passage, url])

            except Exception as e:
                with open(LOG_FILE, 'a', encoding='utf-8') as log:
                    log.write("=====================\n")
                    log.write(f"Error in line: {line} \n")
                    log.write(f"Error: {e} \n")
                    log.write("=====================\n")
                continue


if __name__ == "__main__":
    reverse_conversion(INPUT_FILE, OUTPUT_FILE)