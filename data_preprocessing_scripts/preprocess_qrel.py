import csv
import json
import os
from argparse import ArgumentParser

from tqdm import tqdm

INPUT_FILE = "input_qrel.txt"
OUTPUT_FILE = "output_qrel.txt"
LOG_FILE = "log.txt" 


def main(input_file, output_file):
    with open(output_file, 'w') as output:
        for line in tqdm(open(input_file, "r")):
            try:
                ## processing of different cases
                parts = line.strip().split(" ")
                parts[0] = parts[0].replace("_", "-")
                new_line = ' '.join(parts)
                ## write to output file
                output.write(new_line + '\n')
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