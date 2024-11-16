import csv
from tqdm import tqdm

# INPUT_FILE = "/data/rech/huiyuche/test_multi_tab.tsv"
# OUTPUT_FILE = "/data/rech/huiyuche/TREC_iKAT_2024/data/collections/ikat_23/test.tsv"
INPUT_FILE = "/part/01/Tmp/yuchen/cast23_collection.tsv"
OUTPUT_FILE = "/part/01/Tmp/yuchen/cluweb22B_ikat_v2.tsv"
LOG_FILE = "/data/rech/huiyuche/TREC_iKAT_2024/logs/preprocessing_log.txt" 


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

                ## write to output tsv file (just id and coneents)
                output.write(f"{pid}\t{passage}\n")

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
