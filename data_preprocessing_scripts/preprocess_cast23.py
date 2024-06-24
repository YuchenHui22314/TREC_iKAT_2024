import json
import os
from tqdm import tqdm 

def convert_format(test_path, test_output_path):
    # Load the data from the provided file
    with open(test_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Create a new list to store the data in JSON Lines format
    json_lines_data = []
    # Iterate through each item in the original data
    for item in data:
        queries = []
        number = item["number"]
        title = item["title"]
        ptkb = item["ptkb"]
        for turn in item["turns"]:
            if turn['turn_id'] == '1':
                queries = []
            turn_id = f"{number}-{turn['turn_id']}"
            queries.append(turn["utterance"])
            json_line = {
                "sample_id": turn_id, # topic-tree-turn
                "cur_utt_text": turn["utterance"],
                "oracle_utt_text": turn["resolved_utterance"],
                "ctx_utts_text": queries[:-1],
                "cur_response_text": turn["response"],
                "response_provenance": turn["response_provenance"],
                "number": number,
                "title": title,
                "ptkb": ptkb,   
                "ptkb_provenance": turn["ptkb_provenance"]
                }
            json_lines_data.append(json_line)

    with open(test_output_path, "w") as output_file:
        json.dump(json_lines_data, output_file, indent=4)


def gen_raw_passage_collection(root_path):
    files = os.listdir(root_path)
    #breakpoint()
    with open('cast23_collection.tsv', "w") as fw:
        for filename in tqdm(files):
            file_path = "{}/{}".format(root_path, filename)
            print(file_path)
            if "bz2" in file_path:
                continue
            with open(file_path, "r") as f:
                data = f.readlines()
            for line in tqdm(data):
                line = json.loads(line)
                pid = line['id']
                url = line['url']
                contents = line['contents']
                fw.write('{}\t{}\t{}'.format(pid, contents, url))
                fw.write('\n')

def combine_file(original, rewrite, passage, new):
    with open(original, "r", encoding="utf-8") as f1, open(rewrite, "r", encoding="utf-8") as f2, open(passage, "r", encoding="utf-8") as f3:
        data1 = f1.readlines()
        data2 = f2.readlines()
        data3 = f3.readlines()
        assert len(data1) == len(data2)
        assert len(data2) == len(data3)
        
    with open(new, "w", encoding="utf-8") as g:
        for i in range(len(data1)):
            record_1 = json.loads(data1[i])
            record_2 = json.loads(data2[i])
            record_3 = json.loads(data3[i])
            record_1["rewrite"], record_1["response"], record_1["passage"] = record_2["rewrite"], record_2["response_rewrite"], record_3["passage"]
            g.write(json.dumps(record_1) + "\n")

def count_num(collection):
    with open(collection, "r", encoding="utf-8") as f:
        cur = 0
        for line in f:
            cur += 1
        print(cur)


if __name__ == '__main__':
    test_path = "/data/rech/huiyuche/TREC_iKAT_2024/data/topics/2023_ikat_test_topics.json"
    test_output_path = "/data/rech/huiyuche/TREC_iKAT_2024/data/topics/2023_ikat_test_topics_flattened.json"
    collection = "cast23_collection.tsv"
    rewrite = "23test_response_rewite.jsonl"
    passage = "23testpassage_resloved_utterance.jsonl"
    new = "2023_test_topics_new_LLM.jsonl"
    convert_format(test_path, test_output_path)
    #root_path = "./collection"
    #gen_raw_passage_collection(root_path)
    #count_num(collection)
    #combine_file(test_output_path, rewrite, passage, new)
