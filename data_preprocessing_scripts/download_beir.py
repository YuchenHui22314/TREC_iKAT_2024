import pathlib, os
from beir import util

dataset_list = [
    # "bioasq",     xxxxxx
    # "nq",
    # "hotpotqa",
    # "fiqa",
    # "signal1m",xxxxxx
    # "trec-news",xxxxxx
    "arguana",
    "webis-touche2020",
    "cqadupstack",
    "quora",
    "dbpedia-entity",
    "scidocs",
    "fever",
    "climate-fever",
    "scifact"
]


base_directory = "/data/rech/huiyuche/beir"

for dataset in dataset_list:
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(base_directory, dataset)
    data_path = util.download_and_unzip(url, out_dir)
    print("Dataset {} downloaded here: {}".format(dataset, data_path))