import os.path
import numpy as np
from model.hnsw import HnswBuild, HnswSearch
from model.bert_repre import BertRrepre
from utils import load_json, to_pkl, load_pkl


def build_hnsw_index(config, origin_data):
    hnsw = HnswBuild(config)

    # emb create
    model = BertRrepre(config["bert"])
    sent_repres = model.predict_repre(origin_data)
    inds = [i for i in range(len(origin_data))]

    data = {"emb": sent_repres, "inds": inds}
    print("data: ", len(data["emb"]), data["inds"])

    hnsw.build(data)


def search_hnsw_index(config):
    origin_data = [{"sentence1": "如何将网商贷切换为借呗"}]

    model = BertRrepre(config["bert"])
    sent_repres = model.predict_repre(origin_data)

    data = {"emb": sent_repres}
    hnsw = HnswSearch(config)
    pre_inds, distances = hnsw.search(data, 2)

    basic_info_path = os.path.join(config["data_dir"], "itchat_index.pkl")
    basic_info = load_pkl(basic_info_path)

    print("input: ", origin_data[0]["sentence1"])
    for ind in pre_inds[0]:
        print("pre: ", basic_info[ind]["std"])


def main():
    config_path = "../config/itchat_skill_config.json"
    config = load_json(config_path)

    data_path = "/Users/baojiang/project/itchat/data/itchat_data.txt"

    origin_data = []
    basic_info = {}
    with open(data_path, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            basic_info[index] = {"std": line.strip()}
            origin_data.append({"sentence1": line.strip()})

    data_index_path = os.path.join(config["data_dir"], "itchat_index.pkl")
    to_pkl(basic_info, data_index_path)

    build_hnsw_index(config, origin_data)
    search_hnsw_index(config)


if __name__ == "__main__":
    main()




