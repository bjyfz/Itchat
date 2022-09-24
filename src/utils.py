import json
import pickle


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_pkl(data, path):
    with open(path, "wb+") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pkl(path):
    pkl_file = open(path, "rb")
    return pickle.load(pkl_file)


if __name__ == "__main__":
    a = load_json("../config/itchat_skill_config.json")
    print(a["data_dir"])
