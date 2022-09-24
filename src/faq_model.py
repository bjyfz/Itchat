import hnswlib
import numpy as np
from utils import load_json, to_pkl, load_pkl
from model.bert_repre import BertRrepre


class FaqModel:
    def __init__(self, config):
        self.config = config
        self.bert_repre = BertRrepre(config["bert"])

    def load_model(self):
        pass

    def load_dict(self):
        pass

    def predict(self, record_id, query):
        sent_repres = self.bert_repre.predict_repre([{"sentence1": query}])

        return sent_repres


def main():
    config_path = "../config/itchat_skill_config.json"
    config = load_json(config_path)

    query = "如何将网商贷切换为借呗"
    record_id = "test"

    faq_model = FaqModel(config)

    res = faq_model.predict(record_id, query)

    #print(res)


if __name__ == "__main__":
    main()


