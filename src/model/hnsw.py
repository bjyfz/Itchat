import os
import hnswlib
import numpy as np

class HnswBuild():
    def __init__(self, config):
        self.config = config

    def build(self, data):
        num_elements = len(data["emb"])

        # Declaring index
        # possible options are l2, cosine or ip
        model = hnswlib.Index(space=self.config["hnsw"]["space"], dim=self.config["hnsw"]["dim"])

        # Initing index
        model.init_index(max_elements=num_elements, ef_construction=100, M=16)

        # Controlling the recall by setting ef:
        # higher ef leads to better accuracy, but slower search
        model.set_ef(10)

        # Set number of threads used during batch search/construction
        # By default using all available cores
        model.set_num_threads(4)

        # build data
        print("Adding %d elements" % (len(data["emb"])))
        model.add_items(data["emb"], data["inds"])
        print("Add done")

        # Serializing and deleting the index:
        index_name = self.config["hnsw"]["index_name"]
        index_path = os.path.join(self.config["data_dir"], f"{index_name}.bin")
        model.save_index(index_path)
        print("Saving index to '%s'" % index_path)


class HnswSearch():
    def __init__(self, config):
        self.config = config
        self.model = self.load_model()
        self.warm_up()

    def load_model(self):
        model = hnswlib.Index(space=self.config["hnsw"]["space"], dim=self.config["hnsw"]["dim"])
        index_name = self.config["hnsw"]["index_name"]
        index_path = os.path.join(self.config["data_dir"], f"{index_name}.bin")
        model.load_index(index_path)
        return model

    def warm_up(self):
        num_elements = 100
        dim = self.config["hnsw"]["dim"]
        data = np.float32(np.random.random((num_elements, dim)))
        self.model.knn_query(data, k=1)

    def search(self, data, k):
        inds, distances = self.model.knn_query(data["emb"][0], k=k)
        print("labels: ", inds)
        print("distances: ", distances)
        return inds, distances
