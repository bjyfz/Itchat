from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import json
import numpy as np
from src.utils import load_json
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from src.model.transformers import (WEIGHTS_NAME, BertConfig,
                                    BertForSequenceClassification, BertTokenizer, BertModel,
                                    RobertaConfig, XLNetConfig,
                                    XLNetForSequenceClassification,
                                    XLNetTokenizer,
                                    AlbertForSequenceClassification)

from src.model.transformers import AdamW, WarmupLinearSchedule
from src.metrics.clue_compute_metrics import compute_metrics
from src.processors import clue_output_modes as output_modes
from src.processors import clue_processors as processors
from src.processors import clue_convert_examples_to_features as convert_examples_to_features
from src.processors import collate_fn, xlnet_collate_fn
from src.tools.common import seed_everything, save_numpy
from src.tools.common import init_logger, logger
from src.tools.progressbar import ProgressBar

MODEL_CLASSES = {
    # bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer, BertModel),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'roberta': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'albert': (BertConfig, AlbertForSequenceClassification, BertTokenizer)
}


class BertRrepre:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('mps')
        self.model, self.tokenizer = self.load_model()

    def load_model(self):
        config_class, model_class, tokenizer_class, bert_class = MODEL_CLASSES[self.config["model_type"]]
        tokenizer = tokenizer_class.from_pretrained(self.config["model_dir"], do_lower_case=self.config["do_lower_case"])
        logger.info("Predict the following checkpoints: %s", self.config["model_dir"])
        model = bert_class.from_pretrained(self.config["model_dir"])
        model.to(self.device)
        return model, tokenizer

    def predict_repre(self, data):
        sent_repres = []
        pred_dataset = self.load_and_cache_examples(data, self.tokenizer)

        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset,
                                     sampler=pred_sampler,
                                     batch_size=self.config["pred_batch_size"],
                                     collate_fn=xlnet_collate_fn if self.config["model_type"] in ['xlnet'] else collate_fn
                                     )

        logger.info("******** Running representation {} ********")
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", self.config["pred_batch_size"])

        for step, batch in enumerate(pred_dataloader):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1]}
                if self.config["model_type"] != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if (
                            'bert' in self.config["model_type"] or 'xlnet' in self.config["model_type"]) else None
                    # XLM, DistilBERT and RoBERTa don't use segment_ids
                hidden_reps, cls_reps = self.model(**inputs)
                sent_repres.extend(cls_reps.tolist())

        return sent_repres

    def predict_sim(self):
        pass

    def load_and_cache_examples(self, data, tokenizer):
        processor = processors[self.config["task_name"]]()
        output_mode = output_modes[self.config["task_name"]]

        label_list = processor.get_labels()

        examples = processor.get_input_data_examples(data)

        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=self.config["max_seq_length"],
                                                output_mode=output_mode,
                                                pad_on_left=bool(self.config["model_type"] in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if self.config["model_type"] in ['xlnet'] else 0,
                                                )

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
        if output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels)
        return dataset


def main():
    config_path = "../config/itchat_skill_config.json"
    config = load_json(config_path)
    repre = BertRrepre(config["bert"])


if __name__ == "__main__":
    main()
