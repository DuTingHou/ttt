import torch
from transformers import AutoTokenizer
import numpy as np
from torch.nn import functional as F
import json


class Metric2(object):
    def __init__(self, erp, rel2id_file="./data/raw/nyt10/rel2id.json", thred=0.5, thred2=0.35):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.erp = erp
        self.thred = thred
        self.thred2 = thred2
        rels = json.loads(open(rel2id_file).read())
        self.rel2id = rels[1]
        self.id2rel = rels[0]

    def predict(self, text):
        tokens = self.tokenizer(text)
        token_torch = torch.tensor([tokens["input_ids"]]).long().cuda()
        mask_torch = torch.tensor([tokens["attention_mask"]]).long().cuda()
        encoded_text = self.erp.get_encoded_text(token_torch, mask_torch)
        pred_sub_heads, pred_sub_tails = self.erp.get_subs(encoded_text)
        sub_heads = torch.where(pred_sub_heads[0] > self.thred)[0]
        sub_tails = torch.where(pred_sub_tails[0] > self.thred2)[0]
        subjects = []
        for sub_head in sub_heads:
            sub_tail = sub_tails[sub_tails >= sub_head]
            if len(sub_tail) > 0:
                sub_tail = sub_tail[0]
                subject = ''.join(self.tokenizer.decode(tokens["input_ids"][sub_head: sub_tail + 1]).split())
                subjects.append((subject, sub_head, sub_tail))
        if subjects:
            triple_list = []
            repeated_encoded_text = encoded_text.repeat(len(subjects), 1, 1)
            sub_head_mapping = torch.zeros((len(subjects), 1, encoded_text.size(1)), dtype=torch.float).cuda()
            sub_tail_mapping = torch.zeros((len(subjects), 1, encoded_text.size(1)), dtype=torch.float).cuda()
            for subject_idx, subject in enumerate(subjects):
                sub_head_mapping[subject_idx][0][subject[1]] = 1
                sub_tail_mapping[subject_idx][0][subject[2]] = 1
            pred_obj_heads, pred_obj_tails = self.erp.get_objs_for_specific_sub(sub_head_mapping, sub_tail_mapping,
                                                                                repeated_encoded_text)
            for subject_idx, subject in enumerate(subjects):
                sub = subject[0]
                obj_heads = torch.where(pred_obj_heads[subject_idx] > self.thred)
                obj_tails = torch.where(pred_obj_tails[subject_idx] > self.thred2)
                for obj_head, rel_head in zip(*obj_heads):
                    for obj_tail, rel_tail in zip(*obj_tails):
                        if obj_head <= obj_tail and rel_head == rel_tail:
                            rel = self.id2rel[str(int(rel_head))]
                            obj = ''.join(self.tokenizer.decode(tokens["input_ids"][obj_head: obj_tail + 1]).split())
                            triple_list.append((sub, rel, obj))
                            break

            triple_set = set()
            for s, r, o in triple_list:
                triple_set.add((s, r, o))
            pred_list = list(triple_set)
        else:
            pred_list = []
        return pred_list

