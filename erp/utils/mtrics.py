import torch
from transformers import AutoTokenizer
import numpy as np
from torch.nn import functional as F
import json

class Metric(object):
    def __init__(self ,bert_encode , relation ,rel2id_file="./data/raw/nyt10/rel2id.json" ,thred=0.5 ,thred2 = 0.35):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.bert_encode = bert_encode
        self.relaiton = relation
        self.thred = thred
        self.thred2 = thred2
        rels = json.loads(open(rel2id_file).read())
        self.rel2id = rels[1]
        self.id2rel = rels[0]

    def parser_sub_from_pred(self,text, sub_heads_logits, sub_tails_logits ,thred):
        tokens = self.tokenizer.encode(text)
        #print(sub_heads_logits)
        #print(sub_tails_logits)
        #print(tokens)
        sub_heads , sub_tails = np.where(sub_heads_logits > self.thred)[0], np.where(sub_tails_logits > self.thred2)[0]
        subjects = []
        #print(sub_heads,sub_tails)
        for sub_head in sub_heads:
            sub_tail = sub_tails[sub_tails >= sub_head]
            #print(sub_tail)
            if len(sub_tail) > 0 :
                sub_tail = sub_tail[0]
                #print(tokens[sub_head:sub_tail+1] ,self.tokenizer.decode(tokens[sub_head:sub_tail+1]))
                subject = tokens[sub_head:sub_tail+1]
                subjects.append((subject,sub_head,sub_tail))
        return subjects

    def predict(self,text):
        #tokenize  =self.tokenizer.tokenize(text)
        predict_triples = []
        tokened = self.tokenizer([text])
        tokened["attention_mask"] = torch.Tensor(tokened["attention_mask"]).long().cuda()
        tokened["input_ids"] = torch.Tensor(tokened["input_ids"]).long().cuda()
        sub_heads_logits, sub_tails_logits, seqs = self.bert_encode(tokened["input_ids"],tokened["attention_mask"])
        #sub_heads_logits = F.sigmoid(sub_heads_logits).detach().cpu().numpy()
        #sub_tails_logits = F.sigmoid(sub_tails_logits).detach().cpu().numpy()
        sub_heads_logits = sub_heads_logits.detach().cpu().numpy()
        sub_tails_logits = sub_tails_logits.detach().cpu().numpy()
        subjects = self.parser_sub_from_pred(text , sub_heads_logits[0],sub_tails_logits[0] ,thred = self.thred)
        for subject in subjects:
            subject_token ,sub_head ,sub_tail = subject[0],subject[1],subject[2]
            subject_str = self.tokenizer.decode(subject_token)
            #print(subject_token,subject_str)
            sub_head_in = torch.Tensor([sub_head]).long().cuda()
            sub_tail_in = torch.Tensor([sub_tail]).long().cuda()
            #print(sub_head_in, sub_tail_in)
            object_start ,object_end  = self.relaiton(seqs , sub_head_in,sub_tail_in)
            #object_start = F.sigmoid(object_start)
            #object_end = F.sigmoid(object_end)
            object_start = object_start.detach().cpu().numpy()[0]
            object_end = object_end.detach().cpu().numpy()[0]
            for i in range(0,self.relaiton.rel_num):
                rel_object_start = object_start[i*seqs.size(1):(i+1)*seqs.size(1)]
                rel_object_end = object_end[i*seqs.size(1):(i+1)*seqs.size(1)]
                objects = self.parser_sub_from_pred(text , rel_object_start,rel_object_end ,self.thred2)
                for object in objects:
                    object_str = self.tokenizer.decode(object[0])
                    predict_triples.append((subject_str , self.id2rel[str(i)] ,object_str))
        return predict_triples








