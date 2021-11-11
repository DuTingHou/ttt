from torch import nn
from torch.nn import functional as F
import torch
from transformers import AutoTokenizer, AutoModel


class Bert_Etity(nn.Module):
    def __init__(self, bert_name_or_path ,hidden_size=768):
        super(Bert_Etity,self).__init__()
        self.bert_name_or_path = bert_name_or_path
        self.hidden_size = hidden_size
        self.bert = AutoModel.from_pretrained(self.bert_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_name_or_path)
        self.catelayer_start = nn.Linear(self.hidden_size, 1)
        self.catelayer_end = nn.Linear(self.hidden_size, 1)

    def forward(self, data, att_mask):
        text = data
        result = self.bert(text, att_mask)
        seqs = result[0]
        catelogy_result_start = self.catelayer_start(seqs).reshape(-1,seqs.size(1))
        catelogy_result_end = self.catelayer_end(seqs).reshape(-1,seqs.size(1))
        catelogy_result_start = torch.pow(F.sigmoid(catelogy_result_start) , 2) 
        catelogy_result_end = torch.pow(F.sigmoid(catelogy_result_end) , 2) 
        return catelogy_result_start,catelogy_result_end,seqs

    def predict(self, text):
        text_encode = torch.Tensor(self.tokenizer.encode(text)).long().cuda()
        seqs, _ = self.bert(text_encode)
        catelogy_result_start = self.catelayer_start(seqs).reshape(-1,seqs.size(1))
        catelogy_result_end = self.catelayer_end(seqs).reshape(-1,seqs.size(1))
        return catelogy_result_start ,catelogy_result_end,seqs

class Relation(nn.Module):
    def __init__(self,seq_len=80 ,hidden_size=768,rel_num=24):
        super(Relation,self).__init__()
        self.hidden_size = hidden_size
        self.rel_num = rel_num
        self.seq_len =seq_len
        #self.Bilinear = nn.Bilinear(self.hidden_size , self.hidden_size , self.hidden_size)
        self.ln = nn.LayerNorm(self.hidden_size) 
        self.linear_start = nn.Linear(self.hidden_size , self.rel_num)
        self.linear_end = nn.Linear(self.hidden_size, self.rel_num)

    def forward(self, seqs ,selected_entity_start , selected_entity_end):
        batch_size = seqs.size(0)
        seq_len = seqs.size(1) 
        mask_base = torch.range(0,seq_len-1).reshape(1,-1).repeat(batch_size, 1).long().cuda()
        mask_start = (mask_base >= (selected_entity_start.reshape(batch_size ,1).long()))
        mask_end = (mask_base <= (selected_entity_end.reshape(batch_size ,1).long()))
        mask = (mask_start & mask_end)
        mask = mask.unsqueeze(2).repeat(1, 1, seqs.size(2))
        entity = (seqs * mask).sum(dim=1) / mask.long().sum(dim=1)
        new_seqs = seqs + entity.unsqueeze(dim=1)
        #new_seqs = self.Bilinear(seqs , entity.unsqueeze(dim=1))
        new_seqs = self.ln(new_seqs)
        cate_start = self.linear_start(new_seqs)
        cate_start = cate_start.transpose(2,1).reshape(-1,self.rel_num*seqs.size(1))
        cate_end = self.linear_end(new_seqs)
        cate_end = cate_end.transpose(2, 1).reshape(-1, self.rel_num * seqs.size(1))
        cate_start = torch.pow(F.sigmoid(cate_start),4)
        cate_end = torch.pow(F.sigmoid(cate_end),4)
        return cate_start,cate_end


class Entity_Relation(nn.Module):
    def __init__(self,bert_name_or_path="bert-base-uncased" , hidden_size=768,seq_len=80,rel_num=24):
        super(Entity_Relation, self).__init__()
        self.bert_name_or_path = bert_name_or_path
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.rel_num = rel_num
        self.bert_entity = Bert_Etity(self.bert_name_or_path,self.hidden_size)
        self.relation = Relation(seq_len=self.seq_len,hidden_size=self.hidden_size,rel_num=self.rel_num)

    def forward(self,input_ids,attention_mask,selected_entity_start ,selected_entity_end):
        catelogy_result_start ,catelogy_result_end,seqs = self.bert_entity(input_ids,attention_mask)

        object_start ,object_end = self.relation(seqs , selected_entity_start ,selected_entity_end)
        return catelogy_result_start,catelogy_result_end,object_start,object_end


if __name__=='__main__':
    er = Entity_Relation(bert_name_or_path="bert-base-chinese")
    input_ids = torch.randint(103,150,(2,80))
    attention_mask = torch.ones(2,80)
    select_start = torch.tensor([2,10])
    select_end = torch.tensor([5,13])
    result = er(input_ids,attention_mask,select_start,select_end)
    for item in result:
        print(item.shape)






