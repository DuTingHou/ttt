from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer,BertTokenizer
import json
from random import choice
from tqdm import tqdm

class dataset_NYT(Dataset):
    def __init__(self, data_file, rel2id_file,tokenizer: AutoTokenizer, max_len):
        self.data_file = data_file
        self.max_len = max_len
        self.tokenizer = tokenizer
        rels = json.loads(open(rel2id_file).read())
        self.rel2id = rels[1]
        self.id2rel = rels[0]
        self.datas = self.get_data()

    def get_mention_index(self,sentence_token ,mention_token):
        for i in range(0,len(sentence_token)):
            if sentence_token[i:i+len(mention_token)] == mention_token:
                return i
        return -1

    def get_data(self):
        datas = []
        for line in tqdm(open(self.data_file)):
            info = json.loads(line.strip("\n"))
            text, triples = info["sentText"], info["relationMentions"]
            text_infos = self.tokenizer(text, add_special_tokens=True
                                        , padding=True, truncation=True, max_length=self.max_len)
            triple_proced = []
            sub_head ,sub_tail = [0]*self.max_len,[0]*self.max_len
            obj_head,obj_tail = [0]*self.max_len*len(self.rel2id) ,[0]*self.max_len*len(self.rel2id)
            for triple in triples:
                #print(triple)
                if len(triple.values())!=3:
                    #print("#")
                    continue
                mention1, relation, mention2 = triple.get("em1Text"), triple.get("label"), triple.get("em2Text")
                mention1_e = self.tokenizer.encode(mention1)[1:-1]
                mention2_e = self.tokenizer.encode(mention2)[1:-1]
                mention1_start = self.get_mention_index(text_infos["input_ids"],mention1_e)
                mention2_start = self.get_mention_index(text_infos["input_ids"], mention2_e)
                if mention1_start == -1 or mention2_start == -1:
                    #print("##")
                    continue
                mention1_end = mention1_start + len(mention1_e) -1
                mention2_end = mention2_start + len(mention2_e) -1
                #print(len(text_infos["input_ids"]) )
                if mention1_end > (len(text_infos["input_ids"])-1) or mention2_end >(len(text_infos["input_ids"])-1):
                    print("###")
                    continue
                sub_head[mention1_start]=1
                sub_tail[mention1_end] =1

                triple_proced.append(((mention1_start ,mention1_end) ,
                                      self.rel2id[relation] ,(mention2_start,mention2_end)))
            if len(triple_proced)==0:
                continue
            selected_triple = choice(triple_proced)
            rel_id = selected_triple[1]
            mention2_start,mention2_end = selected_triple[2]
            #print(rel_id*len(self.rel2id)*self.max_len , mention2_start)
            #print(rel_id,self.max_len ,mention2_start ,len(obj_head) ,len(self.rel2id))
            #print(rel_id*self.max_len + mention2_start)
            obj_head[rel_id*self.max_len + mention2_start]=1
            obj_tail[rel_id*self.max_len + mention2_end]=1
            # for triple in triple_proced:
            #     mention1_start ,mention1_end = triple[0]
            #     # rel_id = triple[1]
            #     # mention2_start,mention2_end=triple[2]
            #     sub_head[mention1_start] = 1
            #     try:
            #         sub_tail[mention1_end] = 1
            #     except:
            #         print(triple)
            #         print(line)
            datas.append(
                            ((text_infos["input_ids"] ,text_infos["attention_mask"]),
                            (sub_head,sub_tail,
                            selected_triple[0][0],selected_triple[0][1],
                            obj_head,obj_tail) ,
                            info)
                        )
        return datas

    def collate_data(self, batch_data):
        bs_data = {}
        bs_data["input_ids"] = [item[0][0] for item in batch_data]
        bs_data["attention_mask"] = [item[0][1] for item in batch_data]
        bs_data["infos"] = [item[2] for item in batch_data]
        bs_data["sub_heads"] = [ item[1][0]for item in batch_data]
        bs_data["sub_tails"] = [item[1][1] for item in batch_data]
        bs_data["select_sub_head"] = [ item[1][2] for item in batch_data]
        bs_data["select_sub_tail"] = [item[1][3] for item in batch_data]
        bs_data["rel_obj_head"] = [item[1][4] for item in batch_data]
        bs_data["rel_obj_tail"] = [item[1][5] for item in batch_data]
        return bs_data









    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        return data
