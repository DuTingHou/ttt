import os,sys

import torch

sys.path.append(os.getcwd())
from erp.utils.dataset_NYT2 import dataset_NYT2
from erp.models.Bert_entity import Bert_Etity,Relation,Entity_Relation
from erp.models.CasRel import CasRel
from erp.losses.NYT_Loss import NYT_Loss
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from tqdm import tqdm
from example.test_metric import demo as metric_demo

def demo():
    data_dir = "./data/oth"
    bert_name_or_path = "./source/bert-base-chinese"
    dataset_nyt = dataset_NYT2(infile="{}/test.json".format(data_dir),
                              rel2id_file="{}/rel2id.json".format(data_dir))

    conf = {
            "bert_dim":768,
            "num_relations":len(dataset_nyt.rel2id),
            "bert_name":bert_name_or_path
            }
    erp = CasRel(conf).cuda()
    nyt_loss = NYT_Loss(1.0,1.5)
    opti = Adam(erp.parameters(),lr=1e-5)
    dl = DataLoader(dataset=dataset_nyt, batch_size=64,
                    shuffle=True, num_workers=2,
                    collate_fn=dataset_nyt.collate_data
                    )
    for e in range(0,1):
        pbar = tqdm(dl)
        for batch in pbar:
            batch_ids , batch_atten_mask ,sub_heads ,sub_tails ,\
            selected_sub_starts ,selected_sub_ends ,\
            obj_starts ,obj_tails  = batch["input_ids"].cuda() ,batch["attention_mask"].cuda(),\
            batch["sub_heads"].cuda() ,batch["sub_tails"].cuda() , batch["selected_sub_head"].cuda(),\
            batch["selected_sub_tail"].cuda() ,batch["obj_heads"].cuda(),batch["obj_tails"].cuda()
            
            pred_sub_starts, pred_sub_ends, pred_obj_starts, pred_obj_ends = \
                erp(batch_ids, batch_atten_mask, selected_sub_starts, selected_sub_ends)

            opti.zero_grad()
            loss = nyt_loss(pred_sub_starts, pred_sub_ends, pred_obj_starts, pred_obj_ends, \
                            sub_heads, sub_tails, obj_starts, obj_tails ,batch_atten_mask)
            loss.backward()
            opti.step()
            loss_str = round(loss.detach().cpu().numpy().tolist(), 3)
            pbar.set_description("loss:{}".format(loss_str))
            break
        torch.save(erp , "{}/{}_{}".format("models" , e , "erp.bin"))
        
        #metric_demo(e ,data_dir ,thred1=0.5,thred2=0.2)

if __name__=='__main__':
    demo()

