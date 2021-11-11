import os,sys
sys.path.append(os.getcwd())
from erp.utils.dataset_NYT2 import dataset_NYT2
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def demo():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    dataset_nyt = dataset_NYT2(infile="./data/test.json",
                              rel2id_file="./data/rel2id.json")
    print(len(dataset_nyt.datas))
    dl = DataLoader(dataset=dataset_nyt, batch_size=2,
                    shuffle=True, num_workers=1,
                    collate_fn=dataset_nyt.collate_data
                    )
    for batch in dl:
        print(batch)
        break

if __name__=='__main__':
    demo()

