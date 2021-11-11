import torch
from torch import nn
from torch.nn import functional as F


class NYT_Loss(nn.Module):
    def __init__(self ,a ,b):
        super(NYT_Loss, self).__init__()
        self.a = a
        self.b = b 
        self.pos_weight = torch.tensor(a)

    def forward(self,
                pred_sub_starts,
                pred_sub_ends,
                pred_obj_starts,
                pred_obj_ends,
                label_sub_starts,
                label_sub_ends,
                label_obj_starts,
                label_obj_ends,
                atten_mask):

        sub_start_loss = F.binary_cross_entropy(pred_sub_starts,label_sub_starts ,reduction="none")
        sub_end_loss = F.binary_cross_entropy(pred_sub_ends,label_sub_ends,reduction="none")
        obj_start_loss = F.binary_cross_entropy(pred_obj_starts,label_obj_starts ,reduction="none")
        obj_end_loss = F.binary_cross_entropy(pred_obj_ends ,label_obj_ends ,reduction="none")
        sub_start_loss = (sub_start_loss * atten_mask).sum()/atten_mask.sum()
        sub_end_loss = (sub_end_loss * atten_mask).sum() / atten_mask.sum()
        obj_start_loss = (obj_start_loss * atten_mask.unsqueeze(-1)).sum()/atten_mask.sum()
        obj_end_loss = (obj_end_loss * atten_mask.unsqueeze(-1)).sum() / atten_mask.sum()
        
        loss = sub_start_loss + sub_end_loss + obj_start_loss + obj_end_loss
        return loss

