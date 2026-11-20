from torch import nn
from .audio_transformer import FrameAST_small,FrameAST_base
from .byol import MultiCropWrapper,ByolLoss
import torch
import argparse

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

class FrameATST(nn.Module):
    def __init__(self,
                 arch="small",
                 symmetric=True,
                 pos_type="cut",
                 avg_blocks=0,
                 patch_embed="Linear",
                 **kwargs):
        super().__init__()
        if arch == "small":
            encoder_fn = FrameAST_small
            embed_dim = 384
        elif arch == "base":
            encoder_fn = FrameAST_base
            embed_dim = 768
        else:
            raise RuntimeError("arch {} is not implemented".format(arch))
        self.symmetric = symmetric
        if avg_blocks==0: #atst-frame
            self.student=MultiCropWrapper(encoder_fn(pos_type=pos_type,patch_embed=patch_embed,**kwargs),
                                        embed_dim,
                                        predictor=True)
            self.teacher=MultiCropWrapper(encoder_fn(pos_type=pos_type,patch_embed=patch_embed,**kwargs),
                                        embed_dim,
                                        predictor=False)
        else: # data2vec, no projector, predictor is linear
            self.student=MultiCropWrapper(encoder_fn(pos_type=pos_type,patch_embed=patch_embed,**kwargs),
                                        embed_dim,
                                        projector="linear",
                                        predictor=False)
            self.teacher=MultiCropWrapper(encoder_fn(pos_type=pos_type,patch_embed=patch_embed,avg_blocks=8,**kwargs),
                                        embed_dim,
                                        projector=None,
                                        predictor=False)
        for p in self.teacher.parameters():
            p.requires_grad = False

        if avg_blocks==0: #atst-frame
            self.teacher.load_state_dict({k:v for k,v in self.student.state_dict().items() if "predictor" not in k })
        else: #data2vec
            self.teacher.load_state_dict({k:v for k,v in self.student.state_dict().items() if "projector" not in k })

        self.loss_fn = ByolLoss(symmetric=symmetric)
    
    def forward(self,x,length,mask):
        if self.symmetric:
            tea = self.teacher(x,length,mask,False)
            stu = self.student(x,length,mask,True)
            return self.loss_fn(stu,tea)
        else:
            tea = self.teacher(x[:1],length[:1],mask[:1],False)
            stu = self.student(x[1:],length[1:],mask[1:],True)
            return self.loss_fn(stu,tea)

    def update_teacher(self,m):
        with torch.no_grad():
            for param_q, param_k in zip(self.student.encoder.parameters(), self.teacher.encoder.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            for param_q, param_k in zip(self.student.projector.parameters(), self.teacher.projector.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
    def _init_teacher(self):
        self.teacher.load_state_dict({k:v for k,v in self.student.state_dict().items() if "predictor" not in k })