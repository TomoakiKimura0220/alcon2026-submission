import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from data import read_pair, letterbox, tensor_image
from models import logits_of


def supervised_loss(logits, target):
    valid=target!=255
    if not valid.any(): return logits.sum()*0
    ce=F.cross_entropy(logits,target,ignore_index=255,reduction='none')
    weights=logits.new_tensor([1.,1.,2.])[target.clamp(0,2)]*valid
    ce=(ce*weights).sum()/weights.sum().clamp_min(1)
    prob=logits.softmax(1)*valid[:,None]
    truth=F.one_hot(target.masked_fill(~valid,0),3).permute(0,3,1,2).float()*valid[:,None]
    intersection=(prob*truth).sum((0,2,3))
    dice=(2*intersection+1)/(prob.sum((0,2,3))+truth.sum((0,2,3))+1)
    return ce+.5*(1-dice.mean())


def distillation_loss(student, teacher, target, temperature=2., confidence=.7):
    teacher=teacher.detach().float()
    raw=teacher.softmax(1)
    conf,pred=raw.max(1)
    # Initial implementation: annotated data only. Respect ignore and human labels.
    valid=(target!=255)&(conf>=confidence)&(pred==target)
    soft=(teacher/temperature).softmax(1)
    per_pixel=F.kl_div(F.log_softmax(student/temperature,dim=1),soft,
                       reduction='none').sum(1)*(temperature**2)
    weights=student.new_tensor([1.,1.,2.])[pred]*valid
    loss=(per_pixel*weights).sum()/weights.sum().clamp_min(1)
    coverage=valid.sum().float()/(target!=255).sum().clamp_min(1)
    return loss,coverage


def metrics(cm):
    cm=cm.double()
    intersection=cm.diag()
    union=cm.sum(0)+cm.sum(1)-intersection
    iou=torch.where(union>0,intersection/union,torch.nan)
    return {'iou': [None if torch.isnan(v) else float(v) for v in iou],
            'miou':float(iou.nanmean()),
            'weed_recall':float(intersection[2]/cm[2].sum()) if cm[2].sum()>0 else None,
            'confusion':cm.long().tolist()}


@torch.inference_mode()
def predict_image(model,image,size,device):
    model.eval()
    boxed,_,(left,top,nw,nh)=letterbox(image,None,size)
    x=tensor_image(boxed)[None].to(device)
    logits=logits_of(model,x)[...,top:top+nh,left:left+nw]
    logits=F.interpolate(logits,size=(image.height,image.width),mode='bilinear',align_corners=False)
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)


def weed_ratio(mask,valid=None):
    if valid is None: valid=np.ones(mask.shape,dtype=bool)
    rice=int(((mask==1)&valid).sum()); weed=int(((mask==2)&valid).sum())
    return weed/(rice+weed) if rice+weed else None


@torch.inference_mode()
def evaluate(model,rows,size,device,thresholds=None,save_dir=None):
    model.eval()
    cms=defaultdict(lambda:torch.zeros(3,3,dtype=torch.int64))
    items=[]
    palette=np.array([[60,60,60],[40,185,75],[235,75,55]],dtype=np.uint8)
    if save_dir: Path(save_dir).mkdir(parents=True,exist_ok=True)
    for index,row in enumerate(rows):
        image,target=read_pair(row)
        start=time.perf_counter()
        pred=predict_image(model,image,size,device)
        elapsed=time.perf_counter()-start
        valid=target!=255
        encoded=target[valid].astype(np.int64)*3+pred[valid]
        cm=torch.from_numpy(np.bincount(encoded,minlength=9).reshape(3,3))
        cms['all']+=cm; cms[row['domain']]+=cm
        true_ratio=weed_ratio(target,valid)
        # Mask error on exactly the same valid pixels, so ignore cannot skew the comparison.
        predicted_valid=weed_ratio(pred,valid)
        predicted_full=weed_ratio(pred)
        error=abs(predicted_valid-true_ratio) if predicted_valid is not None and true_ratio is not None else None
        # Ratio MAE requires near-complete masks; IoU still uses all valid pixels.
        ratio_eligible=float(valid.mean())>=.95
        level=int(np.searchsorted(thresholds,predicted_full,side='right')) if thresholds is not None and predicted_full is not None else None
        official=int(row['stage']) if row.get('stage','').strip() else None
        item={'image':row['image'],'domain':row['domain'],'valid_fraction':float(valid.mean()),
              'true_ratio_valid':true_ratio,'pred_ratio_valid':predicted_valid,
              'pred_ratio_full':predicted_full,'ratio_abs_error':error if ratio_eligible else None,
              'pred_stage':level,'official_stage':official,'seconds':elapsed}
        items.append(item)
        if save_dir:
            stem=f'{index:04d}_{Path(row["image"]).stem}'
            Image.fromarray(pred).save(Path(save_dir)/(stem+'_mask.png'))
            color=Image.fromarray(palette[pred])
            Image.blend(image,color,.45).save(Path(save_dir)/(stem+'_overlay.jpg'))
    report={}
    for domain,cm in cms.items():
        selected=items if domain=='all' else [r for r in items if r['domain']==domain]
        ratios=[r['ratio_abs_error'] for r in selected if r['ratio_abs_error'] is not None]
        stages=[r for r in selected if r['official_stage'] is not None and thresholds is not None]
        result=metrics(cm)
        result.update(n=len(selected),ratio_mae=float(np.mean(ratios)) if ratios else None,
                      ratio_n=len(ratios), stage_n=len(stages),
                      stage_correct=sum(r['pred_stage']==r['official_stage'] for r in stages) if stages else None)
        report[domain]=result
    return {'summary':report,'images':items,'thresholds':thresholds,
            'note':'No-plant predictions have null ratio/stage, not an assumed level 0. Stage boundaries are user supplied.'}


def atomic_save(value,path):
    path=Path(path)
    temp=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temp)
    temp.replace(path)
