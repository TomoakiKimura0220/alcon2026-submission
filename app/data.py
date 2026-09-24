"""Explicit splits and label formats; images and masks share geometry."""
import csv
import hashlib
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

MEAN = torch.tensor([.485, .456, .406])[:, None, None]
STD = torch.tensor([.229, .224, .225])[:, None, None]
FIELDS = ['image', 'mask', 'label_format', 'group', 'split', 'domain', 'stage']


def read_manifest(path):
    path = Path(path).resolve()
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not set(FIELDS[:-1]).issubset(reader.fieldnames or []):
            raise ValueError(f'Missing CSV columns: {FIELDS[:-1]}')
        rows = list(reader)
    seen_paths, group_splits = {}, {}
    for row in rows:
        for key in ('image', 'mask'):
            if not row[key].strip():
                raise ValueError(f'Empty {key}: {row}')
            p = Path(row[key])
            row[key] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
            if not Path(row[key]).is_file():
                raise FileNotFoundError(row[key])
        if row['split'] not in ('train', 'val', 'test'):
            raise ValueError('split must be train, val or test')
        if row['label_format'] not in ('rice6', 'alcon3'):
            raise ValueError('label_format must be rice6 or alcon3')
        if row['domain'] not in ('riceseg', 'field') or not row['group'].strip():
            raise ValueError('domain must be riceseg/field and group cannot be empty')
        if row['image'] in seen_paths:
            raise ValueError(f'Duplicate image: {row["image"]}')
        seen_paths[row['image']] = row['split']
        # Global group IDs: prefix RiceSEG/field IDs when preparing CSV.
        previous = group_splits.setdefault(row['group'], row['split'])
        if previous != row['split']:
            raise ValueError(f'Capture group crosses splits: {row["group"]}')
        if row.get('stage', '').strip() and row['stage'] not in ('0','1','2','3'):
            raise ValueError('stage must be blank or the official level 0/1/2/3')
    if not rows:
        raise ValueError('Empty manifest')
    return rows


def read_pair(row):
    with Image.open(row['image']) as im:
        image = im.convert('RGB')
    # Preserve P-mode palette indices. Reject rendered RGB masks instead of guessing colors.
    with Image.open(row['mask']) as im:
        mask = np.array(im)
    if mask.ndim != 2:
        raise ValueError(f'Mask must contain single-channel class IDs: {row["mask"]}')
    if image.size != (mask.shape[1], mask.shape[0]):
        raise ValueError(f'Image/mask dimensions differ: {row["image"]}')
    allowed = [0,1,2,3,4,5,255] if row['label_format'] == 'rice6' else [0,1,2,255]
    if not np.isin(mask, allowed).all():
        raise ValueError(f'Invalid class IDs {np.unique(mask)}: {row["mask"]}')
    if row['label_format'] == 'rice6':
        lut = np.full(256, 255, dtype=np.uint8)
        lut[:6] = [0,1,1,1,2,2]
        mask = lut[mask]
    return image, mask.astype(np.uint8)


def letterbox(image, mask, size):
    w, h = image.size
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w*scale)), max(1, round(h*scale))
    image = image.resize((nw,nh), Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', (size,size), (124,116,104))
    left, top = (size-nw)//2, (size-nh)//2
    canvas.paste(image,(left,top))
    outmask = np.full((size,size),255,dtype=np.uint8)
    if mask is not None:
        resized = np.array(Image.fromarray(mask).resize((nw,nh), Image.Resampling.NEAREST))
        outmask[top:top+nh,left:left+nw] = resized
    return canvas, outmask, (left,top,nw,nh)


def tensor_image(image):
    x = torch.from_numpy(np.array(image).copy()).permute(2,0,1).float()/255
    return (x-MEAN)/STD


class SegData(Dataset):
    def __init__(self, rows, size=512):
        self.rows, self.size = rows, size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        image, mask = read_pair(self.rows[idx])
        # Keep full-scene context in half of examples. Crop coordinates shared by mask.
        if random.random() < .5:
            w,h = image.size
            frac = random.uniform(.55,1.)
            cw,ch = max(1,round(w*frac)),max(1,round(h*frac))
            left,top = random.randint(0,w-cw),random.randint(0,h-ch)
            image = image.crop((left,top,left+cw,top+ch))
            mask = mask[top:top+ch,left:left+cw]
        if random.random() < .5:
            image = ImageOps.mirror(image)
            mask = mask[:,::-1].copy()
        image = ImageEnhance.Brightness(image).enhance(random.uniform(.8,1.2))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(.85,1.15))
        image = ImageEnhance.Color(image).enhance(random.uniform(.85,1.15))
        image,mask,_ = letterbox(image,mask,self.size)
        return tensor_image(image), torch.from_numpy(mask.astype(np.int64))


def audit(rows):
    counts = defaultdict(lambda: {'images':0,'pixels':[0,0,0], 'ignore_pixels':0})
    hashes = {}
    digest = hashlib.sha256()
    for i,row in enumerate(rows):
        image,mask = read_pair(row)
        rawhash = hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
        oldsplit = hashes.setdefault(rawhash,row['split'])
        if oldsplit != row['split']:
            raise ValueError(f'Identical decoded image crosses splits: {row["image"]}')
        if not (mask != 255).any():
            raise ValueError(f'All-ignore mask: {row["mask"]}')
        digest.update((rawhash+str(image.size)+row['group']+row['split']+row['domain']+
                       row.get('stage','')).encode())
        digest.update(mask.tobytes())
        key = row['split']+'/'+row['domain']
        counts[key]['images'] += 1
        for k in range(3):
            counts[key]['pixels'][k] += int((mask==k).sum())
        counts[key]['ignore_pixels'] += int((mask==255).sum())
        if (i+1)%500 == 0:
            print(f'audit {i+1}/{len(rows)}',flush=True)
    return dict(counts), digest.hexdigest()
