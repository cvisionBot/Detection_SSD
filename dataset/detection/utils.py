import cv2
import torch
import numpy as np

from cv2 import cvtColor


def collater(data):
    imgs = [s['image'] for s in data]
    bboxes = [torch.tensor(s['bboxes']) for s in data]
    batch_size = len(imgs)
    
    max_num_annots = max(annots.shape[0] for annots in bboxes)
    
    if max_num_annots > 0:
        padded_annots = (torch.ones((batch_size, max_num_annots, 5)) * -1)
        for idx, annot in enumerate(bboxes):
            if annot.shape[0] > 0:
                padded_annots[idx, :annot.shape[0], :] = annot
    else:
        padded_annots = torch.ones((batch_size, 1, 5)) * -1
    return {'img' : torch.stack(imgs), 'annot' : padded_annots}


def visualize(images, bboxes, batch_idx=0):
    img = images[batch_idx].numpy()
    img = (np.transpose(img, (1, 2, 0)) * 255.).astype(np.uint8).copy()
    
    for b in bboxes[batch_idx]:
        cx, cy, w, h, cid = b.numpy()
        x1 = (cx - w / 2) * 416
        y1 = (cy - h / 2) * 416
        w = w * 416
        h = h * 416
        x2 = (w + x1)
        y2 = (h + y1)
        
        if cid > -1:
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0))
    cv2.imwrite('./dataset/detection/annot_test.png', img)