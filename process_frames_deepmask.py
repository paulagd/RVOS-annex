import numpy as np
import os
from IPython import embed
import subprocess
from tqdm import tqdm, trange
import sys
import json
from PIL import Image
import argparse

from collections import namedtuple
import cv2

davis_path = "../databases/DAVIS2017/JPEGImages/1080p"
gt_path = '../databases/DAVIS2017/Annotations/1080p'
pred_path = 'results/np_masks_davis/'
frame = 0

DEEPMASK = "../deepmask"


def intersection_over_unit(target, prediction):
    # https://www.jeremyjordan.me/evaluating-image-segmentation-models/
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)

    # return the intersection over union value
    return iou_score

# define the `Detection` object
Detection = namedtuple("Detection", ["image_path", "gt", "pred"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gs", action="store_true", default=False, help="Enable to compute global statistics")
    args = parser.parse_args()

    j = 0
    statistics = {}
    global_iou = []
    global_recall = []
    for seq in tqdm(sorted(os.listdir(pred_path)), desc="Computing IoU..."):
        # print("... for ", seq)
        # if j == 5:
        #     break
        # j += 1
        # READ ANNOTATIONS
        gt_img = os.path.join(gt_path, seq, '%05d.png' % frame)
        annot = np.array(Image.open(gt_img))
        annot = np.expand_dims(annot, axis=0).squeeze()

        # READ PREDICTIONS
        pred_img = os.path.join(pred_path, seq, 'mask.npy')
        pred = np.load(pred_img)

        k_max_score = []
        seq_recall = []
        recall = 0
        # PROPOSALS VS GT
        seq_statistic = {}
        k = np.unique(annot)[1:]
        for i in k:
            # annot[np.where(annot==k[0])]
            gt_mask = np.zeros(annot.shape)
            gt_mask[annot == i] = 1
            # embed()
            iou_array = []
            recall = 0
            for j, pred_mask in enumerate(pred):
                # iou
                iou = intersection_over_unit(gt_mask, pred_mask)
                # TODO --> FER SI iou OR ioobject > 0.7 --> descarto
                # iou = intersection_over_unit(gt_mask, pred_mask)
                if iou > 0.5:
                    recall = 1
                iou_array.append(iou)
            max_ind = np.argmax(iou_array)
            seq_statistic[i] = {"recall": recall, "max_score": iou_array[max_ind]}
            global_iou.append(iou_array[max_ind])
            global_recall.append(recall)
            # k_max_score.append(iou_array[max_ind])

        statistics[seq] = seq_statistic
    print("----------- Statistics ----------")
    print(statistics)
    # with open('statistics.json', 'w') as f:
    #     json.dump(statistics, f, ensure_ascii=False)
    # COMPUTING STATISTICS
    # if(args.gs):
    #     with open('statistics.json', 'r') as fp:
    #         data = json.load(fp)
    #     embed()
    #     av_recall = []
    #     av_score = []
    #     for seq in tqdm(sorted(os.listdir(davis_path)), desc="Computing AVERAGE statistics..."):
    #         obj = data[seq]
    #         av_recall.append(obj['recall'])
    #         av_score.append(obj['max_score'])
    embed()
    y = np.array(global_iou)
    top_number = 5
    print("TOP 10 index", np.argsort(global_iou)[-top_number:])
    print("TOP 10 VALUES", y[np.argsort(global_iou[-top_number:])])
    print("---------------------------------------------------")
    print("DOWN 10 index", np.argsort(global_iou)[:top_number])
    print("DOWN 10 VALUES", y[np.argsort(global_iou[:top_number])])
    # TOP 10 index [1 0 7 6 5 3 9 8 2 4]
    print("---------------------------------------------------")
    print("TOTAL STATISTICS:")
    print(" >> Recall: ", np.mean(global_recall))
    print(" >> Score: ", np.mean(global_iou))
