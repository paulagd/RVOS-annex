import numpy as np
import os
from IPython import embed
import subprocess
from tqdm import tqdm, trange
import sys
import json
from PIL import Image

from collections import namedtuple
import cv2

davis_path = "../databases/DAVIS2017/JPEGImages/1080p"
gt_path = '../databases/DAVIS2017/Annotations/1080p'
pred_path = 'results/masks/'
frame = 0

DEEPMASK = "../deepmask"


if __name__ == "__main__":

    with open('statistics.json', 'r') as fp:
        data = json.load(fp)

    av_recall = []
    av_score = []
    for seq in tqdm(sorted(os.listdir(davis_path)), desc="Computing AVERAGE statistics..."):
        obj = data[seq]
        av_recall.append(obj['recall'])
        av_score.append(obj['max_score'])

    print("---------------------------------------------------")
    print("TOTAL STATISTICS:")
    print(" >> Recall: ", np.mean(av_recall))
    print(" >> Score: ", np.mean(av_score))
