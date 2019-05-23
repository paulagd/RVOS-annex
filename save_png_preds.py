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
# gt_path = '../databases/DAVIS2017/Annotations/1080p'
# resolution = "480p"
resolution = "1080p"
# challenge_path = "../databases/test-challenge/"+resolution
# np_masks = 'results/np_masks_davis/'
# pred_path = 'results/png_images_davis/'

# pred_path = 'results/png_images_test_challenge_'+resolution+'/'
# np_masks = 'results/np_masks_test_challenge/'+resolution

np_masks = 'results/np_masks_test_dev/'+resolution
pred_path = 'results/png_images_test_dev_'+resolution+'/'


# img_dir = '/content/drive/My Drive/MASTER/TFM/RVOS/DAVIS_IMGS/davis_imgs/'

frame = 0

PALETTE = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64,
           0, 0, 191, 0, 0, 64, 128, 0, 191, 128, 0, 64, 0, 128]


def intersection_over_unit(target, prediction):
    # https://www.jeremyjordan.me/evaluating-image-segmentation-models/
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)

    # return the intersection over union value
    return iou_score


def intersection_over_object(target, prediction):
    # https://www.jeremyjordan.me/evaluating-image-segmentation-models/
    intersection = np.logical_and(target, prediction)
    # union = np.logical_or(target, prediction)
    io_object = np.sum(intersection) / np.sum(prediction == 1)

    # TODO:   np.sum(intersection) / prediction --> ELIMINAR PREDICTION MASK

    # return the intersection over union value
    return io_object


def NMS(masks):
    NMS_masks = []
    for index, current_mask in enumerate(tqdm(masks)):
        # iou
        if index == 0:
            NMS_masks.append(current_mask)
        else:
            discart = False
            for saved_mask in NMS_masks:
                iou = intersection_over_unit(saved_mask, current_mask)
                io_object = intersection_over_object(saved_mask, current_mask)
                if iou > 0.5 or io_object > 0.5:
                    # if iou > 0.3:
                    # print("IM GONNA BREAK")
                    discart = True
            if not discart:
                NMS_masks.append(current_mask)
    return NMS_masks


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gs", action="store_true", default=False, help="Enable to compute global statistics")
    args = parser.parse_args()

    z = 0
    # for seq in tqdm(sorted(os.listdir(davis_path))):
    for seq in tqdm(sorted(os.listdir(np_masks))):
        # seq = "blackswan"
        # if z == 1:
        #     break
        # masks = top.get_field('mask')
        masks = np.load(os.path.join(np_masks, seq, 'mask.npy'))
        masks = masks.squeeze(1)
        k, h, w = masks.shape

        # import matplotlib.pyplot as plt
        # mask_pred = (np.squeeze(masks[1, :, :]))
        # mask_pred = np.reshape(mask_pred, (h, w))
        # print(mask_pred.shape)
        # print(np.unique(mask_pred))
        # plt.imsave('test.png', mask_pred)
        masks = NMS(masks)

        prediction_png = np.zeros((h, w))
        for i in reversed(range(len(masks))):
            # set the current k value
            current_k = i+1
            # change ones for actual k value
            prediction_png[np.array(masks[i]) == 1] = current_k

        # aux_path = os.path.join('/content/drive/My Drive/MASTER/TFM/RVOS/DAVIS_IMGS/palette_preds/', seq)
        aux_path = os.path.join(pred_path, seq)
        os.makedirs(aux_path, exist_ok=True)

        a = Image.fromarray(prediction_png.astype(np.uint8), mode="P")
        a.putpalette(PALETTE)
        a.save(os.path.join(aux_path, '00000.png'))
        z += 1
