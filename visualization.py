import cv2
import os
from utils import normalize
import numpy as np

def visualizer(pathes, anomaly_map, img_size, save_path, cls_name, gt_mask=None):
    for idx, path in enumerate(pathes):
        cls = path.split('/')[-2]
        filename = path.split('/')[-1]
        vis = cv2.cvtColor(cv2.resize(cv2.imread(path), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
        mask = normalize(anomaly_map[idx])
        pred_vis = apply_ad_scoremap(vis, mask)

        if gt_mask is not None:
            gt_binary = (np.squeeze(gt_mask[idx]) > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(gt_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            gt_vis = vis.copy()
            cv2.drawContours(gt_vis, contours, -1, (0, 255, 0), 2)
            cv2.drawContours(pred_vis, contours, -1, (0, 255, 0), 2)
            combined = np.concatenate([vis, gt_vis, pred_vis], axis=1)
        else:
            combined = pred_vis

        combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)  # BGR
        save_vis = os.path.join(save_path, 'imgs', cls_name[idx], cls)
        if not os.path.exists(save_vis):
            os.makedirs(save_vis)
        cv2.imwrite(os.path.join(save_vis, filename), combined)

def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)
