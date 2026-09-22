import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os

def generate_class_info(dataset_name):
    class_name_map_class_id = {}
    if dataset_name == 'mvtec':
        obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    elif dataset_name == 'visa':
        obj_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif dataset_name == 'mpdd':
        obj_list = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']
    elif dataset_name == 'btad':
        obj_list = ['01', '02', '03']
    elif dataset_name == 'DAGM_KaggleUpload':
        obj_list = ['Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10']
    elif dataset_name == 'SDD':
        obj_list = ['electrical commutators']
    elif dataset_name == 'DTD':
        obj_list = ['Woven_001', 'Woven_127', 'Woven_104', 'Stratified_154', 'Blotchy_099', 'Woven_068', 'Woven_125', 'Marbled_078', 'Perforated_037', 'Mesh_114', 'Fibrous_183', 'Matted_069']
    elif dataset_name == 'colon':
        obj_list = ['colon']
    elif dataset_name == 'ISBI':
        obj_list = ['skin']
    elif dataset_name == 'Chest':
        obj_list = ['chest']
    elif dataset_name == 'thyroid':
        obj_list = ['thyroid']
    for k, index in zip(obj_list, range(len(obj_list))):
        class_name_map_class_id[k] = index

    return obj_list, class_name_map_class_id

class Dataset(data.Dataset):
    def __init__(self, root, transform, target_transform, dataset_name, mode='test', zoom_aug_p=0.0, zoom_target=(0.10, 0.45)):
        self.root = root
        self.zoom_aug_p = zoom_aug_p
        self.zoom_target = zoom_target
        self.transform = transform
        self.target_transform = target_transform
        self.data_all = []
        meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
        name = self.root.split('/')[-1]
        meta_info = meta_info[mode]

        self.cls_names = list(meta_info.keys())
        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)

        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)
    def __len__(self):
        return self.length

    def _sample_crop_box(self, img, m, target_range):
        # square box that contains the whole anomaly (mask m) and makes it fill `target_range`
        # of the box's area. Returns None if no valid box exists (mask empty, or lesion already
        # too large for the image to leave any room to zoom out).
        if not m.any():
            return None
        W, H = img.size
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        side = int(np.sqrt(m.sum() / random.uniform(*target_range)))
        side = min(max(side, y1 - y0, x1 - x0, 32), W, H)
        if side >= min(W, H):
            return None
        lo_x, hi_x = max(0, x1 - side), min(x0, W - side)
        lo_y, hi_y = max(0, y1 - side), min(y0, H - side)
        if lo_x > hi_x or lo_y > hi_y:
            return None
        cx, cy = random.randint(lo_x, hi_x), random.randint(lo_y, hi_y)
        return (cx, cy, cx + side, cy + side)

    def _zoom_around_anomaly(self, img, img_mask):
        # square crop that keeps the whole anomaly and makes it fill `zoom_target` of the crop
        m = np.array(img_mask) > 127
        if img_mask.size != img.size:
            return img, img_mask
        box = self._sample_crop_box(img, m, self.zoom_target)
        if box is None:
            return img, img_mask
        return img.crop(box), img_mask.crop(box)

    def sample_scale_pair(self, index, scale_ranges=((0.05, 0.15), (0.25, 0.45)), max_tries=10):
        """
        For an anomalous sample, build two crops around the SAME lesion at two different,
        non-overlapping target scales (default: lesion fills 5-15% of crop A's area vs 25-45%
        of crop B's area) -- i.e. two views of one lesion seen "zoomed out" vs "zoomed in".

        Returns (img_a, img_b, box_a_norm, box_b_norm) where img_a/img_b are transformed image
        tensors (via self.transform) and box_*_norm = (x0, y0, x1, y1) is the lesion's own
        bounding box in [0, 1] coordinates relative to that crop (guaranteed to lie inside
        [0, 1]^2 since every crop is built to fully contain the lesion). Returns None if this
        sample has no usable mask or no valid crop could be found for either scale.
        """
        data = self.data_all[index]
        if data['anomaly'] != 1:
            return None
        img_path, mask_path = data['img_path'], data['mask_path']
        img = Image.open(os.path.join(self.root, img_path))
        if os.path.isdir(os.path.join(self.root, mask_path)):
            return None
        m = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
        if not m.any():
            return None
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1

        views = []
        for target_range in scale_ranges:
            box = None
            for _ in range(max_tries):
                box = self._sample_crop_box(img, m, target_range)
                if box is not None:
                    break
            if box is None:
                return None
            cx, cy, cx2, cy2 = box
            side = cx2 - cx
            crop = img.crop(box)
            box_norm = ((x0 - cx) / side, (y0 - cy) / side, (x1 - cx) / side, (y1 - cy) / side)
            views.append((self.transform(crop), torch.tensor(box_norm, dtype=torch.float32)))
        (img_a, box_a), (img_b, box_b) = views
        return img_a, img_b, box_a, box_b

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                              data['specie_name'], data['anomaly']
        img = Image.open(os.path.join(self.root, img_path))
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            if os.path.isdir(os.path.join(self.root, mask_path)):
                # just for classification not report error
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
        if self.zoom_aug_p > 0 and anomaly == 1 and random.random() < self.zoom_aug_p:
            img, img_mask = self._zoom_around_anomaly(img, img_mask)
        # transforms
        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(   
            img_mask) if self.target_transform is not None and img_mask is not None else img_mask
        img_mask = [] if img_mask is None else img_mask
        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
                'img_path': os.path.join(self.root, img_path), "cls_id": self.class_name_map_class_id[cls_name]}    