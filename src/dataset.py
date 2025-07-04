import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image


class CocoDetectionDataset(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        
        # 读取COCO格式标注文件
        with open(ann_file, 'r', encoding='utf-8') as f:
            coco = json.load(f)
        self.images = coco['images']  # 图片信息列表
        self.annotations = coco['annotations']  # 标注信息列表
        
        # 建立image_id到标注的映射
        self.img_id_to_anns = {}
        for ann in self.annotations:
            self.img_id_to_anns.setdefault(ann['image_id'], []).append(ann)
        self.id_to_img = {img['id']: img for img in self.images}

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        
        # 获取该图片的所有标注
        anns = self.img_id_to_anns.get(img_info['id'], [])
        boxes = []
        labels = []
        for ann in anns:
            bbox = ann['bbox']
            # COCO格式bbox为[x, y, w, h]，转为[x1, y1, x2, y2]
            boxes.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
            labels.append(ann['category_id'] + 1)
            # labels.append(ann['category_id'])
        
        # 没有目标时，返回shape为(0, 4)的boxes和(0,)的labels
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # 构造目标检测模型需要的target字典
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_info['id']])
        }
        
        # 图像预处理
        if self.transforms:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.images)


# 用于DataLoader批量加载时的自定义打包函数
# 保证每个batch是元组列表，适配目标检测模型输入
def collate_fn(batch):
    return tuple(zip(*batch))
