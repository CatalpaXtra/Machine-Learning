import os
import json
import torch
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from model import get_model


def predict(score_thresh=0.1):
    # 相关参数
    test_img_dir = 'data/test/images'
    pred_json_path = 'data/test/pred.json'
    model_save_path = 'model/model.pth'
    num_classes = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 进行推理
    with open(pred_json_path, 'r', encoding='utf-8') as f:
        pred_json = json.load(f)
    results = []
    for img_info in tqdm(pred_json['images'], desc="Inferencing"):
        img_path = os.path.join(test_img_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        img_tensor = T.ToTensor()(img).to(device)
        with torch.no_grad():
            output = model([img_tensor])[0]
        
        # 遍历每个检测框，按阈值筛选
        for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
            if score < score_thresh:
                continue
            x1, y1, x2, y2 = box.tolist()
            results.append({
                "image_id": img_info['id'],
                "category_id": int(label) - 1,
                "bbox": [x1, y1, x2-x1, y2-y1],
                "score": float(score),
            })

    # 输出结果
    print(f"Number of reasoning results: {len(results)}")
    pred_json['annotations'] = results
    with open(pred_json_path, 'w', encoding='utf-8') as f:
        json.dump(pred_json, f, ensure_ascii=False, indent=2)
    print(f"Saved predictions to {pred_json_path}") 