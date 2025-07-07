import os
import json
import torch
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from rcnn.model import get_model
import numpy as np
from collections import defaultdict
import torchvision.ops


def predict_single_model(model_path, test_img_dir, pred_json_path, score_thresh=0.5, num_classes=4):
    """对单个模型进行预测"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 读取测试图片信息
    with open(pred_json_path, 'r', encoding='utf-8') as f:
        pred_json = json.load(f)
    
    results = []
    for img_info in tqdm(pred_json['images'], desc=f"Inferencing {os.path.basename(model_path)}"):
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
    
    return results


def ensemble_predictions(all_results, ensemble_method='nms', nms_threshold=0.5):
    """集成多个模型的预测结果"""
    # 按image_id分组
    grouped_predictions = defaultdict(list)
    
    for fold_results in all_results:
        for pred in fold_results:
            image_id = pred['image_id']
            grouped_predictions[image_id].append(pred)
    
    # 集成预测结果
    final_results = []
    
    for image_id, predictions in grouped_predictions.items():
        if ensemble_method == 'average':
            # 按category_id分组，对相同类别的框进行平均
            category_predictions = defaultdict(list)
            for pred in predictions:
                category_predictions[pred['category_id']].append(pred)
            
            for category_id, cat_preds in category_predictions.items():
                # 平均bbox坐标和分数
                avg_bbox = np.mean([pred['bbox'] for pred in cat_preds], axis=0)
                avg_score = np.mean([pred['score'] for pred in cat_preds])
                
                final_results.append({
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": avg_bbox.tolist(),
                    "score": float(avg_score),
                })
                
        elif ensemble_method == 'max':
            # 按category_id分组，取最高分数的预测
            category_predictions = defaultdict(list)
            for pred in predictions:
                category_predictions[pred['category_id']].append(pred)
            
            for category_id, cat_preds in category_predictions.items():
                best_pred = max(cat_preds, key=lambda x: x['score'])
                final_results.append(best_pred)
                
        elif ensemble_method == 'nms':
            # 使用torchvision的NMS进行集成
            final_results.extend(ensemble_with_nms(predictions, nms_threshold))
    
    return final_results


def ensemble_with_nms(predictions, nms_threshold=0.5):
    """使用torchvision的NMS集成同一张图片的多个预测结果"""
    if not predictions:
        return []
    
    # 按category_id分组
    category_predictions = defaultdict(list)
    for pred in predictions:
        category_predictions[pred['category_id']].append(pred)
    
    final_results = []
    
    for category_id, cat_preds in category_predictions.items():
        if len(cat_preds) == 1:
            # 只有一个预测，直接添加
            final_results.append(cat_preds[0])
            continue
        
        # 转换为tensor格式用于NMS
        boxes = []
        scores = []
        preds_dict = []  # 保存原始预测字典
        
        for pred in cat_preds:
            # 将[x, y, w, h]转换为[x1, y1, x2, y2]
            x, y, w, h = pred['bbox']
            boxes.append([x, y, x + w, y + h])
            scores.append(pred['score'])
            preds_dict.append(pred)
        
        # 转换为tensor
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        
        # 应用NMS
        keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, nms_threshold)
        
        # 保留NMS后的预测结果
        for idx in keep_indices:
            final_results.append(preds_dict[idx])
    
    return final_results


def ensemble_with_weighted_nms(predictions, nms_threshold=0.5, weight_by_fold=True):
    """使用加权NMS集成多个模型的预测结果"""
    if not predictions:
        return []
    
    # 按category_id分组
    category_predictions = defaultdict(list)
    for pred in predictions:
        category_predictions[pred['category_id']].append(pred)
    
    final_results = []
    
    for category_id, cat_preds in category_predictions.items():
        if len(cat_preds) == 1:
            final_results.append(cat_preds[0])
            continue
        
        # 转换为tensor格式
        boxes = []
        scores = []
        preds_dict = []
        
        for pred in cat_preds:
            x, y, w, h = pred['bbox']
            boxes.append([x, y, x + w, y + h])
            scores.append(pred['score'])
            preds_dict.append(pred)
        
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        
        # 应用NMS
        keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, nms_threshold)
        
        # 对NMS后的结果进行加权平均
        for idx in keep_indices:
            # 找到与当前框IoU较高的其他框进行加权平均
            current_box = boxes_tensor[idx]
            current_score = scores_tensor[idx]
            
            # 计算与其他保留框的IoU
            similar_boxes = []
            similar_scores = []
            
            for other_idx in keep_indices:
                if other_idx == idx:
                    continue
                
                other_box = boxes_tensor[other_idx]
                iou = calculate_iou(current_box, other_box)
                
                # 如果IoU较高，认为是同一个目标的不同预测
                if iou > 0.3:  # 可以调整这个阈值
                    similar_boxes.append(other_box)
                    similar_scores.append(scores_tensor[other_idx])
            
            # 如果有相似的框，进行加权平均
            if similar_boxes:
                all_boxes = torch.cat([current_box.unsqueeze(0)] + [b.unsqueeze(0) for b in similar_boxes], dim=0)
                all_scores = torch.cat([current_score.unsqueeze(0)] + [s.unsqueeze(0) for s in similar_scores], dim=0)
                
                # 按分数加权平均
                weights = all_scores / all_scores.sum()
                weighted_box = (all_boxes * weights.unsqueeze(1)).sum(dim=0)
                weighted_score = all_scores.mean()
                
                # 转换回[x, y, w, h]格式
                x1, y1, x2, y2 = weighted_box.tolist()
                final_results.append({
                    "image_id": preds_dict[idx]["image_id"],
                    "category_id": category_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(weighted_score),
                })
            else:
                # 没有相似的框，直接使用当前预测
                x1, y1, x2, y2 = current_box.tolist()
                final_results.append({
                    "image_id": preds_dict[idx]["image_id"],
                    "category_id": category_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(current_score),
                })
    
    return final_results


def calculate_iou(box1, box2):
    """计算两个框的IoU"""
    # box格式: [x1, y1, x2, y2]
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 计算交集
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # 计算并集
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def predict_kfold_models(k=5, score_thresh=0.5, ensemble_method='nms', nms_threshold=0.5):
    """对K折交叉验证的多个模型分别进行预测和集成"""
    # 相关参数
    test_img_dir = 'data/test/images'
    pred_json_path = 'data/test/pred.json'
    model_dir = 'model'
    num_classes = 4
    
    # 检查模型文件
    model_files = []
    for i in range(1, k+1):
        model_path = os.path.join(model_dir, f'model_kfold_fold{i}.pth')
        if os.path.exists(model_path):
            model_files.append(model_path)
        else:
            print(f"Warning: Model file {model_path} not found")
    
    if not model_files:
        print("No model files found!")
        return
    
    print(f"Found {len(model_files)} model files: {model_files}")
    
    # 对每个模型进行预测
    all_results = []
    for model_path in model_files:
        print(f"\n=== Predicting with {os.path.basename(model_path)} ===")
        results = predict_single_model(
            model_path=model_path,
            test_img_dir=test_img_dir,
            pred_json_path=pred_json_path,
            score_thresh=score_thresh,
            num_classes=num_classes
        )
        all_results.append(results)
        print(f"Model {os.path.basename(model_path)}: {len(results)} predictions")
    
    # 集成预测结果
    print(f"\n=== Ensemble predictions using {ensemble_method} method ===")
    if ensemble_method == 'nms':
        print(f"NMS threshold: {nms_threshold}")
    final_results = ensemble_predictions(all_results, ensemble_method, nms_threshold)
    
    # 输出结果
    print(f"Final ensemble results: {len(final_results)} predictions")
    
    # 保存结果
    with open(pred_json_path, 'r', encoding='utf-8') as f:
        pred_json = json.load(f)
    pred_json['annotations'] = final_results
    
    with open(pred_json_path, 'w', encoding='utf-8') as f:
        json.dump(pred_json, f, ensure_ascii=False, indent=2)
    print(f"Saved ensemble predictions to {pred_json_path}")


def predict_single_model_validation(model_path, test_img_dir, pred_json_path, score_thresh=0.5, num_classes=4):
    """对单个模型进行预测并保存单独结果"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 读取测试图片信息
    with open(pred_json_path, 'r', encoding='utf-8') as f:
        pred_json = json.load(f)
    
    results = []
    for img_info in tqdm(pred_json['images'], desc=f"Inferencing {os.path.basename(model_path)}"):
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
    
    # 保存单个模型的结果
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    # 提取fold编号
    fold_num = model_name.split('_')[-1]  # 获取 'fold1', 'fold2' 等
    single_pred_path = f'data/test/pred{fold_num}.json'
    
    # 创建完整的预测JSON结构
    pred_json_copy = {
        'images': pred_json['images'],
        'annotations': results,
        'categories': pred_json.get('categories', [])  # 如果有categories信息也保留
    }
    
    with open(single_pred_path, 'w', encoding='utf-8') as f:
        json.dump(pred_json_copy, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {model_name} predictions to {single_pred_path}")
    print(f"Model {model_name}: {len(results)} predictions")
    return results


def predict_kfold_individual_validation(k=5, score_thresh=0.5):
    """对K折交叉验证的每个模型分别进行预测和验证"""
    # 相关参数
    test_img_dir = 'data/test/images'
    pred_json_path = 'data/test/pred.json'
    model_dir = 'model'
    num_classes = 4
    
    # 检查模型文件
    model_files = []
    for i in range(1, k+1):
        model_path = os.path.join(model_dir, f'model_kfold_fold{i}.pth')
        if os.path.exists(model_path):
            model_files.append(model_path)
        else:
            print(f"Warning: Model file {model_path} not found")
    
    if not model_files:
        print("No model files found!")
        return
    
    print(f"Found {len(model_files)} model files for individual validation")
    print("Will generate separate prediction files for each fold model:")
    for i in range(1, k+1):
        print(f"  - pred{i}.json (from model_kfold_fold{i}.pth)")
    
    # 对每个模型分别进行预测和验证
    individual_results = {}
    total_predictions = 0
    
    for model_path in model_files:
        print(f"\n=== Individual validation for {os.path.basename(model_path)} ===")
        results = predict_single_model_validation(
            model_path=model_path,
            test_img_dir=test_img_dir,
            pred_json_path=pred_json_path,
            score_thresh=score_thresh,
            num_classes=num_classes
        )
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        individual_results[model_name] = results
        total_predictions += len(results)
    
    print(f"\n=== Summary ===")
    print(f"Total models processed: {len(model_files)}")
    print(f"Total predictions across all models: {total_predictions}")
    print(f"Average predictions per model: {total_predictions / len(model_files):.1f}")
    
    # 列出生成的文件
    print(f"\nGenerated prediction files:")
    for i in range(1, k+1):
        pred_file = f'data/test/predfold{i}.json'
        if os.path.exists(pred_file):
            print(f"  ✓ {pred_file}")
        else:
            print(f"  ✗ {pred_file} (not found)")
    
    return individual_results


def predict(score_thresh=0.5):
    """原始的单模型预测函数"""
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


def main(predict_mode='kfold_ensemble', score_thresh=0.5, k=5, ensemble_method='nms', nms_threshold=0.5):
    """主函数，支持多种预测模式"""
    if predict_mode == 'single':
        # 单模型预测
        predict(score_thresh=score_thresh)
    elif predict_mode == 'kfold_ensemble':
        # K折模型集成预测
        predict_kfold_models(k=k, score_thresh=score_thresh, ensemble_method=ensemble_method, nms_threshold=nms_threshold)
    elif predict_mode == 'kfold_individual':
        # K折模型分别预测验证
        predict_kfold_individual_validation(k=k, score_thresh=score_thresh)
    else:
        print(f"Unknown predict_mode: {predict_mode}")


if __name__ == '__main__':
    main()