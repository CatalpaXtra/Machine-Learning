import json
import os
from PIL import Image
import shutil


def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"错误: JSON文件格式错误 {file_path}: {e}")
        return None


def save_json_file(data, file_path):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"成功保存文件: {file_path}")
        return True
    except Exception as e:
        print(f"错误: 保存文件失败 {file_path}: {e}")
        return False


def is_bbox_outside_image(bbox, image_width, image_height):
    xmin, ymin, width, height = bbox
    
    # 计算边界框的右下角坐标
    xmax = xmin + width
    ymax = ymin + height
    
    # 检查边界框是否完全在图片外侧
    if xmax <= 0 or ymax <= 0:  # 边界框完全在图片左侧或上方
        return True
    if xmin >= image_width or ymin >= image_height:  # 边界框完全在图片右侧或下方
        return True
    
    # 检查边界框是否部分超出图片边界
    # 如果边界框的任何部分超出图片边界，则认为是违法的
    if xmin < 0 or ymin < 0 or xmax > image_width or ymax > image_height:
        return True
    
    return False


def clean_annotations(train_json_path, images_dir):
    print("开始清洗标注数据...")
    
    # 加载JSON数据
    data = load_json_file(train_json_path)
    if data is None:
        return None
    
    # 创建图片ID到图片信息的映射
    image_info_map = {}
    for img in data.get("images", []):
        image_info_map[img["id"]] = {
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"]
        }
    
    print(f"找到 {len(image_info_map)} 张图片")
    
    # 统计信息
    total_annotations = len(data.get("annotations", []))
    removed_annotations = 0
    valid_annotations = []
    
    # 处理每个标注
    for annotation in data.get("annotations", []):
        image_id = annotation["image_id"]
        image_info = image_info_map[image_id]
        image_path = os.path.join(images_dir, image_info["file_name"])
        
        with Image.open(image_path) as img:
            actual_width, actual_height = img.size
        
        # 检查边界框是否在图片外侧
        bbox = annotation["bbox"]
        if is_bbox_outside_image(bbox, actual_width, actual_height):
            print(f"移除边界框在图片外侧的标注: 图片ID {image_id}, 边界框 {bbox}, 图片尺寸 {actual_width}x{actual_height}")
            removed_annotations += 1
        else:
            valid_annotations.append(annotation)
    
    # 更新数据
    data["annotations"] = valid_annotations
    
    # 打印统计信息
    print(f"\n清洗完成:")
    print(f"  总标注数量: {total_annotations}")
    print(f"  移除标注数量: {removed_annotations}")
    print(f"  保留标注数量: {len(valid_annotations)}")
    print(f"  移除比例: {removed_annotations/total_annotations*100:.2f}%" if total_annotations > 0 else "  移除比例: 0%")
    
    return data


def backup_original_file(file_path):
    backup_path = file_path + ".backup"
    if not os.path.exists(backup_path):
        try:
            
            shutil.copy2(file_path, backup_path)
            print(f"已备份原始文件: {backup_path}")
            return True
        except Exception as e:
            print(f"警告: 备份文件失败: {e}")
            return False
    else:
        print(f"备份文件已存在: {backup_path}")
        return True


def main():
    # 文件路径
    train_json_path = "data/train/train.json"
    images_dir = "data/train/images"
    
    print("=== 数据清洗工具 ===")
    print(f"训练数据文件: {train_json_path}")
    print(f"图片目录: {images_dir}")
    print()
    
    # 检查文件是否存在
    if not os.path.exists(train_json_path):
        print(f"错误: 找不到文件 {train_json_path}")
        return
    
    if not os.path.exists(images_dir):
        print(f"错误: 找不到目录 {images_dir}")
        return
    
    # 备份原始文件
    if not backup_original_file(train_json_path):
        print("警告: 无法备份原始文件，建议手动备份后再继续")
        response = input("是否继续清洗数据? (y/N): ")
        if response.lower() != 'y':
            print("操作已取消")
            return
    
    # 清洗数据
    cleaned_data = clean_annotations(train_json_path, images_dir)
    
    if cleaned_data is None:
        print("清洗失败")
        return
    
    # 保存清洗后的数据
    print("\n保存清洗后的数据...")
    if save_json_file(cleaned_data, train_json_path):
        print("数据清洗完成!")
    else:
        print("保存失败!")


if __name__ == "__main__":
    main() 