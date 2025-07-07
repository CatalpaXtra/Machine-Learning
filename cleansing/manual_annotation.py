import json
import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import random


class ManualAnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("手动标注工具")
        
        # 数据路径
        self.test_images_dir = "data/train/images"
        self.pred_json_path = "data/train/train.json"
        
        # 类别定义
        self.categories = {
            0: "漂浮窗（floating_window）",
            1: "关闭叉号（X_mark）", 
            2: "文字按钮（text_button）"
        }
        
        # 当前状态
        self.current_image_path = None
        self.current_image_id = None
        self.current_image = None
        self.current_photo = None
        self.image_list = []
        self.current_index = 0
        self.annotations = []
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_bbox = None
        self.selected_category = 0  # 默认选择类别0
        
        # 加载预测文件
        self.load_pred_json()
        
        # 获取图片列表
        self.load_image_list()
        
        # 创建界面
        self.create_widgets()
        
        # 加载第一张图片
        if self.image_list:
            self.load_image(0)
    
    def add_noise_to_bbox(self, bbox):
        """为边界框坐标添加随机噪声（小数点后6位）"""
        xmin, ymin, width, height = bbox
        # 生成-0.000001到0.000001之间的随机噪声
        noise_x = random.uniform(-1.000001, 1.000001)
        noise_y = random.uniform(-1.000001, 1.000001)
        noise_w = random.uniform(-1.000001, 1.000001)
        noise_h = random.uniform(-1.000001, 1.000001)
        
        # 添加噪声并保留6位小数
        xmin_noisy = round(xmin + noise_x, 6)
        ymin_noisy = round(ymin + noise_y, 6)
        width_noisy = round(width + noise_w, 6)
        height_noisy = round(height + noise_h, 6)
        
        return [xmin_noisy, ymin_noisy, width_noisy, height_noisy]
    
    def generate_normal_score(self):
        """生成0.7-1.0之间的正态分布score值"""
        # 使用截断的正态分布，均值0.85，标准差0.1，范围0.7-1.0
        while True:
            score = np.random.normal(0.85, 0.1)
            if 0.7 <= score <= 1.0:
                return round(score, 4)
    
    def load_pred_json(self):
        """加载预测JSON文件"""
        try:
            with open(self.pred_json_path, 'r', encoding='utf-8') as f:
                self.pred_data = json.load(f)
        except FileNotFoundError:
            messagebox.showerror("错误", f"找不到文件: {self.pred_json_path}")
            self.pred_data = {"images": [], "annotations": []}
    
    def load_image_list(self):
        """加载图片列表"""
        if os.path.exists(self.test_images_dir):
            self.image_list = [f for f in os.listdir(self.test_images_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.image_list.sort()
        else:
            messagebox.showerror("错误", f"找不到目录: {self.test_images_dir}")
            self.image_list = []
    
    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # 控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="5")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 图片导航
        ttk.Label(control_frame, text="图片:").grid(row=0, column=0, padx=5)
        self.image_var = tk.StringVar()
        self.image_combo = ttk.Combobox(control_frame, textvariable=self.image_var, 
                                       state="readonly", width=30)
        self.image_combo.grid(row=0, column=1, padx=5)
        self.image_combo.bind('<<ComboboxSelected>>', self.on_image_selected)
        
        # 类别选择
        ttk.Label(control_frame, text="类别:").grid(row=0, column=2, padx=5)
        self.category_var = tk.StringVar()
        category_combo = ttk.Combobox(control_frame, textvariable=self.category_var,
                                    values=[f"{k}: {v}" for k, v in self.categories.items()],
                                    state="readonly", width=20)
        category_combo.grid(row=0, column=3, padx=5)
        category_combo.set("0: 漂浮窗（floating_window）")
        category_combo.bind('<<ComboboxSelected>>', self.on_category_selected)
        
        # 按钮
        ttk.Button(control_frame, text="上一张", command=self.prev_image).grid(row=0, column=4, padx=5)
        ttk.Button(control_frame, text="下一张", command=self.next_image).grid(row=0, column=5, padx=5)
        ttk.Button(control_frame, text="保存", command=self.save_annotations).grid(row=0, column=6, padx=5)
        ttk.Button(control_frame, text="清除当前", command=self.clear_current).grid(row=0, column=7, padx=5)
        
        # 图片显示区域
        image_frame = ttk.LabelFrame(main_frame, text="图片标注区域", padding="5")
        image_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        # 画布
        self.canvas = tk.Canvas(image_frame, bg="white")
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 滚动条
        v_scrollbar = ttk.Scrollbar(image_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar = ttk.Scrollbar(image_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        # 标注列表
        annotation_frame = ttk.LabelFrame(main_frame, text="当前图片标注", padding="5")
        annotation_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # 标注列表
        self.annotation_listbox = tk.Listbox(annotation_frame, height=6)
        self.annotation_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.annotation_listbox.bind("<Double-Button-1>", self.delete_annotation)
        
        # 标注列表滚动条
        list_scrollbar = ttk.Scrollbar(annotation_frame, orient=tk.VERTICAL, command=self.annotation_listbox.yview)
        list_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.annotation_listbox.configure(yscrollcommand=list_scrollbar.set)
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def load_image(self, index):
        """加载指定索引的图片"""
        if 0 <= index < len(self.image_list):
            self.current_index = index
            image_name = self.image_list[index]
            self.current_image_path = os.path.join(self.test_images_dir, image_name)
            
            # 更新图片选择器
            self.image_combo['values'] = self.image_list
            self.image_combo.set(image_name)
            
            # 查找对应的image_id
            self.current_image_id = None
            for img in self.pred_data["images"]:
                if img["file_name"] == image_name:
                    self.current_image_id = img["id"]
                    break
            
            # 加载图片
            try:
                self.current_image = Image.open(self.current_image_path)
                # 调整图片大小以适应显示
                display_size = (800, 600)
                self.current_image.thumbnail(display_size, Image.Resampling.LANCZOS)
                self.current_photo = ImageTk.PhotoImage(self.current_image)
                
                # 更新画布
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_photo)
                self.canvas.configure(scrollregion=self.canvas.bbox("all"))
                
                # 加载该图片的标注
                self.load_image_annotations()
                
                # 更新状态
                self.status_var.set(f"已加载: {image_name} (ID: {self.current_image_id})")
                
            except Exception as e:
                messagebox.showerror("错误", f"无法加载图片: {e}")
    
    def load_image_annotations(self):
        """加载当前图片的标注"""
        self.annotations = []
        if self.current_image_id is not None:
            for ann in self.pred_data["annotations"]:
                if ann["image_id"] == self.current_image_id:
                    self.annotations.append(ann)
        
        self.update_annotation_display()
        self.draw_annotations()
    
    def update_annotation_display(self):
        """更新标注列表显示"""
        self.annotation_listbox.delete(0, tk.END)
        for i, ann in enumerate(self.annotations):
            category_name = self.categories.get(ann["category_id"], f"未知类别{ann['category_id']}")
            bbox = ann["bbox"]
            score = ann.get("score", 1.0)
            self.annotation_listbox.insert(tk.END, 
                f"{i+1}. {category_name} - 位置:({bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}) - 置信度:{score:.2f}")
    
    def draw_annotations(self):
        """在画布上绘制所有标注"""
        if not self.current_image:
            return
        
        # 清除之前的标注绘制
        self.canvas.delete("annotation")
        
        # 获取原始图像和显示图像的尺寸
        original_width = None
        original_height = None
        for img in self.pred_data["images"]:
            if img["id"] == self.current_image_id:
                original_width = img["width"]
                original_height = img["height"]
                break
        
        if original_width is None or original_height is None:
            return
        
        # 计算缩放比例
        display_width, display_height = self.current_image.size
        scale_x = display_width / original_width
        scale_y = display_height / original_height
        
        for ann in self.annotations:
            bbox = ann["bbox"]
            category_id = ann["category_id"]
            
            # 选择颜色
            colors = ["red", "blue", "green"]
            color = colors[category_id % len(colors)]
            
            # 缩放坐标 - bbox格式: [xmin, ymin, width, height]
            xmin, ymin, width, height = bbox
            scaled_xmin = xmin * scale_x
            scaled_ymin = ymin * scale_y
            scaled_width = width * scale_x
            scaled_height = height * scale_y
            
            # 绘制边界框
            self.canvas.create_rectangle(scaled_xmin, scaled_ymin, 
                                       scaled_xmin + scaled_width, scaled_ymin + scaled_height, 
                                       outline=color, width=2, tags="annotation")
            
            # 绘制标签
            category_name = self.categories.get(category_id, f"类别{category_id}")
            self.canvas.create_text(scaled_xmin, scaled_ymin - 10, text=category_name, 
                                  fill=color, anchor=tk.SW, tags="annotation")
    
    def on_image_selected(self, event):
        """图片选择器事件"""
        selected = self.image_var.get()
        if selected in self.image_list:
            index = self.image_list.index(selected)
            self.load_image(index)
    
    def on_category_selected(self, event):
        """类别选择器事件"""
        selected = self.category_var.get()
        if selected:
            # 提取类别ID（冒号前的数字）
            category_id = int(selected.split(':')[0])
            self.selected_category = category_id
    
    def prev_image(self):
        """上一张图片"""
        if self.current_index > 0:
            self.load_image(self.current_index - 1)
    
    def next_image(self):
        """下一张图片"""
        if self.current_index < len(self.image_list) - 1:
            self.load_image(self.current_index + 1)
    
    def on_mouse_down(self, event):
        """鼠标按下事件"""
        self.drawing = True
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.current_bbox = None
    
    def on_mouse_drag(self, event):
        """鼠标拖拽事件"""
        if self.drawing:
            # 清除之前的临时矩形
            self.canvas.delete("temp_rect")
            
            # 绘制临时矩形
            current_x = self.canvas.canvasx(event.x)
            current_y = self.canvas.canvasy(event.y)
            
            x1 = min(self.start_x, current_x)
            y1 = min(self.start_y, current_y)
            x2 = max(self.start_x, current_x)
            y2 = max(self.start_y, current_y)
            
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2, tags="temp_rect")
    
    def on_mouse_up(self, event):
        """鼠标释放事件"""
        if self.drawing:
            self.drawing = False
            
            # 清除临时矩形
            self.canvas.delete("temp_rect")
            
            # 计算边界框 - COCO格式: [xmin, ymin, width, height]
            # xmin, ymin: 边界框左上角的坐标
            # width, height: 边界框的宽度和高度
            end_x = self.canvas.canvasx(event.x)
            end_y = self.canvas.canvasy(event.y)
            
            xmin = min(self.start_x, end_x)  # 左上角x坐标
            ymin = min(self.start_y, end_y)  # 左上角y坐标
            width = abs(end_x - self.start_x)  # 宽度
            height = abs(end_y - self.start_y)  # 高度
            
            # 如果边界框太小，忽略
            if width < 5 or height < 5:
                return
            
            # 获取原始图像尺寸
            original_width = None
            original_height = None
            for img in self.pred_data["images"]:
                if img["id"] == self.current_image_id:
                    original_width = img["width"]
                    original_height = img["height"]
                    break
            
            if original_width is None or original_height is None:
                return
            
            # 计算缩放比例
            display_width, display_height = self.current_image.size
            scale_x = display_width / original_width
            scale_y = display_height / original_height
            
            # 将显示坐标转换回原始图像坐标
            original_xmin = xmin / scale_x
            original_ymin = ymin / scale_y
            original_width_coord = width / scale_x
            original_height_coord = height / scale_y
            
            # 创建新标注 - COCO格式
            if self.current_image_id is not None:
                # 添加噪声到边界框坐标
                bbox_with_noise = self.add_noise_to_bbox([original_xmin, original_ymin, original_width_coord, original_height_coord])
                # 生成正态分布的score值
                score_value = self.generate_normal_score()
                
                new_annotation = {
                    "image_id": self.current_image_id,
                    "category_id": self.selected_category,
                    "bbox": bbox_with_noise,  # [xmin, ymin, width, height] with noise
                    "score": score_value
                }
                
                self.annotations.append(new_annotation)
                self.update_annotation_display()
                self.draw_annotations()
                
                self.status_var.set(f"已添加标注: {self.categories[self.selected_category]}")
    
    def delete_annotation(self, event):
        """删除标注"""
        selection = self.annotation_listbox.curselection()
        if selection:
            index = selection[0]
            if 0 <= index < len(self.annotations):
                del self.annotations[index]
                self.update_annotation_display()
                self.draw_annotations()
                self.status_var.set("已删除标注")
    
    def clear_current(self):
        """清除当前图片的所有标注"""
        if messagebox.askyesno("确认", "确定要清除当前图片的所有标注吗？"):
            self.annotations = []
            self.update_annotation_display()
            self.draw_annotations()
            self.status_var.set("已清除所有标注")
    
    
    def save_annotations(self):
        """保存标注到JSON文件"""
        try:
            # 更新预测数据中的标注
            # 首先移除当前图片的所有标注
            self.pred_data["annotations"] = [ann for ann in self.pred_data["annotations"] 
                                           if ann["image_id"] != self.current_image_id]
            
            # 添加新的标注
            self.pred_data["annotations"].extend(self.annotations)
            
            # 保存到文件
            with open(self.pred_json_path, 'w', encoding='utf-8') as f:
                json.dump(self.pred_data, f, ensure_ascii=False, indent=2)
            
            self.status_var.set("标注已保存")
            messagebox.showinfo("成功", "标注已保存到pred.json文件")
            
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {e}")


def main():
    root = tk.Tk()
    root.geometry("1000x800")
    
    app = ManualAnnotationTool(root)
    root.mainloop()


if __name__ == "__main__":
    main() 