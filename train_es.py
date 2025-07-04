import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from tqdm import tqdm
from dataset import CocoDetectionDataset, collate_fn
from model import get_model

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
    return total_loss / len(val_loader)


def train(epochs=10, batch_size=5, patience=10):
    # 参数
    train_img_dir = 'data/train/images'
    train_ann_file = 'data/train/train.json'
    model_save_path = 'model/model.pth'
    num_classes = 4  # 背景类别 + 3个类别
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载完整数据集
    full_dataset = CocoDetectionDataset(
        img_dir=train_img_dir,
        ann_file=train_ann_file,
        transforms=T.ToTensor()
    )
    n = len(full_dataset)
    indices = list(range(n))
    split = int(n * 0.1)  # 前10%做验证
    val_indices = indices[:split]
    train_indices = indices[split:]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = get_model(num_classes)
    model.to(device)
    
    # 使用Adam优化器，学习率1e-4
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-4)

    best_val_loss = float('inf')
    counter = 0

    try:
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for images, targets in pbar:
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    
                    # 前向传播，计算损失函数
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()
                    epoch_loss += losses.item()
                    pbar.set_postfix({'loss': f'{epoch_loss/(pbar.n+1):.4f}'})

            # 验证
            val_loss = evaluate(model, val_loader, device)
            print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_save_path)
                counter = 0
                print("模型提升，已保存。")
            else:
                counter += 1
                print(f"验证集无提升，早停计数：{counter}/{patience}")
                if counter >= patience:
                    print("早停触发，训练终止")
                    break

    except KeyboardInterrupt:
        print('\nTraining was interrupted and the current model is being saved...')
        torch.save(model.state_dict(), model_save_path)
        print('Model saved to model.pth')
        return

    # 保存模型
    torch.save(model.state_dict(), model_save_path)
    print('Model saved to model.pth')