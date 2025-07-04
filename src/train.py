import torch
import random
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from tqdm import tqdm
from src.dataset import CocoDetectionDataset, collate_fn
from src.model import get_model
from sklearn.model_selection import KFold


"""
执行标准训练，不包含验证集划分和早停机制
"""
def train(epochs=10, batch_size=5):
    # 相关参数
    train_img_dir = 'data/train/images'
    train_ann_file = 'data/train/train.json'
    model_save_path = 'model/model.pth'
    num_classes = 4  # 背景类别 + 3个类别
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    train_dataset = CocoDetectionDataset(
        img_dir=train_img_dir,
        ann_file=train_ann_file,
        transforms=T.ToTensor()
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # 获取检测模型
    model = get_model(num_classes)
    model.to(device)
    
    # 使用Adam优化器，学习率1e-4
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-4)
    
    # 开始训练
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
                    optimizer.zero_grad()  # 梯度清零
                    losses.backward()      # 反向传播
                    optimizer.step()       # 优化器更新参数
                    epoch_loss += losses.item()  # 累加loss
                    
                    pbar.set_postfix({'loss': f'{epoch_loss/(pbar.n+1):.4f}'})
    except KeyboardInterrupt:
        print('\nTraining was interrupted and the current model is being saved...')
        torch.save(model.state_dict(), model_save_path)
        print(f'Model saved to {model_save_path}')
        return

    # 保存模型
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')



"""
执行带有验证集划分与早停的训练
"""
def evaluate(model, val_loader, device):
    model.train()  # 保持训练模式以获得loss字典
    total_loss = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
    return total_loss / len(val_loader)


def train_with_validation(epochs=10, batch_size=5, patience=3):
    # 相关参数
    train_img_dir = 'data/train/images'
    train_ann_file = 'data/train/train.json'
    model_save_path = 'model/model_wv.pth'
    num_classes = 4  # 背景类别 + 3个类别
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载完整数据集
    full_dataset = CocoDetectionDataset(
        img_dir=train_img_dir,
        ann_file=train_ann_file,
        transforms=T.ToTensor()
    )
    
    # 取随机10%做验证
    n = len(full_dataset)
    indices = list(range(n))
    random.shuffle(indices)
    split = int(n * 0.1)
    val_indices = indices[:split]
    train_indices = indices[split:]

    # 加载数据
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 获取检测模型
    model = get_model(num_classes)
    model.to(device)
    
    # 使用Adam优化器，学习率1e-4
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-4)

    # 开始训练
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

            # 交叉验证
            val_loss = evaluate(model, val_loader, device)
            print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_save_path)
                counter = 0
                print("Model improved, saved.")
            else:
                counter += 1
                print(f"No improvement on validation set, early stopping counter: {counter}/{patience}")
                if counter >= patience:
                    print("Early stopping triggered, terminating training.")
                    break
    except KeyboardInterrupt:
        print('\nTraining was interrupted and the current model is being saved...')
        torch.save(model.state_dict(), model_save_path)
        print(f'Model saved to {model_save_path}')
        return

    # 保存模型
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')


"""
K折交叉验证训练，输出每个fold的最佳验证损失和平均损失
"""
def train_with_kfold(k=5, epochs=10, batch_size=5, patience=3):
    train_img_dir = 'data/train/images'
    train_ann_file = 'data/train/train.json'
    num_classes = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载完整数据集
    full_dataset = CocoDetectionDataset(
        img_dir=train_img_dir,
        ann_file=train_ann_file,
        transforms=T.ToTensor()
    )
    n = len(full_dataset)
    indices = list(range(n))
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    val_losses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n===== Fold {fold+1}/{k} =====")
        model_save_path = f"model/model_kfold_fold{fold+1}.pth"
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        model = get_model(num_classes)
        model.to(device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=1e-4)

        best_val_loss = float('inf')
        counter = 0
        try:
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0
                with tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{epochs}") as pbar:
                    for images, targets in pbar:
                        images = [img.to(device) for img in images]
                        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                        loss_dict = model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())
                        optimizer.zero_grad()
                        losses.backward()
                        optimizer.step()
                        epoch_loss += losses.item()
                        pbar.set_postfix({'loss': f'{epoch_loss/(pbar.n+1):.4f}'})
                val_loss = evaluate(model, val_loader, device)
                print(f"Fold {fold+1} Epoch {epoch+1}, Val Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), model_save_path)
                    counter = 0
                    print(f"Model improved, saved to {model_save_path}.")
                else:
                    counter += 1
                    print(f"No improvement on validation set, early stopping counter: {counter}/{patience}")
                    if counter >= patience:
                        print("Early stopping triggered.")
                        break
        except KeyboardInterrupt:
            print(f'\nTraining was interrupted at Fold {fold+1}, saving current model...')
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved to {model_save_path}')
            val_losses.append(best_val_loss)
            break
        
        val_losses.append(best_val_loss)
        print(f"Best Val Loss for Fold {fold+1}: {best_val_loss:.4f}")

    print(f"\nK-Fold Cross Validation Results: {val_losses}")
    print(f"Average Val Loss: {sum(val_losses)/len(val_losses):.4f}")


def main(train_mode='kfold', epochs=10, batch_size=5, patience=3, k=5):
    if train_mode == 'normal':
        train(epochs=epochs, batch_size=batch_size)
    elif train_mode == 'kfold':
        train_with_kfold(k=k, epochs=epochs, batch_size=batch_size, patience=patience)
    else:
        train_with_validation(epochs=epochs, batch_size=batch_size, patience=patience)


if __name__ == '__main__':
    main()