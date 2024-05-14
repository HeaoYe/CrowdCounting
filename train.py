from torch.autograd import Variable
from torchvision import transforms
from CSRNet import CSRNet
import torch.utils.data
from tqdm import tqdm
from torch import nn
import warnings
import dataset
import shutil
import torch
import time
import os


# 初始化设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.device(device)
# 初始化变量
works = 4
pin_memory = True
start_epoch = 0
epochs = 50
batch_size = 1
lr = 1e-6
momentum = 0.95
decay = 5 * 1e-4
best_mae = 1e8


def main(checkpoint=None):
    global start_epoch, best_mae

    # 初始化模型
    model = CSRNet(init_weights=True, load_vgg=True).to(device)
    criterion = nn.MSELoss(size_average=False).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=decay)

    # 加载checkpoint
    if checkpoint is not None:
        checkpoint_f = checkpoint
        path = os.path.join('pth_save', checkpoint_f)
        if os.path.isfile(path):
            print(f'loading checkpoint {checkpoint_f}...')
            checkpoint = torch.load(path)
            start_epoch = checkpoint['epoch']
            best_mae = checkpoint['best_mae']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f'loaded checkpoint {checkpoint_f} (epoch {start_epoch})')
        else:
            print(f'no checkpoint named {checkpoint_f}')
    # 开始训练
    for epoch in range(start_epoch, epochs):
        # 训练
        train(epoch, model, criterion, optimizer)
        # 计算mae
        epoch_mae = accuracy(epoch, model)
        # 输出mae信息
        is_best = False
        if epoch_mae < best_mae:
            best_mae = epoch_mae
            is_best = True
        print(f'  Epoch {epoch} MAE: {epoch_mae}')
        print(f'  Best MAE: {best_mae}')
        # 保存checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'best_mae': best_mae,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best)
        print('save checkpoint done')


def train(epoch, model, criterion, optimizer):
    """训练一个epoch"""
    print(f'Epoch {epoch} begins...')
    time.sleep(0.1)
    # 初始化训练集
    train_loader = torch.utils.data.DataLoader(
        dataset.ShanghaiTech(
            train=True, shuffle=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        ), num_workers=works, pin_memory=pin_memory, batch_size=batch_size
    )
    train_bar = tqdm(train_loader, desc='training loss: ')
    model.train()
    for i, info in enumerate(train_bar):
        img, target = info
        # 初始化数据
        img = Variable(img.to(device))
        target = Variable(target.type(torch.FloatTensor).unsqueeze(0).to(device))
        # 更新梯度
        output = model(img)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 输出训练信息
        train_bar.desc = f'training loss: {int(loss*100/batch_size)/100}  '


def accuracy(epoch, model):
    """测试平均mae"""
    print(f'Epoch {epoch} test begins...')
    time.sleep(0.1)
    # 初始化测试集
    test_loader = torch.utils.data.DataLoader(
        dataset.ShanghaiTech(
            train=False, shuffle=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        ), num_workers=works, pin_memory=pin_memory, batch_size=batch_size
    )
    test_bar = tqdm(test_loader, desc='testing mae_total: ')
    model.eval()
    mae = 0
    with torch.no_grad():
        for i, info in enumerate(test_bar):
            img, target = info
            # 初始化数据
            img = Variable(img.to(device))
            # 计算mae
            output = model(img)
            mae += abs(torch.sum(output) - torch.sum(target))
            # 输出测试信息
            test_bar.desc = f"testing mae_total: {int(mae*100)/100}  "
    mae = (mae / batch_size) / len(test_bar)
    return mae


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    """保存"""
    filename = os.path.join('pth_save', filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'pth_save/model_best.pth')


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pre = None
    # pre = 'checkpoint.pth'
    # pre = 'model_best.pth'
    main(checkpoint=pre)
