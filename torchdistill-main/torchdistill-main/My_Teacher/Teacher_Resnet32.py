import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchdistill.common.main_util import is_main_process, save_ckpt
from torchvision.datasets import ImageFolder
from torchdistill.misc.log import setup_log_file, SmoothedValue, MetricLogger
from torchvision import transforms
from torchvision import models
from torch.optim.lr_scheduler import MultiStepLR
from My_Dataset import trian_dataset, My_save_ckpt
from torchdistill.optim.registry import get_optimizer, get_scheduler
from torchdistill.common.constant import def_logger
import pandas as pd
import numpy as np
import torch, gc
import time
from torchdistill.eval.classification import compute_accuracy
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

# 设置训练的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 初始化日志
logger = def_logger.getChild(__name__)


# 创建自己的数据集

def setup_data_my_loaders():
    data_train = trian_dataset(data_dir='train')
    train_iter = DataLoader(dataset=data_train, batch_size=128, shuffle=True, pin_memory=True)
    data_valid = trian_dataset(data_dir='valid')
    valid_iter = DataLoader(dataset=data_valid, batch_size=128, shuffle=True, pin_memory=True)

    return train_iter, valid_iter


# 加载数据集
train_iter, valid_iter = setup_data_my_loaders()
# 加载模型
model = models.resnet34(pretrained=False)
# 修改模型结构
model.fc = torch.nn.Linear(512, 8)
# 优化器选择
lr = 0.001
# 优化器选择，给定网络的所有参数然后调整学习率以及权重衰减等参数
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# 学习率衰减
params = {'milestones': [3], 'gamma': 0.1}
lr_scheduler = get_scheduler(optimizer=optimizer, scheduler_type='MultiStepLR', param_dict=params)

# 模型放在多个GPU上面
# 将模型放到多个GPU上
# device_ids = [0, 1, 2, 3]
# model = nn.DataParallel(model, device_ids=device_ids)
# Dataset中已默认有了 Resize((224, 224)) + ToTensor()，再设RandomHorizontalFlip和预训练模型用的normalize


# 数据增强
train_augs = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_augs = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def evaluate(model, data_loader, device, log_freq=1000, title=None, header='Test:'):
    model.to(device)

    if title is not None:
        logger.info(title)
    model.eval()
    metric_logger = MetricLogger(delimiter='  ')
    for image, target in metric_logger.log_every(data_loader, log_freq, header):
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(image)
        acc1, acc5 = compute_accuracy(output, target, topk=(1, 5))
        # FIXME need to take into account that the datasets
        # could have been padded in distributed setup
        batch_size = image.shape[0]
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    top1_accuracy = metric_logger.acc1.global_avg
    top5_accuracy = metric_logger.acc5.global_avg
    logger.info(' * Acc@1 {:.4f}\tAcc@5 {:.4f}\n'.format(top1_accuracy, top5_accuracy))
    return metric_logger.acc1.global_avg


# 训练
def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    ckpt_file_path = './teacher_Resnet34.pt'
    best_val_top1_accuracy = 0.0
    net = net.to(device)
    print('training on ', device)
    batch_count = 0
    gc.collect()
    torch.cuda.empty_cache()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            # 训练时使用数据增强
            X = train_augs(X)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        val_top1_accuracy = evaluate(model, valid_iter, device, log_freq=1000, header='Validation:')
        if val_top1_accuracy > best_val_top1_accuracy and is_main_process():
            logger.info('Best top-1 accuracy: {:.4f} -> {:.4f}'.format(best_val_top1_accuracy, val_top1_accuracy))
            logger.info('Updating ckpt at {}'.format(ckpt_file_path))
            best_val_top1_accuracy = val_top1_accuracy
            My_save_ckpt(model, optimizer, lr_scheduler, best_val_top1_accuracy, ckpt_file_path)
        print('epoch %d, loss %.4f, train acc %.3f,  time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - start))


def train_fine_tuning(net, optimizer, train_iter, valid_iter, num_epochs=100):
    loss = torch.nn.CrossEntropyLoss()
    train(train_iter, valid_iter, net, loss, optimizer, device, num_epochs)


train_fine_tuning(model, optimizer, train_iter=train_iter, valid_iter=valid_iter)
path = 'Resnet34.pt'
torch.save(model, path)
