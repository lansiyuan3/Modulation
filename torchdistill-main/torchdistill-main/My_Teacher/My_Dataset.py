from types import BuiltinFunctionType, BuiltinMethodType, FunctionType
import os
import pandas as pd
import numpy as np
from torchdistill.common.file_util import make_parent_dirs
from torchdistill.common.main_util import save_on_master
from torchdistill.common.module_util import check_if_wrapped
from torchdistill.optim.registry import SCHEDULER_DICT
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision
from torch import nn

from torchdistill.common import misc_util
DATASET_DICT = dict()
DATASET_DICT.update(torchvision.datasets.__dict__)
from torch.utils.data import Dataset

def register_dataset(cls_or_func):
    DATASET_DICT[cls_or_func.__name__] = cls_or_func
    return cls_or_func

# 创建自己的数据集对象_cifar10
class dataset(Dataset):
    def __init__(self, imgs, labels, transform=None):
        self.imgs = imgs
        self.labels = labels

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])  # 如果设定transform，采取默认的转换操作

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        label = self.labels[item]
        data_in = self.imgs[item][0]  # 届时传入一个ImageFolder对象，需要取[0]获取图片数据数据，不要标签
        data = self.transform(data_in)
        return data, label

# 自己制作，读取训练集调制后的信号类别
def trian_dataset(data_dir):
    # 读取训练集图片
    if data_dir == 'train':
        data_images = ImageFolder(os.path.join(r'D:\Demo-Copy\Modulation', data_dir))
        # 加载训练数据集csv文件
        train_csv = pd.read_csv(r'D:\Demo-Copy\Modulation\trainLabels.csv')
        # 获取某个元素的索引的方法
        # 这个class_to_num即作为类别号到类别名称的映射，获取类别名称唯一的
        class_to_num = train_csv.label.unique()
        train_csv['class_num'] = train_csv['label'].apply(lambda x: np.where(class_to_num == x)[0][0])
        labels = train_csv.class_num
        dataset_train = dataset(imgs=data_images, labels=labels)
        return dataset_train
    else:
        data_images = ImageFolder(os.path.join(r'D:\Demo-Copy\Modulation', data_dir))
        # 加载验证集数据集csv文件
        valid_csv = pd.read_csv(r'D:\Demo-Copy\Modulation\validLabels.csv')
        # 获取某个元素的索引的方法
        # 这个class_to_num即作为类别号到类别名称的映射，获取类别名称唯一的
        class_to_num = valid_csv.label.unique()
        valid_csv['class_num'] = valid_csv['label'].apply(lambda x: np.where(class_to_num == x)[0][0])
        labels = valid_csv.class_num
        dataset_valid = dataset(imgs=data_images, labels=labels)
        return dataset_valid

def My_save_ckpt(model, optimizer, lr_scheduler, best_value, output_file_path):
    make_parent_dirs(output_file_path)
    model_state_dict = model.module.state_dict() if check_if_wrapped(model) else model.state_dict()
    lr_scheduler_state_dict = lr_scheduler.state_dict() if lr_scheduler is not None else None
    save_on_master({'model': model_state_dict, 'optimizer': optimizer.state_dict(), 'best_value': best_value,
                    'lr_scheduler': lr_scheduler_state_dict}, output_file_path)
