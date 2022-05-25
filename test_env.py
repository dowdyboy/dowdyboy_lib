import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg16

import argparse
import random

from dowdyboy_lib.trainer import TrainerConfig, Trainer

parser = argparse.ArgumentParser(description='test runtime env')
parser.add_argument('--epoch', type=int, default=3, help='run num epoch')
parser.add_argument('--mix', type=str, default='no', help='fp mix')
parser.add_argument('--data-size', type=int, default=1000, help='dataset size')
parser.add_argument('--batch-size', type=int, default=2, help='batch size')
parser.add_argument('--num-workers', type=int, default=4, help='loader num workers')
parser.add_argument('--save-interval', type=int, default=1, help='save interval')
args = parser.parse_args()


class TestDataset(Dataset):

    def __init__(self, size):
        super(TestDataset, self).__init__()
        self.size = size

    def __getitem__(self, idx):
        return torch.rand(3, 224, 224), 0 if random.random() < 0.5 else 1

    def __len__(self):
        return self.size


def build_data():
    train_dataset = TestDataset(args.data_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    return train_loader, train_dataset


def build_model():
    model = vgg16(pretrained=False)
    return model


def build_optimizer(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, )
    return optimizer


def train_step(trainer: Trainer, bat, bat_idx, global_step):
    [model] = trainer.get_models()
    [optimizer], _ = trainer.get_optimizers()
    [loss_func] = trainer.get_components()
    bat_x, bat_y = bat
    pred_y = model(bat_x)
    loss = loss_func(pred_y, bat_y)
    return loss


def main():
    cfg = TrainerConfig(
        epoch=args.epoch,
        mixed_precision=args.mix,
        save_interval=args.save_interval,
    )
    trainer = Trainer(cfg)
    trainer.print(args)

    train_loader, train_dataset = build_data()
    trainer.print(f'data size : {len(train_dataset)}')

    model = build_model()
    loss_func = nn.CrossEntropyLoss()
    trainer.print(model)

    optimizer = build_optimizer(model)
    trainer.print(optimizer)
    trainer.print(f'cuda : {torch.cuda.is_available()}')

    trainer.set_train_dataloader(train_loader)
    trainer.set_model(model)
    trainer.set_optimizer(optimizer)
    trainer.set_component(loss_func)

    trainer.fit(
        train_step=train_step
    )

    return


if __name__ == '__main__':
    main()
