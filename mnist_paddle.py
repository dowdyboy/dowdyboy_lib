import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader
import paddle.vision.transforms as transforms

import argparse

from dowdyboy_lib.paddle.trainer import TrainerConfig, Trainer

parser = argparse.ArgumentParser(description='test runtime env')
parser.add_argument('--epoch', type=int, default=1, help='run num epoch')
parser.add_argument('--mix', type=str, default='no', help='fp mix')
parser.add_argument('--batch-size', type=int, default=100, help='batch size')
parser.add_argument('--num-workers', type=int, default=8, help='loader num workers')
parser.add_argument('--save-interval', type=int, default=5, help='save interval')
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
parser.add_argument('--out-dir', type=str, default='./output/', help='out dir')
args = parser.parse_args()

# 多层卷积神经网络实现
class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)
        # 定义池化层，池化核的大小kernel_size为2，池化步长为2
        self.max_pool1 = nn.MaxPool2D(kernel_size=2, stride=2)
        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.conv2 = nn.Conv2D(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2)
        # 定义池化层，池化核的大小kernel_size为2，池化步长为2
        self.max_pool2 = nn.MaxPool2D(kernel_size=2, stride=2)
        # 定义一层全连接层，输出维度是1
        self.fc = nn.Linear(in_features=980, out_features=10)

    # 定义网络前向计算过程，卷积后紧接着使用池化层，最后使用全连接层计算最终输出
    # 卷积层激活函数使用Relu，全连接层不使用激活函数
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc(x)
        return x


def build_data():
    train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transforms.ToTensor())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, train_dataset, val_loader, val_dataset


def build_model():
    model = MNIST()
    return model


def build_optimizer(model):
    lr_scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.01, step_size=1, gamma=0.9, )
    optimizer = paddle.optimizer.SGD(parameters=model.parameters(), learning_rate=lr_scheduler, )
    return optimizer, lr_scheduler


def train_step(trainer: Trainer, bat, bat_idx, global_step):
    [model] = trainer.get_models()
    [optimizer], _ = trainer.get_optimizers()
    [loss_func] = trainer.get_components()
    bat_x, bat_y = bat
    pred_y = model(bat_x)
    loss = loss_func(pred_y, bat_y)
    trainer.set_records({'train_loss': loss.item()})
    trainer.log({'train_loss': loss.item()}, global_step)
    return loss


def val_step(trainer: Trainer, bat, bat_idx, global_step):
    [model] = trainer.get_models()
    [loss_func] = trainer.get_components()
    bat_x, bat_y = bat
    pred_y = model(bat_x)
    loss = loss_func(pred_y, bat_y)
    right_count = paddle.sum(bat_y == paddle.argmax(pred_y, axis=1, keepdim=True)).item()
    total_count = bat_y.shape[0]
    val_acc = right_count / float(total_count)
    trainer.set_records({
        'val_loss': loss.item(),
        'val_acc': val_acc
    })
    trainer.log({'val_loss': loss.item(), 'val_acc': val_acc}, global_step)
    return loss


def on_epoch_end(trainer: Trainer, ep):
    [optimizer], _ = trainer.get_optimizers()
    rec = trainer.get_records()
    # print(rec)
    ep_train_loss = paddle.mean(rec['train_loss']).item()
    ep_val_loss = paddle.mean(rec['val_loss']).item()
    ep_val_acc = paddle.mean(rec['val_acc']).item()
    trainer.print(f'ep_val_acc: {ep_val_acc}, ep_train_loss: {ep_train_loss}, ep_val_loss: {ep_val_loss}, lr: {optimizer.get_lr()}')
    trainer.log({'ep_val_loss': ep_val_loss, 'ep_train_loss': ep_train_loss, 'ep_val_acc': ep_val_acc}, ep)
    return


def main():
    cfg = TrainerConfig(
        epoch=args.epoch,
        mixed_precision=args.mix,
        multi_gpu=True,
        save_best=True,
        save_interval=args.save_interval,
        out_dir=args.out_dir,
    )
    trainer = Trainer(cfg)
    trainer.print(args)

    train_loader, train_dataset, val_loader, val_dataset = build_data()
    trainer.print(f'train data size : {len(train_dataset)}')
    trainer.print(f'val data size : {len(val_dataset)}')

    model = build_model()
    loss_func = nn.CrossEntropyLoss()
    trainer.print(model)

    optimizer, lr_scheduler = build_optimizer(model)
    trainer.print(optimizer)
    trainer.print(lr_scheduler)
    trainer.print(f'cuda version: {paddle.version.cuda()}')

    trainer.set_train_dataloader(train_loader)
    trainer.set_val_dataloader(val_loader)
    trainer.set_model(model)
    trainer.set_component(loss_func)
    trainer.set_optimizer(optimizer, lr_scheduler)

    if args.checkpoint is not None:
        trainer.load_checkpoint(args.checkpoint)

    trainer.fit(
        train_step=train_step,
        val_step=val_step,
        on_epoch_end=on_epoch_end
    )

    return


if __name__ == '__main__':
    main()
    # paddle.distributed.spawn(main)
