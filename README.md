![image](http://www.dowdyboy.com/wp-content/uploads/2018/11/659583085462226804.png)

# “dowdyboy”深度学习开发库

一个简单好用的深度学习开发库，提供了一个简单的深度学习框架，
可以快速开发深度学习算法。

包括一些常用API的封装，以及基于Pytorch和paddlepaddle的训练器。

通过这个库，你可以轻松组织深度学习代码，并且方便的运行在CPU/单GPU/多GPU上。

## 特点
- 接口精简，使用方便
- 提供一套简洁的深度学习代码范式
- 分别为Pytorch和paddlepaddle提供了训练器
- 一套代码可以同时运行在CPU/单GPU/多GPU上

## 模块结构

### Pytorch部分
- args：解析脚本参数，支持json文件和yaml文件
- assist：辅助函数，如计算输出特征图大小
- dataset_util：数据集的工具函数
- log：文本日志记录工具
- model_util：模型的工具函数，如冻结层、计算模型参数量、算力
- rand：随机数工具函数，如计算随机数种子、轮盘赌算法
- trainer：训练器

### paddlepaddle部分
- paddle.log：文本日志记录工具
- paddle.trainer：训练器

## 训练器

训练器是该库的核心，它通过接口定义规定和一套深度学习编程范式，
使得依据该训练器所编写的代码能够方便的运行在CPU/单GPU/多GPU上。

同时，它负责维护训练周期、维护进度条、自动保存模型检查点、
自动执行优化器和学习率更新器等繁琐的工作。
并且，它还能一键切换到半精度训练模式。

Pytorch版本和paddlepaddle版本的训练器大部分接口是一致的。
下面以Pytorch版本的训练器为例，说明训练器的类和接口。

使用案例可见项目中的test脚本。

### TrainerConfig

训练器配置类，用于描述训练器的各种参数。

- name：一个名字，工程名称，随意
- epoch：要训练多少epoch
- out_dir：工程输出目录，记录日志、模型等
- mixed_precision：是否使用半精度模式，可以为\['no', 'fp16', 'bf16']
- cpu：是否强制使用CPU训练
- log_with：采用的日志记录器，默认就行，用tensorboard
- enable_save_checkpoint：是否保存模型检查点
- save_interval：多少个epoch保存一次模型检查点
- save_best：是否保存最好的模型检查点
- save_best_type：怎么计算最小，可以是\['min', 'max']
- save_best_rec：通过哪个记录保存最好模型
- save_last：是否保存最后一次的模型检查点
- seed：随机数种子
- auto_optimize：是否开启自动优化器调用，简单模型默认开启就行了
- auto_schedule：是否开启自动学习率调度
- auto_free：是否每个epoch自动释放显存
- find_unused_parameters：如果模型中有没有用到的可学习参数，就要设置为True

### Trainer

创建trainer只需要传递一个TrainerConfig类型的参数即可。

#### set_save_best_calc_func(self, func)

设置计算最优模型的函数，func是一个函数，
签名为 `func(trainer: Trainer) -> best_rec`；
默认计算方式是求均值。

#### set_train_dataloader(self, train_loader)

设置训练数据加载器，train_loader是一个DataLoader类型的参数。

#### set_val_dataloader(self, val_loader)

设置验证数据加载器，val_loader是一个DataLoader类型的参数。

#### set_test_dataloader(self, test_loader)

设置测试数据加载器，test_loader是一个DataLoader类型的参数。

#### set_model(self, model)

设置待训练模型，model是一个Module类型的参数。

#### set_models(self, model_list: list)

设置待训练模型列表，model_list是一个list类型的参数。

#### get_models(self)

获取待训练模型列表。用于正向计算。

#### get_raw_models(self)

获取原始的待训练模型列表，去除了分布式的外包。

#### set_component(self, component)

设置组件，组件是指没有可训练参数的Module，如损失函数。

#### set_components(self, component_list)

设置组件列表，component_list是一个list类型的参数。

#### get_components(self)

获取组件列表。用于正向计算。

#### set_optimizer(self, optimizer, lr_scheduler=None)

设置优化器，optimizer是一个Optimizer类型的参数，
lr_scheduler是一个LRScheduler类型的参数。

#### set_optimizers(self, optimizer_list: list, lr_scheduler_list=None)

设置优化器列表，optimizer_list是一个list类型的参数，
lr_scheduler_list是一个list类型的参数。

#### get_optimizers(self)

获取优化器列表。在auto_optimize为False的时候，自己进行参数更新。

#### backward(self, loss)

计算反向传播，loss是一个Tensor类型的参数。

#### zero_grad(self, optimizer)

清空梯度，optimizer是get_optimizers获取到的。

#### step(self, optimizer=None, lr_scheduler=None)

更新参数，optimizer、lr_scheduler是get_optimizers获取到的。

#### device(self)

获取当前设备

#### is_local_main_process(self)

判断是否是本地主进程，用于多卡训练判断主进程。

#### print(self, txt)

打印文本日志，txt是一个str类型的参数。

#### log(self, value_dict, step)

记录标量日志，如loss等。value_dict是一个dict类型的参数，
键是标量名称，值是标量值，一般为float。

#### set_records(self, value_dict)

调用记录器进行数据记录，可以记录训练过程中的数据。
这些数据可以在每个epoch结束用于计算best_rec，或者进行日志记录等。
不同的进程设置的数据会被自动合并成一个tensor。
value_dict是一个dict类型的参数，键是记录名称，值是记录值，一般为float。

#### get_records(self)

获取记录器记录的数据。在每个epoch结束，
获取到的是所有进程自动合并的tensor。

#### set_bar_state(self, state_dict)

设置进度条显示的内容，state_dict是一个dict类型的参数。
如可以显示acc、iou等，默认只显示loss。

#### fit(self, train_step, val_step=None, on_epoch_end=None)

开始训练，
train_step是一个钩子函数，
`train_step(trainer: Trainer, bat, bat_idx, global_step) -> loss`
是每一步训练的函数，简单情况返回一个loss就行，如果auto_optimize为True；
如果复杂模型，auto_optimize为False，则需要内部自己调用优化器。

val_step是一个钩子函数，
`val_step(trainer: Trainer, bat, bat_idx, global_step) -> loss`
是每一步验证的函数，返回一个loss。

on_epoch_end是一个钩子函数，
`on_epoch_end(trainer: Trainer, ep) -> None`
是每一个epoch结束的函数，可以用来日志记录等。

#### test(self, test_step, on_test_end=None)

开始测试，
test_step是一个钩子函数，
`test_step(trainer: Trainer, bat, bat_idx, global_step) -> None`
是每一步测试的函数，测试完事可以输出结果。
on_test_end是一个钩子函数，
`on_test_end(trainer: Trainer) -> None`
是测试结束的函数，可以用来日志记录等。

#### load_checkpoint(self, checkpoint_dir)

加载模型，checkpoint_dir是模型保存的目录。

