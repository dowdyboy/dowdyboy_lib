import paddle
from paddle.io import DataLoader, DistributedBatchSampler
from paddle.vision.models import vgg16
from paddle.optimizer import Adam
from paddle.optimizer.lr import StepDecay


if __name__ == '__main__':
    for x in enumerate([100, 101, 102, 103]):
        print(x)
    model = vgg16()
    lr_schedule = StepDecay(learning_rate=0.01, step_size=10, gamma=0.9)
    optimizer = Adam(learning_rate=lr_schedule, )
    optimizer.clear_grad()
    optimizer.step()
    model.eval()
    lr_schedule.step()
    model.state_dict()
    optimizer.state_dict()
    lr_schedule.state_dict()
    paddle.Tensor().item()
    model.set_state_dict()
    optimizer.set_state_dict()
    lr_schedule.set_state_dict()
    optimizer.get_lr()
    paddle.Tensor()
    DataLoader()
    print()

# import numpy as np
# import paddle
# from paddle.distributed import init_parallel_env
#
# paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
# init_parallel_env()
# tensor_list = []
# if paddle.distributed.ParallelEnv().local_rank == 0:
#     np_data1 = np.array([[4, 5, 6], [4, 5, 6]])
#     np_data2 = np.array([[4, 5, 6], [4, 5, 6]])
#     data1 = paddle.to_tensor(np_data1)
#     data2 = paddle.to_tensor(np_data2)
#     paddle.distributed.all_gather(tensor_list, data1)
# else:
#     np_data1 = np.array([[1, 2, 3], [1, 2, 3]])
#     np_data2 = np.array([[1, 2, 3], [1, 2, 3]])
#     data1 = paddle.to_tensor(np_data1)
#     data2 = paddle.to_tensor(np_data2)
#     paddle.distributed.all_gather(tensor_list, data2)

