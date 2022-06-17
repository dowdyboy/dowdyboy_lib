import paddle
from paddle.io import DataLoader, DistributedBatchSampler, Dataset
from paddle.vision.models import vgg16
from paddle.optimizer import Adam
from paddle.optimizer.lr import StepDecay


class DemoDataSet(Dataset):

    def __init__(self):
        super(DemoDataSet, self).__init__()

    def __getitem__(self, item):
        return paddle.randn([28, 28]), '/data/a.img'

    def __len__(self):
        return 100


if __name__ == '__main__':
    ld = DataLoader(DemoDataSet(), batch_size=4, shuffle=True)
    for d in ld:
        print(d)
        break
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

