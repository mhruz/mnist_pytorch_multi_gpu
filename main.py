# Inspired by:
# https://nextjournal.com/gkoehler/pytorch-mnist
# https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import torch.multiprocessing as mp
import argparse

def train_net(local_rank, global_rank_offset, size, gpu_rank, args):

    rank = local_rank + global_rank_offset

    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    batch_size_train = args.batch_size_train

    random_seed = 1
    torch.set_deterministic(args.determ)
    torch.manual_seed(random_seed)

    # Data loading code
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=size,
        rank=rank
    )

    train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('/files/', train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ])),
            batch_size=batch_size_train, shuffle=True)



if __name__ == "__main__":
    # parse commandline
    parser = argparse.ArgumentParser(description='Runs training and evaluation of MNIST dataset.')
    parser.add_argument('epochs', type=int, help='number of epoch for which to train, default=3', default=3)
    parser.add_argument('batch_size_train', type=int, help='training batch size, default=64', default=64)
    parser.add_argument('batch_size_test', type=int, help='testing batch size, default=1000', default=1000)
    parser.add_argument('number_of_gpus', type=int, help='number of GPUs to use for train/test, default=1', default=1)
    parser.add_argument('--determ', type=bool,
                        help='whether to use deterministic computation, if NO, do not use (--determ False),'
                             ' but omit the parameter', default=False)
    args = parser.parse_args()

    gpus_per_node = int(os.environ['PBS_NGPUS'])
    size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    node_id = int(os.environ['OMPI_COMM_WORLD_RANK'])
    gpu_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])

    print("gpus_per_node: {}".format(gpus_per_node))
    print("size: {}".format(size))
    print("node_id: {}".format(node_id))
    print("gpu_rank: {}".format(gpu_rank))

    # create ONE process -> local_rank is always 0, node_id is the global_rank_offset
    mp.spawn(train_net, args=(node_id, size, gpu_rank, args), nprocs=gpus_per_node)

