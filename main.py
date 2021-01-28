# Inspired by:
# https://nextjournal.com/gkoehler/pytorch-mnist
# https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html

import sys
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def train_net(rank, size, args):
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    batch_size_train = args.batch_size_train

    random_seed = 1
    torch.set_deterministic(args.determ)
    torch.manual_seed(random_seed)

    dist.init_process_group('nccl', rank=rank, world_size=size)

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
        dataset=train_dataset,
        batch_size=batch_size_train,
        shuffle=args.determ,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler)

    # create the network
    model = DDP(Net().to(rank), device_ids=[rank])
    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    # choose criterion
    loss_fn = nn.NLLLoss()
    loss = None

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

    dist.barrier()

    if args.output is not None:
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss.item()
        }, args.output)


if __name__ == "__main__":
    # parse commandline
    parser = argparse.ArgumentParser(description='Runs training and evaluation of MNIST dataset.')
    parser.add_argument('--epochs', type=int, help='number of epoch for which to train, default=3', default=3)
    parser.add_argument('--batch_size_train', type=int, help='training batch size, default=64', default=64)
    parser.add_argument('--batch_size_test', type=int, help='testing batch size, default=1000', default=1000)
    parser.add_argument('--number_of_gpus', type=int, help='number of GPUs to use for train/test')
    parser.add_argument('--output', type=str,
                        help='optional path to the output model, by default the model is not saved')
    parser.add_argument('--determ', type=bool,
                        help='whether to use deterministic computation, if NO, do not use (--determ False),'
                             ' but omit the parameter', default=False)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA device not present! Exiting...")
        sys.exit()

    gpus_on_node = torch.cuda.device_count()

    print("gpus_on_node: {}".format(gpus_on_node))

    if args.number_of_gpus is not None:
        gpus_on_node = args.number_of_gpus

    # create ONE process -> local_rank is always 0, node_id is the global_rank_offset
    mp.spawn(train_net, args=(gpus_on_node, args), nprocs=gpus_on_node)
