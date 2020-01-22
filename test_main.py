import termplotlib as tpl
import math
import os
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from gradual_warmup_lr_scheduler import GradualWarmupScheduler
import sys
import  matplotlib as plt

sys.path.append('.')


def check_annealing(model, optimizer, param_dict):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=param_dict['t_max'], eta_min=param_dict['eta_min'], last_epoch=-1)
    # lr_list = [0. for i in range(param_dict['epochs']) for j in range(param_dict['steps'])]
    lr_list = []
    for epoch in range(param_dict['epochs']):
        for idx in range(param_dict['steps']):
            now_itr = epoch * param_dict['steps'] + idx
            now_lr = scheduler.get_lr()
            # lr_list[epoch * param_dict['steps'] + idx] = now_lr
            lr_list.append(now_lr[0])
            optimizer.step()
            scheduler.step()
            if optimizer.param_groups[0]['lr'] == param_dict['eta_min']:
                if param_dict['whole_decay']:
                    annealed_lr = param_dict['lr'] * (1 + math.cos(
                        math.pi * now_itr / (param_dict['epochs'] * param_dict['steps']))) / 2
                    optimizer.param_groups[0]['initial_lr'] = annealed_lr
                param_dict['t_max'] *= param_dict['t_mult']
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=param_dict['t_max'], eta_min=param_dict['eta_min'], last_epoch=-1)
    return lr_list


# def show_graph(lr_lists, epochs, steps, out_name=None):
#     fig = tpl.figure()
#     # fig.plot(x, y, width=60, height=20)
#     x = list(range(epochs * steps))
#     fig.plot(x, lr_lists, width=200, height=40)
#     fig.show()


def show_graph(lr_lists, epochs, steps, out_name='test'):
    import matplotlib.pyplot as plt
    plt.clf()
    plt.rcParams['figure.figsize'] = [20, 5]
    x = list(range(epochs * steps))
    plt.plot(x, lr_lists, label="line L")
    plt.plot()
    plt.ylim(10e-5, 1)
    plt.yscale("log")
    plt.xlabel("iterations")
    plt.ylabel("learning rate")
    plt.title("Check Cosine Annealing Learing Rate with {}".format(out_name))
    plt.legend()
    plt.show()


def test_scheduler():
    max_epoch = 100
    max_step = 5005
    v = torch.zeros(10)
    multiplier = 1024
    base_lr = 0.1 / multiplier
    optimizer = torch.optim.SGD([v], lr=base_lr)
    # torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)
    # When last_epoch=-1, sets initial lr as lr.
    # optimizer (Optimizer) – Wrapped optimizer.
    # T_max (int) – Maximum number of iterations.
    # eta_min (float) – Minimum learning rate. Default: 0.
    # last_epoch (int) – The index of last epoch. Default: -1.
    # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epoch * max_step, 0.0001)

    # optimizer (Optimizer): Wrapped optimizer.
    # T_0 (int): Number of iterations for the first restart.
    # T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
    # eta_min (float, optional): Minimum learning rate. Default: 0.
    # last_epoch (int, optional): The index of last epoch. Default: -1.
    scheduler_cosine_restart = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 30 * max_step, 2)

    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=multiplier, total_epoch=10 * max_step, after_scheduler=scheduler_cosine_restart)

    lr_list = []

    for epoch in range(0, max_epoch):
        # scheduler.step(epoch)
        # optim.step()
        # print(epoch, optim.param_groups[0]['lr'], scheduler.get_lr())

        for step in range(0, max_step):
            optimizer.step()
            # curr_epoch = epoch + float(step) / max_step
            scheduler.step()
            lr_list.append(optimizer.param_groups[0]['lr'])
            # print("-", scheduler.last_epoch, scheduler.get_lr()[0], optimizer.param_groups[0]['lr'])
            # print(epoch, optimizer.param_groups[0]['lr'], scheduler.get_lr())
    show_graph(lr_list, max_epoch, max_step)
    # show_graph_plt(lr_list, max_epoch, max_step)


def test_wc_scheduler():

    epochs = 90
    steps = 5005
    lr = 0.1

    t01_tmult2 = {
        'epochs': epochs,
        'steps': steps,
        't_max': steps * 1,
        't_mult': 2,
        'eta_min': 0,
        'lr': lr,
        'whole_decay': False,
        'out_name': "T_0={}-T_mult={}".format(steps * 1, 2),
    }

    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    t01_tmult2_out = check_annealing(model, optimizer, t01_tmult2)

    show_graph(t01_tmult2_out, epochs, steps, t01_tmult2['out_name'])


# # Cosine Annealing with Warm up for PyTorch

# ## Example
# ```
# >> model = ...
# >> optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5) # lr is min lr
# >> scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=250, T_mult=2, eta_max=0.1, T_up=50)
# >> for epoch in range(n_epoch):
# >>     train()
# >>     valid()
# >>     scheduler.step()
# ```

# CosineAnnealingWarmUpRestarts(optimizer, T_0=150, T_mult=1, eta_max=0.1,  T_up=10, gamma=0.5)
# CosineAnnealingWarmUpRestarts(optimizer, T_0=50, T_mult=2, eta_max=0.1,  T_up=10, gamma=0.5)
# CosineAnnealingWarmUpRestarts(optimizer, T_0=100, T_mult=1, eta_max=0.1,  T_up=10, gamma=0.5)
# CosineAnnealingWarmUpRestarts(optimizer, T_0=250, T_mult=1, eta_max=0.1, T_up=50)
# CosineAnnealingWarmUpRestarts(optimizer, T_0=250, T_mult=2, eta_max=0.1, T_up=50)

def main():
    test_scheduler()
    test_wc_scheduler()


if __name__ == '__main__':
    main()
