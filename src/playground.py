#     ___   _     __               __
#    / _ | (_)___/ /  ___ ___ ____/ /
#   / __ |/ / __/ _ \/ -_) _ `/ _  /
#  /_/ |_/_/_/ /_//_/\__/\_,_/\_,_/
#
import pandas as pd
import torch
from torch.nn.modules import LeakyReLU
import matplotlib.pyplot as plt
import numpy as np
from src.utils import KUL_PAL
import seaborn as sns
import src.utils.helper as hlp
from os.path import join
import matplotlib as mpl
import math

hlp.hi('Analysis', log_dir='../../logs_cv')
vis_dir = join(hlp.DATA_DIR, '../visuals')

if __name__ == '__main__':
    activation = LeakyReLU()
    colors = sns.color_palette(KUL_PAL)
    sns.palplot(colors)
    plt.show()


    # DASHBOARD
    input, output = [], []
    boundary = 5
    start, stop = -boundary, boundary
    size = 14

    for i in np.arange(start, stop, .01):
        x = torch.tensor(i)
        y = activation(x)
        input.append(float(x))
        output.append(float(y))

    fig, ax = plt.subplots(1,1,figsize=(6,6),dpi=300)
    sns.set_theme(context='paper',font_scale=3)
    ax.plot(input,output,color='#DD8A2E',linewidth=2)
    ax.set_xlim(start, stop)
    ax.set_ylim(start, stop)
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.set_xlabel('Input',fontdict={'size':size, 'weight':'bold'})
    ax.set_ylabel('Activation',fontdict={'size':size, 'weight':'bold'})
    sns.despine(left=True, bottom=True)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.savefig(join(vis_dir, 'leakyrelu.pdf'),bbox_inches='tight', pad_inches=0)
    plt.show()

    ###
    # Plot
    ###
    mpl.rcParams.update(mpl.rcParamsDefault)
    fig, ax = plt.subplots(1,1,figsize=(12,6),dpi=300)
    scheduler = pd.read_csv('../data/misc/scheduler.csv', sep=';')
    sns.set_theme(context='paper',font_scale=3)
    ax.plot(scheduler.Value, color='#DD8A2E',linewidth=2)
    ax.set_xlabel('Step',fontdict={'size':size, 'weight':'bold'})
    ax.set_ylabel('Learning rate',fontdict={'size':size, 'weight':'bold'})

    plt.show()

    ###########
    #
    ##########

    import torch
    from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
    from torchvision.models import resnet18
    import matplotlib.pyplot as plt

    '''optimizer = optim.AdamW,
    optimizer_params = {'lr': 1e-4, 'weight_decay': 1e-5},

    # Learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts,
    scheduler_config = {'interval': 'epoch'},
    scheduler_params = {'T_0': 50, 'eta_min': 3e-5},'''

    #
    model = resnet18(pretrained=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    mode = 'cosineAnnWarm'
    if mode == 'cosineAnn':
        scheduler = CosineAnnealingLR(optimizer, T_0=50, eta_min=3e-5)
    elif mode == 'cosineAnnWarm':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, eta_min=3e-5)
        '''
        Take T_0=5, T_mult=1 as an example:
             T_0: The learning rate returns to the epoch position of the initial value for the first time.
             T_mult: This controls the speed at which the learning rate rebounds
                     -If T_mult=1, the learning rate returns to the maximum at T_0,2*T_0,3*T_0,...,i*T_0,...(initial learning rate)
                             -Return to the maximum at 5, 10, 15, 20, 25,...
                     -If T_mult>1, the learning rate is at T_0,(1+T_mult)*T_0,(1+T_mult+T_mult**2)*T_0,...,(1+T_mult+T_mult**2+. ..+T_0**i)*T0, return to the maximum value
                             -Return to the maximum at 5,15,35,75,155,...
        example:
            T_0=5, T_mult=1
        '''
    plt.figure()
    max_epoch = 500
    iters = 1
    cur_lr_list = []
    for epoch in range(max_epoch):
        print('epoch_{}'.format(epoch))
        for batch in range(iters):
            scheduler.step(epoch + batch / iters)
            optimizer.step()
            # scheduler.step()
            cur_lr = optimizer.param_groups[-1]['lr']
            cur_lr_list.append(cur_lr)
            print('cur_lr:', cur_lr)
        print('epoch_{}_end'.format(epoch))
    x_list = list(range(len(cur_lr_list)))

    mpl.rcParams.update(mpl.rcParamsDefault)
    size=25
    sns.set_theme(context='paper',style='white', font_scale=2)
    fig, ax = plt.subplots(1, 1, figsize=(16, 4), dpi=300)
    ax.plot(x_list, cur_lr_list, color='#DD8A2E', linewidth=2)
    ax.set_xlabel('Epochs', fontdict={'size': size, 'weight': 'bold'})
    ax.set_ylabel('Learning rate', fontdict={'size': size, 'weight': 'bold'})
    sns.despine(left=True, bottom=True)
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.tight_layout()
    plt.savefig(join(vis_dir, 'scheduler.pdf'), bbox_inches='tight', pad_inches=0)
    plt.show()


    ###########
    # AlexNet #
    ###########
    sns.set_theme(context='paper', style='white', font_scale=1)
    # reporting the best results per-team (and top 10)
    data = {'2011':
                [0.25770, 0.31010, 0.35960, 0.50450],
            '2012':
                [0.26172, 0.26979, 0.27058, 0.29576, 0.33419, 0.34464],
            '2013':
                [0.11197, 0.12953, 0.13511, 0.13555, 0.13748, 0.13985, 0.14182,
                 0.14291, 0.15193, 0.15245],
            '2014':
                [0.06656, 0.07325, 0.0806, 0.08111, 0.09508, 0.09794, 0.10222, 0.11229, 0.11326, 0.12376],
            '2015':
                [0.03567, 0.03581, 0.04581, 0.04873, 0.05034, 0.05477, 0.05858, 0.06314, 0.06482, 0.06828],
            '2016':
                [0.02991, 0.03031, 0.03042, 0.03171, 0.03256, 0.03291, 0.03297, 0.03351, 0.03352, 0.03416]
            }

    # image net human top 5 error rate
    human = 5.1 / 100

    points = []
    for k, v in data.items():
        for x in v:
            points.append((k, x))
    x, y = zip(*points)

    plt.figure(figsize=(8,8),dpi=300)
    #plt.title('ImageNet competition results', fontsize=22)
    plt.xlabel('Year', fontsize=20,weight='bold')
    plt.ylabel('Error rate', fontsize=20, weight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.scatter(x, y, marker='o', facecolors='none', edgecolors='C0', lw=2, s=80, label='Competing systems')
    plt.scatter('2012',0.15315, marker='o', facecolors=colors[0],lw=2, s=80, label='AlexNet')
    plt.plot(data.keys(), [human for _ in range(len(data))], '--', color='grey', lw=2, label='Human performance')
    plt.legend(fontsize=16)
    sns.despine(left=True,bottom=True)
    plt.savefig(join(vis_dir, 'imagenet_comp.pdf'), bbox_inches='tight', pad_inches=0)
    plt.show()
    #plt.savefig('imagenet-history.svg')

    ###########################
    # Vanishing gradient plot #
    ###########################

    def sigmoid(x):
        a = []
        for item in x:
            a.append(1 / (1 + math.exp(-item)))
        return a

    x = np.arange(-10., 10., 0.2)
    sig = sigmoid(x)
    dsig = [sig_val*(1-sig_val) for sig_val in sig]

    size = 14

    mpl.rcParams.update(mpl.rcParamsDefault)
    sns.set_theme(context='paper',style='white', font_scale=1)

    fig, ax = plt.subplots(1, 1, figsize=(6,4), dpi=300)
    ax.plot(x, sig, color='#DD8A2E', linewidth=2, label='Sigmoid')
    ax.plot(x, dsig, color='#DD8A2E', linestyle='--', linewidth=2, label='Sigmoid derivative')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    #ax.set_xlabel('Input', fontdict={'size': size, 'weight': 'bold'})
    #ax.set_ylabel('Activation', fontdict={'size': size, 'weight': 'bold'})
    sns.despine(left=True, bottom=True)
    plt.legend()
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(join(vis_dir, 'sigmoid.pdf'), bbox_inches='tight', pad_inches=0)
    plt.show()

    #####################
    # Contraction order #
    #####################
    import numpy as np
    import opt_einsum as oe

    dim = 10
    Z = np.random.rand(dim, dim, dim, dim)
    Y = np.random.rand(dim, dim)
    X = np.random.rand(dim, dim)

    path2 = oe.contract_path('ij,jk,klmn->ilmn',X,Y,Z)

    path1a = oe.contract_path('ij,jklm->iklm', Y, Z)
    T = oe.contract('ij,jklm->iklm', Y, Z)

    path1b = oe.contract_path('ij,jklm->iklm', X,T)

    cost_opt = path2[1].opt_cost
    cost_bad = path1a[1].opt_cost + path1b[1].opt_cost


