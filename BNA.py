import copy
import time
from math import ceil

import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import numpy as np
import os
import argparse
import scipy.stats as stats

from src.GTSRB import GTSRB
from src.ImageNette import ImageNette
from src.TinyImageNet import TinyImageNetDataset
from src.VGGFaceBD import VGGFaceBD
# from src.resnet_2 import ResNet18, ResNet50
from src.resnet import ResNet18, ResNet50
from torchvision.models import resnet34
from src.MobileNet import MobileNet
from src.VGG import vgg16_bn
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from src.utils import add_backdoor
import copy as cp
from matplotlib import pyplot, pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='backdoor mitigation')
parser.add_argument('--dataset', default='CIFA10', help='the dataset to use')
parser.add_argument('--target_class', type=int, default=9, help='the target class')
parser.add_argument('--attack_idx', default='1_all_exp', help='attack index')
parser.add_argument('--pattern_position', default='bottom_right', help='pattern position')
parser.add_argument('--model_type', default='resnet18', help='model type')
parser.add_argument('--p1', default='3x3', help='pattern size')
parser.add_argument('--p2', default=100, type=int, help='# poison')
parser.add_argument('--tau', default=150, type=float, help='scale factor')
parser.add_argument('--bin_size', default=0.1, type=float, help='bin size')
parser.add_argument('--pert_mask', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--save', action='store_true')
args = parser.parse_args()
args_dict = vars(args)
print(args_dict)

class SoftHistogram(nn.Module):
    def __init__(self, binsize, min, max, sigma=3 * 50):
        super(SoftHistogram, self).__init__()
        self.binsize = binsize
        self.bins = ceil((max - min)/binsize)
        self.min = min
        self.max = max
        self.sigma = sigma
        # self.binsize = float(max - min) / float(bins)
        self.centers = float(min) + self.binsize * (torch.arange(self.bins).float() + 0.5)

    def forward(self, x):
        if x.get_device() >= 0:  # on cuda
            self.centers = self.centers.to(x.get_device())
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.sigmoid(self.sigma * (x + self.binsize / 2)) - torch.sigmoid(self.sigma * (x - self.binsize / 2))
        x = x.sum(dim=1)
        return x


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def BN_bound_divergence_layer_by_layer():
    BATCH_SIZE = 1350
    BS = 32
    Epoch = 10
    # lr = 0.001
    bound_int = 3  # 5
    momentum = 1

    t_est = args.target_class  # true target class
    tt = args.target_class

    binwidth = args.bin_size  # 0.1
    temperature = args.tau  # 3 * 50

    if args.dataset == 'CIFA10':
        NC = 10
        N_clean = 100  # 100
    if args.dataset == 'CIFA100':
        NC = 100
        N_clean = 10
    if args.dataset == 'TinyImageNet':
        NC = 100
        N_clean = 10
    if args.dataset == 'ImageNette':
        NC = 10
    if args.dataset == 'GTSRB':
        NC = 43
        N_clean = 50
    if args.dataset == 'VGGFace':
        NC = 18
        N_clean = 2
        if args.attack_idx == 'Refool':
            NC = 10
    if args.model_type == 'resnet18':
        net = ResNet18(num_classes=NC)
        net_poisoned = ResNet18(num_classes=NC)
    if args.model_type == 'resnet34':
        net = resnet34(num_classes=NC)
        net_poisoned = resnet34(num_classes=NC)
    if args.model_type == 'mobilenet':
        net = MobileNet(num_classes=NC)
        net_poisoned = MobileNet(num_classes=NC)
    if args.model_type == 'vgg16':
        net = vgg16_bn(num_classes=NC)
        net_poisoned = vgg16_bn(num_classes=NC)

    if 'exp' in args.attack_idx:
        model_path = './model_{}/{}/{}/{}/model_{}_{}.pth'.format(args.attack_idx, args.model_type, args.dataset, args.pattern_position, args.p1, args.p2)
    else:
        model_path = './model_{}/{}/{}/{}/model.pth'.format(args.attack_idx, args.model_type, args.dataset, args.pattern_position)
        # model_path = './model/ResNet_clean.pth'
    if args.pert_mask:
        if 'exp' in args.attack_idx:
            pattern = torch.load(
                './attacks_{}/pert_mask_estimated_AD/{}/{}_{}/{}/pert_{}'.format(args.attack_idx, args.dataset, args.p1, args.p2, args.pattern_position, tt))
            mask = torch.load(
                './attacks_{}/pert_mask_estimated_AD/{}/{}_{}/{}/mask_{}'.format(args.attack_idx, args.dataset, args.p1, args.p2, args.pattern_position, tt))
        else:
            pattern = torch.load(
                './attacks_{}/pert_mask_estimated_AD/{}/{}/pert_{}'.format(args.attack_idx, args.dataset, args.pattern_position, tt))
            mask = torch.load(
                './attacks_{}/pert_mask_estimated_AD/{}/{}/mask_{}'.format(args.attack_idx, args.dataset, args.pattern_position, tt))
    else:
        if 'exp' in args.attack_idx:
            pert = torch.load('./attacks_{}/pert_estimated_AD/{}/{}_{}/{}/pert_{}'.format(args.attack_idx, args.dataset, args.p1, args.p2, args.pattern_position, tt))
        else:
            pert = torch.load('./attacks_{}/pert_estimated_AD/{}/{}/pert_{}'.format(args.attack_idx, args.dataset, args.pattern_position, tt))
            print(torch.norm(pert))

    net.to(device)
    net_poisoned.to(device)
    net.load_state_dict(torch.load(model_path))
    net_poisoned.load_state_dict(torch.load(model_path))

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = None
    if args.dataset == 'CIFA10':
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    if args.dataset == 'CIFA100':
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    if args.dataset == 'TinyImageNet':
        testset = TinyImageNetDataset(root_dir='./data/tiny-imagenet-200', mode='val', download=False,
                                      transform=transform_test, subset=NC)
    if args.dataset == 'ImageNette':
        testset = ImageNette(root='./data/imagenette', split='val', transform=transform_test)
    if args.dataset == 'GTSRB':
        testset = GTSRB(root='./data/GTSRB', split='test', transform=transform_test)
    if args.dataset == 'VGGFace':
        testset = VGGFaceBD(root='./data/VGG-Face2/data', img_size=224, train=False, transform=transform_test)
    Test_loader = Data.DataLoader(
        dataset=testset,
        batch_size=BS,
        shuffle=True, num_workers=1, )

    if 'exp' in args.attack_idx:
        test_attacks = torch.load('./attacks_{}/{}/size_{}/{}/test_attacks'.format(args.attack_idx, args.dataset, args.p1, args.pattern_position))
    else:
        test_attacks = torch.load('./attacks_{}/{}/{}/test_attacks'.format(args.attack_idx, args.dataset, args.pattern_position))
    test_attacks_image = test_attacks['image']
    # test_attacks_label = test_attacks['label']
    test_attacks_label = test_attacks['groundTruth']
    test_attacks = Data.TensorDataset(test_attacks_image, test_attacks_label)
    Test_attack_loader = Data.DataLoader(
        dataset=test_attacks,
        batch_size=BS,
        shuffle=True, num_workers=1, )

    images_source_attack = None
    label_source_attack = None
    images_source_clean = None
    images_target_clean = None
    if 'exp' in args.attack_idx:
        ind_test = torch.load('./attacks_{}/{}/size_{}/{}/ind_test'.format(args.attack_idx, args.dataset, args.p1, args.pattern_position)).astype(int)
    else:
        ind_test = torch.load('./attacks_{}/{}/{}/ind_test'.format(args.attack_idx, args.dataset, args.pattern_position)).astype(int)  # indices of backdoor samples (exclude t_est)
    idx_mask = np.ones(len(testset.targets), dtype=bool)
    idx_mask[ind_test] = False
    validation = np.array(testset.targets)[idx_mask]
    ind_local = np.where(idx_mask == True)[0]
    for s in range(NC):
        # if s != t_est:
        ind_source = [i for i, label in enumerate(validation) if label == s]
        if args.dataset in ['GTSRB', 'ImageNette']:
            N_clean = int(len(ind_source)/4)
        ind_source = np.random.choice(ind_source, N_clean, False)
        for i in ind_source:
            clean_image = testset.__getitem__(ind_local[i])[0].unsqueeze(0)
            if images_source_clean is None:
                images_source_clean = clean_image
            else:
                images_source_clean = torch.cat([images_source_clean, clean_image])
            
            if args.pert_mask:
                pert_image = torch.clamp(testset.__getitem__(ind_local[i])[0] * (1 - mask) + pattern * mask, min=0,
                                         max=1).unsqueeze(0)
            else:
                pert_image = add_backdoor(testset.__getitem__(ind_local[i])[0], pert).unsqueeze(0)
                # pert_image = F.grid_sample(testset.__getitem__(ind_local[i])[0].unsqueeze(0), pert.repeat(1,1,1,1), align_corners=True)
            if images_source_attack is None:
                images_source_attack = pert_image
                label_source_attack = torch.tensor([s], dtype=torch.long)
            else:
                images_source_attack = torch.cat([images_source_attack, pert_image], dim=0)
                label_source_attack = torch.cat([label_source_attack, torch.tensor([s], dtype=torch.long)], dim=0)
    label_source_clean = label_source_attack.clone()


    print(images_source_attack.shape, images_source_clean.shape)
    images_attack = Data.TensorDataset(images_source_attack, label_source_attack)
    images_attack = Data.DataLoader(
        dataset=images_attack,
        batch_size=BATCH_SIZE,
        shuffle=True, num_workers=1, )
    images_clean = Data.TensorDataset(images_source_clean, label_source_clean)
    images_clean = Data.DataLoader(
        dataset=images_clean,
        batch_size=BATCH_SIZE,
        shuffle=True, num_workers=1, )

    # predictions before BN
    print('before:')
    net.eval()
    clean_target_preds_before = np.empty(0)
    attack_test_preds_before = np.empty(0)
    total = 0
    correct = 0
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(Test_loader):  # for each training step
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            prediction = net(batch_x)  # input x and predict based on x
            _, predicted = prediction.max(1)
            total += batch_y.shape[0]
            correct += predicted.eq(batch_y).sum().item()

            ind = batch_y == t_est
            if ind.sum().item() == 0:
                continue
            batch_x = batch_x[ind]
            prediction = net(batch_x)  # input x and predict based on x
            _, predicted = prediction.max(1)
            clean_target_preds_before = np.concatenate((clean_target_preds_before, predicted.cpu().detach().numpy()))
    print('clean acc: ', correct / total)
    print('clean target acc: {:.4f}'.format(
        (clean_target_preds_before == t_est).sum() / clean_target_preds_before.shape[0]))

    total = 0
    correct = 0
    target_num = 0
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(Test_attack_loader):  # for each training step
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            prediction = net(batch_x)  # input x and predict based on x
            _, predicted = prediction.max(1)
            total += batch_y.shape[0]
            correct += predicted.eq(batch_y).sum().item()
            attack_test_preds_before = np.concatenate((attack_test_preds_before, predicted.cpu().detach().numpy()))
            target_num += predicted.eq(t_est).sum().item()
    print('ASR: {:.4f}'.format(target_num / total))
    print('SIA: {:.4f}'.format(correct / total))

    internal_layer_feature = {}
    input_feature = {}
    BN_layers = []
    BN_layers_dict = {}
    forward_hook_handles = {}
    hist_clean = {}
    bin_max = {}
    bin_min = {}
    start_time = time.time()

    # 1. get histograms from each BN layer on clean samples
    # 1.1 register hook in each BN layer to get output activation
    def get_activation(name):
        def hook(model, input, output):
            internal_layer_feature[name] = output.detach()

        return hook

    for name, layer in net.named_modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            BN_layers.append(name)
            BN_layers_dict[name] = layer
            forward_hook_handles[name] = layer.register_forward_hook(get_activation(name))

    # 1.2 histograms from each BN layer on clean samples
    with torch.no_grad():
        for _, (batch_x, _) in enumerate(images_attack):
            batch_x = batch_x.to(device)
            net(batch_x)
            for name in BN_layers:
                activations = internal_layer_feature[name]
                activations = activations.permute([1, 0, 2, 3])
                activations = activations.reshape(activations.shape[0], -1).cpu()
                bin_max[name] = activations.max(dim=1)[0]
                bin_min[name] = activations.min(dim=1)[0]
                # print(max(bin_max[name]), min(bin_min[name]))

    with torch.no_grad():
        for _, (batch_x, _) in enumerate(images_clean):
            batch_x = batch_x.to(device)
            net(batch_x)
            for name in BN_layers:
                activations = internal_layer_feature[name]
                activations = activations.permute([1, 0, 2, 3])
                activations = activations.reshape(activations.shape[0], -1).cpu()
                bin_max[name] = torch.maximum(bin_max[name], activations.max(dim=1)[0])
                bin_min[name] = torch.minimum(bin_min[name], activations.min(dim=1)[0])
               
                PMFs = []
                for i in range(activations.shape[0]):
                    bin_num = ceil((bin_max[name][i] - bin_min[name][i])/binwidth)
                    if bin_num == 0:
                        PMF = None
                    else:
                        PMF = torch.histc(activations[i], bins=bin_num, min=bin_min[name][i], max=bin_max[name][i])
                    # PMF = PMF/PMF.sum()
                    PMFs.append(PMF)
                hist_clean[name] = PMFs
                # print(max(bin_max[name]), min(bin_min[name]))
        # return

    # 1.3 remove hooks
    for name in BN_layers:
        forward_hook_handles[name].remove()

    # 2. align distribution in BN layers
    # 2.1 set momentum in BN layers
    for _, layer in net.named_modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            layer.momentum = momentum
    # 2.2 update BN statistics (mean & var) using pytorch as the initial values
    net.train()
    for param in net.parameters():
        param.requires_grad = False
    # for epoch in range(Epoch):
    for _, (batch_x, _) in enumerate(images_attack):
        batch_x = batch_x.to(device)
        net(batch_x)
    net.eval()

    # 2.3 optimize mean&var&bound layer by layer
    bounds = {name: torch.ones(len(hist_clean[name]), 1, 1)*bound_int for name in BN_layers}
    
    for name in BN_layers:
        bounds[name] = bounds[name].to(device)
        bounds[name].requires_grad = False

    def get_activation(name, bound):
        def hook(model, input, output):
            output = torch.minimum(output, bound)
            internal_layer_feature[name] = output
            input_feature[name] = input[0].detach()
            return output
        return hook

    # for each channel in each BN layer
    KLD = torch.nn.KLDivLoss(reduction='batchmean')
    for name in BN_layers:
        if args.verbose:
            print('layer ', name)
        current_layer = BN_layers_dict[name]
        # initialization and turn on back-propagation
        means = current_layer.running_mean.clone()
        vars = current_layer.running_var.clone()
        forward_hook_handles[name] = current_layer.register_forward_hook(get_activation(name, bounds[name]))
        means = means.to(device)
        vars = vars.to(device)
        bounds[name].requires_grad = True
        means.requires_grad = True
        vars.requires_grad = True

        # get the input activation for the current layer
        for step, (batch_x, _) in enumerate(images_attack):
            # feed images with estimated pattern
            batch_x = batch_x.to(device)
            net(batch_x)
            input_activations = input_feature[name]

        for i in range(len(hist_clean[name])):
            # init optimizer, scheduler, loss function
            lr = 0.01
            optimizer = torch.optim.Adam([bounds[name], means, vars], lr=lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1, last_epoch=-1)

            # optimize mean&var&bounds by minimizing total variation distance
            loss_old = 0
            for epoch in range(Epoch):
                activations = current_layer(input_activations)
                activations = torch.minimum(activations, bounds[name])
                activations = activations.permute([1, 0, 2, 3])
                activations = activations.reshape(activations.shape[0], -1)
                activations = activations[i]
                PMF_c = hist_clean[name][i]
                if PMF_c is None:
                    break
                PMF_c = PMF_c.to(device)
                softhist = SoftHistogram(binsize=binwidth, min=bin_min[name][i], max=bin_max[name][i], sigma=temperature)
                PMF_b = softhist(activations)

                # choice 1 total variation distance between PMF_c and PMF_b
                loss = torch.abs(PMF_b - PMF_c).sum()
                # # choice 2 JS divergence between PMF_c and PMF_b
                # PMF_c /= PMF_c.sum()
                # PMF_b /= PMF_b.sum()
                # PMF_m = (PMF_c + PMF_b)/2
                # loss = KLD(PMF_m, PMF_c) + KLD(PMF_m, PMF_b)
                # # choice 3 KL divergence between PMF_c and PMF_b
                # PMF_c /= PMF_c.sum()
                # PMF_b /= PMF_b.sum()
                # loss = KLD(PMF_c, PMF_b)
                loss_new = loss.item()

                optimizer.zero_grad()  # clear gradients for next train
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

                # update mean&var by current values
                # BN layer's mean&var cannot be trained
                current_layer.running_mean = means.clone().detach()
                current_layer.running_var = vars.clone().detach()

                if epoch == 0:
                    loss_old = loss_new
                else:
                    if abs(loss_old-loss_new)/loss_old < 1e-5:
                        break
                    else:
                        loss_old = loss_new
                # print('epoch ', epoch, loss_new)
                scheduler.step()
            if args.verbose:
                print('channel {} converges in epoch {}'.format(i, epoch))

        # turn off back-propagation for the current layer
        bounds[name].requires_grad = False
        means.requires_grad = False
        vars.requires_grad = False

    print('time (after transformation parameter optimization) ', time.time()-start_time)

    ########## combine poisoned model and new model ##########
    print('after:')
    clean_target_preds_after = np.empty(0)
    attack_test_preds_after = np.empty(0)
    # clean test set acc
    net_poisoned.eval()
    net.eval()
    total = testset.__len__()
    correct = 0
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(Test_loader):  # for each training step
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device, dtype=torch.long)
            prediction = net_poisoned(batch_x)  # input x and predict based on x
            _, predicted = prediction.max(1)

            ind = predicted != t_est
            correct += predicted[ind].eq(batch_y[ind]).sum().item()

    correct_t = 0
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(Test_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device, dtype=torch.long)
            ind = batch_y == t_est
            prediction = net_poisoned(batch_x)  # input x and predict based on x
            _, predicted_poisoned = prediction.max(1)
            prediction = net(batch_x)  # input x and predict based on x
            _, predicted = prediction.max(1)

            correct_t += torch.logical_and(predicted_poisoned[ind].eq(t_est), predicted[ind].eq(t_est)).sum().item()

            clean_target_preds_after = np.concatenate((clean_target_preds_after, predicted[ind].cpu().detach().numpy()))

    print('test acc: {:.4f}'.format((correct+correct_t) / total))

    # attack success rate
    net.eval()
    net_poisoned.eval()
    total = 0
    correct = 0  # sia
    target_num = 0  # asr
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(Test_attack_loader):  # for each training step
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device, dtype=torch.long)
            total += batch_y.size(0)
            prediction = net(batch_x)
            _, predicted = prediction.max(1)
            prediction = net_poisoned(batch_x)  # input x and predict based on x
            _, predicted_poisoned = prediction.max(1)

            target_num += torch.logical_and(predicted_poisoned.eq(t_est), predicted.eq(t_est)).sum().item()

            correct += predicted_poisoned.eq(batch_y).sum().item()
            ind = predicted_poisoned == t_est
            correct += predicted[ind].eq(batch_y[ind]).sum().item()

            attack_test_preds_after = np.concatenate((attack_test_preds_after, predicted.cpu().detach().numpy()))
    print('SIA: {:.4f}'.format(correct / total))
    print('ASR: {:.4f}'.format(target_num / total))

    # in-flight detection: for those detected as the target class, if prediction changes, then it is backdoor
    print('in-flight detection:')
    # FPR -- how many clean target class imagess are falsely detected as backdoor image
    ind_t = clean_target_preds_before == t_est
    ind = clean_target_preds_before[ind_t] != clean_target_preds_after[ind_t]
    print('FPR: {:.4f}'.format(ind.sum() / ind_t.sum()))
    # TPR -- how many backdoor-trigger imagess are correctly detected
    ind_t = attack_test_preds_before == t_est
    ind = attack_test_preds_before[ind_t] != attack_test_preds_after[ind_t]
    print('TPR: {:.4f}'.format(ind.sum() / ind_t.sum()))

    print('time (after in-flight detection) ', time.time()-start_time)


    # save model & bounds
    if args.save:
        torch.save(net.state_dict(), './model_{}/{}/{}/{}/model_BN_min_hist_TV.pth'.format(args.attack_idx, args.model_type, args.dataset, args.pattern_position))
        torch.save(bounds, './model_{}/{}/{}/{}/bounds_TV'.format(args.attack_idx, args.model_type, args.dataset, args.pattern_position))

