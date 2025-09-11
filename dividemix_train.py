import argparse
import random
import signal
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import tqdm
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import ResNet50_Weights
from transformers import AutoImageProcessor, AutoModelForImageClassification

import data
import dividemix_dataloader as dataloader
import lib
import src.training
from src.util_siglip import set_freezing

writer = SummaryWriter()


stop = threading.Event()

def handle_signal(signum, frame):
    stop.set()

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT,  handle_signal)

parser = argparse.ArgumentParser(description='DivideMix training')
parser.add_argument('--checkpoint_dir', default='dividemix_checkpoints', type=str, help='directory for checkpoints')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
# parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=5, type=int)
# parser.add_argument('--data_path', default='../../Clothing1M/data', type=str, help='path to dataset')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=len(data.species_labels), type=int)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Training
def train(epoch, net1, net2, optimizer, labeled_trainloader, unlabeled_trainloader):
    net1.train()
    net2.eval()  # fix one network and train the other

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1

    for batch_idx, (inputs_lab_1, inputs_lab_2, labels_x, probs_x, ids) in tqdm.tqdm(enumerate(labeled_trainloader), total=len(labeled_trainloader), desc="Training"):
        try:
            inputs_unlab_1, inputs_unlab_2, id = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_unlab_1, inputs_unlab_2, id = next(unlabeled_train_iter)
        batch_size = inputs_lab_1.size(0)

        # Transform label to one-hot
        # labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        probs_x = probs_x.reshape(-1, 1).float()

        inputs_lab_1, inputs_lab_2, labels_x, probs_x = inputs_lab_1.cuda(), inputs_lab_2.cuda(), labels_x.cuda(), probs_x.cuda()
        inputs_unlab_1, inputs_unlab_2 = inputs_unlab_1.cuda(), inputs_unlab_2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net1(inputs_unlab_1).logits
            outputs_u12 = net1(inputs_unlab_2).logits
            outputs_u21 = net2(inputs_unlab_1).logits
            outputs_u22 = net2(inputs_unlab_2).logits

            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1)
                  + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4
            ptu = pu ** (1 / args.T)  # temparature sharpening

            targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
            targets_u = targets_u.detach()

            # label refinement of labeled samples
            outputs_x1 = net1(inputs_lab_1).logits
            outputs_x2 = net1(inputs_lab_2).logits

            px = (torch.softmax(outputs_x1, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = probs_x * labels_x + (1 - probs_x) * px
            ptx = px ** (1 / args.T)  # temparature sharpening

            targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
            targets_x = targets_x.detach()

        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)

        all_inputs = torch.cat([inputs_lab_1, inputs_lab_2, inputs_unlab_1, inputs_unlab_2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a[:batch_size * 2] + (1 - l) * input_b[:batch_size * 2]
        mixed_target = l * target_a[:batch_size * 2] + (1 - l) * target_b[:batch_size * 2]

        logits = net1(mixed_input).logits

        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))

        # regularization
        prior = torch.ones(args.num_class) / args.num_class
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        loss = Lx + penalty

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # sys.stdout.write('\r')
        # sys.stdout.write('Clothing1M | Epoch [%3d/%3d] Iter[%3d/%3d]\t  Labeled loss: %.4f '
        #                  % (epoch, args.num_epochs, batch_idx + 1, num_iter, Lx.item()))
        # sys.stdout.flush()


def warmup(net, optimizer, dataloader):
    net.train()
    for batch_idx, (images, labels, ids) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc="Warm-up"):
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()

        outputs = net(images).logits

        loss = CEloss(outputs, labels)
        penalty = conf_penalty(outputs)
        L = loss + penalty

        L.backward()
        optimizer.step()

        # sys.stdout.write('\r')
        # sys.stdout.write(
        #     '|Warm-up: Iter[%3d/%3d]\t CE-loss: %.4f  Conf-Penalty: %.4f' % (batch_idx + 1, args.num_batches,
        #                                                                      loss.item(), penalty.item()))
        # sys.stdout.flush()


def val(net, val_loader, k):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, targets, ids) in tqdm.tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation"):
            images, targets = images.cuda(), targets.cuda()
            outputs = net(images).logits

            _, target_idx = torch.max(targets, dim=1)
            _, predicted = torch.max(outputs, dim=1)

            total += targets.size(0)
            correct += predicted.eq(target_idx).cpu().sum().item()
    acc = 100. * correct / total

    # print("\n| Validation\t Net%d  Acc: %.2f%%" % (k, acc))
    # if acc > best_acc[k - 1]:
    #     best_acc[k - 1] = acc
    #     print('| Saving Best Net%d ...' % k)
    #     save_point = './checkpoint/%s_net%d.pth.tar' % (args.id, k)
    #     torch.save(net.state_dict(), save_point)
    return acc


def test(net1, net2, test_loader):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, ids) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing"):
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs1 = net1(inputs).logits
            outputs2 = net2(inputs).logits
            outputs = outputs1 + outputs2

            _, target_idx = torch.max(targets, dim=1)
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += predicted.eq(target_idx).cpu().sum().item()
    acc = 100. * correct / total
    print("\n| Test Acc: %.2f%%\n" % (acc))
    return acc


def eval_train(epoch, model, eval_loader: DataLoader):
    model.eval()
    num_samples = len(eval_loader.dataset)
    losses = torch.zeros(num_samples)
    ids = []
    n = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, ids_) in tqdm.tqdm(enumerate(eval_loader), total=len(eval_loader), desc="Eval-training"):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs).logits
            loss = CE(outputs, targets)

            losses[n:n + loss.size(0)] = loss.detach().cpu()
            n += loss.size(0)

            ids.extend(ids_)

            # for b in range(inputs.size(0)):
            #     losses[n] = loss[b]
            #     paths.append(id[b])
            #     n += 1
            # sys.stdout.write('\r')
            # sys.stdout.write('| Evaluating loss Iter %3d\t' % (batch_idx))
            # sys.stdout.flush()

    losses = (losses - losses.min()) / (losses.max() - losses.min())
    losses = losses.reshape(-1, 1)

    gmm = GaussianMixture(n_components=2, max_iter=20, reg_covar=5e-4, tol=1e-2)
    gmm.fit(losses)

    prob = gmm.predict_proba(losses)
    prob = prob[:, gmm.means_.argmin()]

    return prob, ids


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


def create_model_siglip2_base_256(num_labels):
    model_id = "google/siglip2-base-patch16-256"  # FixRes вариант

    model = AutoModelForImageClassification.from_pretrained(
        model_id,
        num_labels=num_labels,

        ignore_mismatched_sizes=True,  # создаст новую голову нужного размера
    )

    model.cuda()

    model.tracking_loss = []
    model.tracking_loss_val = []
    model.tracking_accuracy = []
    # model.tracking_val_probs = []
    # the last epoch we finished training on
    model.epoch = None

    return model, AutoImageProcessor.from_pretrained(model_id)


def create_model_resnet_50(num_labels):
    model = models.convnext_large

    resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(2048, num_labels)
    model = model.cuda()
    return model


print('| Building net')
# net1 = create_model_resnet_50(args.num_class)
# net2 = create_model_resnet_50(args.num_class)

storage = src.training.CheckpointsStorage(dir=Path('dividemix_checkpoint'))

latest1, epoch1 = storage.latest(r'net1_(\d+)\.pth')
latest2, epoch2 = storage.latest(r'net2_(\d+)\.pth')

if epoch1 != epoch2:
    raise ValueError('Checkpoints of models out of sync.')

epoch = 0

if latest1 and latest2:
    checkpoint1 = torch.load(latest1, weights_only=False)
    checkpoint2 = torch.load(latest2, weights_only=False)

    epoch = epoch1 + 1

    net1 = checkpoint1['model']
    net2 = checkpoint2['model']

    optimizer1 = checkpoint1['optimizer']
    optimizer2 = checkpoint2['optimizer']

    preprocessor1 = AutoImageProcessor.from_pretrained(net1.name_or_path)
    preprocessor2 = AutoImageProcessor.from_pretrained(net2.name_or_path)
else:
    net1, preprocessor1 = create_model_siglip2_base_256(args.num_class)
    net2, preprocessor2 = create_model_siglip2_base_256(args.num_class)

    # optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
    # optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)

    optimizer1 = optim.AdamW(net1.parameters(), lr=args.lr, weight_decay=1e-3)
    optimizer2 = optim.AdamW(net2.parameters(), lr=args.lr, weight_decay=1e-3)

loader = dataloader.DividemixDataloaderFactory(
    data_dir=Path(__file__).parent / 'data',
    batch_size=args.batch_size,
    num_workers=6,
    x_train=data.x_train,
    y_train=data.y_train,
    x_eval=data.x_eval,
    y_eval=data.y_eval,
    preprocessor=lambda image: preprocessor1(images=image, return_tensors="pt")['pixel_values'].squeeze(0),
    aug_train=lib.siglip2_training_transform,
    aug_inference=lib.siglip2_inference_transform,
)

warmup_epochs = 5

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()

prob1 = None
prob2 = None

for epoch in range(epoch, epoch + args.num_epochs):
    if stop.is_set():
        break

    print(f"Starting epoch { epoch }")

    # lr = args.lr
    # if epoch >= 40:
    #     lr /= 10
    # for param_group in optimizer1.param_groups:
    #     param_group['lr'] = lr
    # for param_group in optimizer2.param_groups:
    #     param_group['lr'] = lr

    if epoch < warmup_epochs:  # warm up
        set_freezing(net1, optimizer1, 'classifier_only')
        set_freezing(net2, optimizer2, 'classifier_only')

        print('Warmup Net1')
        train_loader = loader.get_warmup_dataloader()
        warmup(net1, optimizer1, train_loader)

        print('Warmup Net2')
        train_loader = loader.get_warmup_dataloader()
        warmup(net2, optimizer2, train_loader)
    else:
        if prob1 is not None and prob2 is not None:
            set_freezing(net1, optimizer1, 'classifier_and_encoder')
            set_freezing(net2, optimizer2, 'classifier_and_encoder')

            pred1 = prob1 > args.p_threshold  # divide dataset
            pred2 = prob2 > args.p_threshold

            print('Train Net1')
            labeled_trainloader, unlabeled_trainloader = loader.get_train_dataloader(ids2, prob2, pred2)  # co-divide
            train(epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader)  # train net1

            print('Train Net2')
            labeled_trainloader, unlabeled_trainloader = loader.get_train_dataloader(ids1, prob1, pred1)  # co-divide
            train(epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader)  # train net2

    if epoch >= warmup_epochs - 1:
        val_loader = loader.get_validation_dataloader()  # validation
        acc1 = val(net1, val_loader, 1)
        acc2 = val(net2, val_loader, 2)

        print(f'Validation Epoch:{ epoch }      Acc1: {acc1:.2f}  Acc2:{acc2:.2f}')

        print('==== net 1 evaluate next epoch training data loss ====')
        eval_loader = loader.get_eval_train_dataloader()  # evaluate training data loss for next epoch
        prob1, ids1 = eval_train(epoch, net1, eval_loader)

        print('==== net 2 evaluate next epoch training data loss ====')
        eval_loader = loader.get_eval_train_dataloader()
        prob2, ids2 = eval_train(epoch, net2, eval_loader)

    lib.save_model(net1, optimizer1, f'./dividemix_checkpoint/net1_{str(epoch).rjust(3, "0")}.pth')
    lib.save_model(net2, optimizer2, f'./dividemix_checkpoint/net2_{str(epoch).rjust(3, "0")}.pth')

test_loader = loader.get_test_dataloader()
acc = test(net1, net2, test_loader)

print('Test Accuracy:%.2f\n' % (acc))

writer.close()