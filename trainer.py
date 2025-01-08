import argparse
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet") or name.startswith("highway")
                     and callable(resnet.__dict__[name]))

parser = argparse.ArgumentParser(description='ResNet/Highway110 for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet110',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet110)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--wm', '--width-mult', dest='width_mult', default=1.0, type=float,
                    metavar='WIDTH_MULTIPLIER', help='layer width scaling factor')
parser.add_argument('--amp', dest='amp', action='store_true',
                    help='use AMP (bf16) ')
parser.add_argument('--no-compile', dest='no_compile', action='store_true',
                    help='disable torch.compile')
parser.add_argument('--final-conv', dest='final_conv', action='store_true',
                    help='add a final conv layer after the blocks')
parser.add_argument('--save-every', dest='save_every',
                    help='saves checkpoints at every specified number of epochs',
                    type=int, default=1000)
best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    # Check the save_dir exists or not
    save_dir = f"log_{args.arch}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"created {save_dir}")

    model = resnet.__dict__[args.arch](width_mult=args.width_mult, final_conv=args.final_conv)
    model.cuda()
    if not args.no_compile:
        print('torch.compile is ON, starting training will take a few mins')
        model = torch.compile(model, mode='max-autotune')
    
    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"model has {num_params:,} parameters")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True, persistent_workers=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False, persistent_workers=True,
        num_workers=args.workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=0.9,
                                weight_decay=1e-4)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])

    if args.arch in ['resnet110']:
        # the idenity mappings paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1


    logs = defaultdict(list)
    logs['args'] = {'arch': args.arch, 'lr': args.lr, 'epochs': args.epochs, 'batch_size': args.batch_size, 'width_mult': args.width_mult, 'final_conv': args.final_conv}
    logs['num_params'] = num_params
    print(f"args: {logs['args']}")

    for epoch in range(args.epochs):

        if args.arch in ['resnet110'] and epoch == 1:
            # switch back the lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        # train for one epoch
        start = time.time()
        train_loss = train(train_loader, model, criterion, optimizer, epoch)
        train_time = time.time() - start

        # evaluate
        start = time.time()
        test_loss, acc = validate(val_loader, model, criterion)
        test_time = time.time() - start
        
        current_lr = optimizer.param_groups[0]['lr']
        lr_scheduler.step()

        # remember best prec@1 and save checkpoint
        best_prec1 = max(acc, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, os.path.join(save_dir, 'checkpoint.th'))
        
        status = f"epoch {epoch:3d} | train loss: {train_loss:.4f} | test loss: {test_loss:.4f} | test acc: {acc:.2f} | train time: {train_time:.3f} | test time: {test_time:.3f} | lr: {current_lr:.1e}"
        print(status)
        with open(os.path.join(save_dir, 'out.txt'), 'a') as f: 
            f.write(status + '\n')
        
        logs['train_loss'].append(train_loss)
        logs['test_loss'].append(test_loss)
        logs['test_acc'].append(acc)
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump(logs, f)


def train(train_loader, model, criterion, optimizer, epoch) -> torch.tensor:
    device = next(model.parameters()).device
    loss_sum = torch.tensor([0.0], device=device)

    model.train()

    for (input, target) in train_loader:

        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        context = torch.autocast('cuda', dtype=torch.bfloat16) if args.amp else nullcontext()
        with context:
            output = model(input)
            loss = criterion(output, target)

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        loss_sum += (loss.detach().float() * input.size(0))
        
    return loss_sum.cpu().item() / 50_000


@torch.no_grad()
def validate(val_loader, model, criterion) -> tuple[float, float]:
    device = next(model.parameters()).device
    loss_sum = torch.tensor([0.0], device=device)
    acc_sum = torch.tensor([0.0], device=device)

    model.eval()

    for (input, target) in val_loader:
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        context = torch.autocast('cuda', dtype=torch.bfloat16) if args.amp else nullcontext()
        with context:
            output = model(input)
            loss = criterion(output, target)

        loss_sum += loss.detach().float() * input.size(0)
        acc_sum += (output.detach().max(dim=1)[1] == target).sum()

    return loss_sum.item() / 10_000, acc_sum.float().item() / 100  # acc is in %


if __name__ == '__main__':
    main()
