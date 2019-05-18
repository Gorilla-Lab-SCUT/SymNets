##############################################################################
# The simplified official code for the CVPR19 paper: Domain-Symnetric Networks for Adversarial Domain Adaptation
##############################################################################
import json
import os
import shutil
import time

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from data.prepare_data import generate_dataloader  # Prepare the data and dataloader
from models.resnet import resnet  # The model construction
from opts import opts  # The options for the project
from trainer import train  # For the training process
from trainer import validate  # For the validate (test) process
from trainer import adjust_learning_rate
from models.DomainClassifierTarget import DClassifierForTarget
from models.DomainClassifierSource import DClassifierForSource
from models.EntropyMinimizationPrinciple import EMLossForTarget
import ipdb

best_prec1 = 0

def main():
    global args, best_prec1
    current_epoch = 0
    epoch_count_dataset = 'source' ##
    args = opts()
    if args.arch == 'alexnet':
        raise ValueError('the request arch is not prepared', args.arch)
        # model = alexnet(args)
        # for param in model.named_parameters():
        #     if param[0].find('features1') != -1:
        #         param[1].require_grad = False
    elif args.arch.find('resnet') != -1:
        model = resnet(args)
    else:
        raise ValueError('Unavailable model architecture!!!')
    # define-multi GPU
    model = torch.nn.DataParallel(model).cuda()
    print(model)
    criterion_classifier_target = DClassifierForTarget(nClass=args.num_classes).cuda()
    criterion_classifier_source = DClassifierForSource(nClass=args.num_classes).cuda()
    criterion_em_target = EMLossForTarget(nClass=args.num_classes).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    # To apply different learning rate to different layer
    if args.arch == 'alexnet':
        optimizer = torch.optim.SGD([
            # {'params': model.module.features1.parameters(), 'name': 'pre-trained'},
            {'params': model.module.features2.parameters(), 'name': 'pre-trained'},
            {'params': model.module.classifier.parameters(), 'name': 'pre-trained'},
            {'params': model.module.fc.parameters(), 'name': 'new-added'}
        ],
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay, 
                                    nesterov=True)
    elif args.arch.find('resnet') != -1:
        optimizer = torch.optim.SGD([
            {'params': model.module.conv1.parameters(), 'name': 'pre-trained'},
            {'params': model.module.bn1.parameters(), 'name': 'pre-trained'},
            {'params': model.module.layer1.parameters(), 'name': 'pre-trained'},
            {'params': model.module.layer2.parameters(), 'name': 'pre-trained'},
            {'params': model.module.layer3.parameters(), 'name': 'pre-trained'},
            {'params': model.module.layer4.parameters(), 'name': 'pre-trained'},
            #{'params': model.module.fc.parameters(), 'name': 'pre-trained'}
            {'params': model.module.fc.parameters(), 'name': 'new-added'}
        ],
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay, 
                                    nesterov=True)
    else:
        raise ValueError('Unavailable model architecture!!!')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("==> loading checkpoints '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("==> loaded checkpoint '{}'(epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError('The file to be resumed from is not exited', args.resume)
    if not os.path.isdir(args.log):
        os.makedirs(args.log)
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state) + '\n')
    log.close()

    cudnn.benchmark = True
    # process the data and prepare the dataloaders.
    source_train_loader, source_val_loader, target_train_loader, val_loader = generate_dataloader(args)
    #test only
    if args.test_only:
        validate(val_loader, model, criterion, -1, args)
        return
    # start time
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write('\n-------------------------------------------\n')
    log.write(time.asctime(time.localtime(time.time())))
    log.write('\n-------------------------------------------')
    log.close()
    source_train_loader_batch = enumerate(source_train_loader)
    target_train_loader_batch = enumerate(target_train_loader)
    batch_number_s = len(source_train_loader)
    batch_number_t = len(target_train_loader)
    if batch_number_s < batch_number_t:
        epoch_count_dataset = 'target'
    while (current_epoch < args.epochs):
        # train for one iteration
        adjust_learning_rate(optimizer, current_epoch, args)
        source_train_loader_batch, target_train_loader_batch, current_epoch, new_epoch_flag = train(source_train_loader, source_train_loader_batch, target_train_loader, target_train_loader_batch, model, criterion_classifier_source, criterion_classifier_target, criterion_em_target, criterion, optimizer, current_epoch, epoch_count_dataset, args)
        # evaluate on the val data
        if new_epoch_flag:
            if (current_epoch + 1) % args.test_freq == 0 or current_epoch == 0:
                prec1 = validate(val_loader, model, criterion, current_epoch, args)
                # record the best prec1 and save checkpoint
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                if is_best:
                    log = open(os.path.join(args.log, 'log.txt'), 'a')
                    log.write('     Best acc: %3f' % (best_prec1))
                    log.close()
                    save_checkpoint({
                        'epoch': current_epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_prec1,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best, args)

    # end time
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write('\n-------------------------------------------\n')
    log.write(time.asctime(time.localtime(time.time())))
    log.write('\n-------------------------------------------\n')
    log.close()


def save_checkpoint(state, is_best, args):
    filename = 'checkpoint.pth.tar'
    dir_save_file = os.path.join(args.log, filename)
    torch.save(state, dir_save_file)
    if is_best:
        shutil.copyfile(dir_save_file, os.path.join(args.log, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()





