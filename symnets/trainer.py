import time
import torch
import os
import math
import ipdb
import torch.nn.functional as F

def train(source_train_loader, source_train_loader_batch, target_train_loader, target_train_loader_batch, model, criterion_classifier_source, criterion_classifier_target, criterion_em_target, criterion, optimizer, epoch, epoch_count_dataset, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_classifier = AverageMeter()
    losses_G = AverageMeter()
    top1_source = AverageMeter()
    top1_target = AverageMeter()
    model.train()
    new_epoch_flag = False
    end = time.time()    
    try:
        (input_source, target_source) = source_train_loader_batch.__next__()[1]
    except StopIteration:
        if epoch_count_dataset == 'source':
            epoch = epoch + 1
            new_epoch_flag = True
        source_train_loader_batch = enumerate(source_train_loader)
        (input_source, target_source) = source_train_loader_batch.__next__()[1]

    try:
        (input_target, _) = target_train_loader_batch.__next__()[1]
    except StopIteration:
        if epoch_count_dataset == 'target':
            epoch = epoch + 1
            new_epoch_flag = True
        target_train_loader_batch = enumerate(target_train_loader)
        (input_target, _) = target_train_loader_batch.__next__()[1]
    data_time.update(time.time() - end)

    target_source_temp = target_source + args.num_classes
    target_source_temp = target_source_temp.cuda(async=True)
    target_source_temp_var = torch.autograd.Variable(target_source_temp) #### labels for target classifier

    target_source = target_source.cuda(async=True)
    input_source_var = torch.autograd.Variable(input_source)
    target_source_var = torch.autograd.Variable(target_source) ######## labels for source classifier.
    ############################################ for source samples
    output_source = model(input_source_var)
    loss_task_s_Cs = criterion(output_source[:,:args.num_classes], target_source_var)
    loss_task_s_Ct = criterion(output_source[:,args.num_classes:], target_source_var)

    loss_domain_st_Cst_part1 = criterion_classifier_source(output_source)
    loss_category_st_G = 0.5 * criterion(output_source, target_source_var) + 0.5 * criterion(output_source, target_source_temp_var)


    input_target_var = torch.autograd.Variable(input_target)
    output_target = model(input_target_var)
    loss_domain_st_Cst_part2 = criterion_classifier_target(output_target)
    loss_domain_st_G = 0.5 * criterion_classifier_target(output_target) + 0.5 * criterion_classifier_source(output_target)
    loss_target_em = criterion_em_target(output_target)

    lam = 2 / (1 + math.exp(-1 * 10 * epoch / args.epochs)) - 1
    if args.flag == 'no_em':
        loss_classifier = loss_task_s_Cs + loss_task_s_Ct + loss_domain_st_Cst_part1 + loss_domain_st_Cst_part2       ### used to classifier
        loss_G = loss_category_st_G + lam * loss_domain_st_G   ### used to feature extractor

    elif args.flag == 'symnet':    #
        loss_classifier = loss_task_s_Cs + loss_task_s_Ct + loss_domain_st_Cst_part1 + loss_domain_st_Cst_part2   ### used to classifier
        loss_G = loss_category_st_G + lam * (loss_domain_st_G + loss_target_em)  ### used to feature extractor

    else:
        raise ValueError('unrecognized flag:', args.flag)

    # mesure accuracy and record loss
    prec1_source, _ = accuracy(output_source.data[:, :args.num_classes], target_source, topk=(1,5))
    prec1_target, _ = accuracy(output_source.data[:, args.num_classes:], target_source, topk=(1,5))
    losses_classifier.update(loss_classifier.data[0], input_source.size(0))
    losses_G.update(loss_G.data[0], input_source.size(0))
    top1_source.update(prec1_source[0], input_source.size(0))
    top1_target.update(prec1_target[0], input_source.size(0))

    #compute gradient and do SGD step
    optimizer.zero_grad()
    loss_classifier.backward(retain_graph=True)
    temp_grad = []
    for param in model.parameters():
        temp_grad.append(param.grad.data.clone())
    grad_for_classifier = temp_grad
    
    optimizer.zero_grad()
    loss_G.backward()
    temp_grad = []
    for param in model.parameters():
        temp_grad.append(param.grad.data.clone())
    grad_for_featureExtractor = temp_grad
    
    count = 0
    for param in model.parameters():
        temp_grad = param.grad.data.clone()
        temp_grad.zero_()
        if count < 159:  ########### the feautre extractor of the ResNet-50
            temp_grad = temp_grad + grad_for_featureExtractor[count]
        else:
            temp_grad = temp_grad + grad_for_classifier[count]
        temp_grad = temp_grad
        param.grad.data = temp_grad
        count = count + 1
    optimizer.step()
    batch_time.update(time.time() - end)
    end = time.time()
    if (epoch + 1) % args.print_freq == 0 or epoch == 0:
        print('Train: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss@C {loss_c.val:.4f} ({loss_c.avg:.4f})\t'
              'Loss@G {loss_g.val:.4f} ({loss_g.avg:.4f})\t'
              'top1S {top1S.val:.3f} ({top1S.avg:.3f})\t'
              'top1T {top1T.val:.3f} ({top1T.avg:.3f})'.format(
               epoch, args.epochs, batch_time=batch_time,
               data_time=data_time, loss_c=losses_classifier, loss_g=losses_G, top1S=top1_source, top1T=top1_target))
        if new_epoch_flag:
            log = open(os.path.join(args.log, 'log.txt'), 'a')
            log.write("\n")
            log.write("Train:epoch: %d, loss@min: %4f, loss@max: %4f, Top1S acc: %3f, Top1T acc: %3f" % (epoch, losses_classifier.avg, losses_G.avg, top1_source.avg, top1_target.avg))
            log.close()
    
    return source_train_loader_batch, target_train_loader_batch, epoch, new_epoch_flag


def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter()
    losses_source = AverageMeter()
    losses_target = AverageMeter()
    top1_source = AverageMeter()
    top1_target = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target,_) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input) #, volatile=True)
        target_var = torch.autograd.Variable(target) #, volatile=True)
        # compute output
        with torch.no_grad():
            output = model(input_var)
        loss_source = criterion(output[:, :args.num_classes], target_var)
        loss_target = criterion(output[:, args.num_classes:], target_var)
        # measure accuracy and record loss
        prec1_source, _ = accuracy(output.data[:, :args.num_classes], target, topk=(1, 5))
        prec1_target, _ = accuracy(output.data[:, args.num_classes:], target, topk=(1, 5))

        losses_source.update(loss_source.data[0], input.size(0))
        losses_target.update(loss_target.data[0], input.size(0))

        top1_source.update(prec1_source[0], input.size(0))
        top1_target.update(prec1_target[0], input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Test: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'LS {lossS.val:.4f} ({lossS.avg:.4f})\t'
                  'LT {lossT.val:.4f} ({lossT.avg:.4f})\t'
                  'top1S {top1S.val:.3f} ({top1S.avg:.3f})\t'
                  'top1T {top1T.val:.3f} ({top1T.avg:.3f})'.format(
                   epoch, i, len(val_loader), batch_time=batch_time, lossS=losses_source, lossT=losses_target,
                   top1S=top1_source, top1T=top1_target))

    print(' * Top1@S {top1S.avg:.3f} Top1@T {top1T.avg:.3f}'
          .format(top1S=top1_source, top1T=top1_target))
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write("\n")
    log.write("                                    Test:epoch: %d, LS: %4f, LT: %4f, Top1S: %3f, Top1T: %3f" %\
              (epoch, losses_source.avg, losses_target.avg, top1_source.avg, top1_target.avg))
    log.close()
    return max(top1_source.avg, top1_target.avg)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Adjust the learning rate according the epoch"""
    ## annealing strategy 1
    # epoch_total = int(args.epochs / args.test_freq)
    # epoch = int((epoch + 1) / args.test_freq)
    lr = args.lr / pow((1 + 10 * epoch / args.epochs), 0.75)
    lr_pretrain = args.lr * 0.1 / pow((1 + 10 * epoch / args.epochs), 0.75) # 0.001 / pow((1 + 10 * epoch / epoch_total), 0.75)
    ## annealing strategy 2
    # exp = epoch > args.schedule[1] and 2 or epoch > args.schedule[0] and 1 or 0
    # lr = args.lr * (args.gamma ** exp)
    # lr_pretrain = lr * 0.1 #1e-3
    for param_group in optimizer.param_groups:
        if param_group['name'] == 'pre-trained':
            param_group['lr'] = lr_pretrain
        else:
            param_group['lr'] = lr




def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
