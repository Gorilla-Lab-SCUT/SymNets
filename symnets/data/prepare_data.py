import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from data.folder_new import ImageFolder_new


def generate_dataloader(args):
    # Data loading code
    traindir_source = os.path.join(args.data_path_source, args.src)
    traindir_target = os.path.join(args.data_path_source_t, args.src_t)
    valdir = os.path.join(args.data_path_target, args.tar)
    if not os.path.isdir(traindir_source):
        # split_train_test_images(args.data_path)
        raise ValueError('Null path of source train data!!!')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    source_train_dataset = datasets.ImageFolder(
        traindir_source,
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    source_train_loader = torch.utils.data.DataLoader(
        source_train_dataset, batch_size=args.batch_size_s, shuffle=True,
        drop_last=True, num_workers=args.workers, pin_memory=True, sampler=None
    )

    source_val_dataset = ImageFolder_new(
        traindir_source,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )
    source_val_loader = torch.utils.data.DataLoader(
        source_val_dataset, batch_size=args.batch_size_s, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None
    )
    
    target_train_dataset = datasets.ImageFolder(
        traindir_target,
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    target_train_loader = torch.utils.data.DataLoader(
        target_train_dataset, batch_size=args.batch_size_t, shuffle=True,
        drop_last=True, num_workers=args.workers, pin_memory=True, sampler=None
    )
    target_val_loader = torch.utils.data.DataLoader(
        ImageFolder_new(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size_t, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    return source_train_loader, source_val_loader, target_train_loader, target_val_loader

