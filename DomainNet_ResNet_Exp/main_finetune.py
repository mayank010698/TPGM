import argparse
import copy
import torch
from torch.utils import data
from datasets import get_loader
from engine_finetune import train_one_epoch
from models import get_model
import os
from datetime import datetime
from utils import (
    accuracy,
    AverageMeter,
    dump_logs,
)
from utils.clip_utils import clip_config
from utils.tpgm import tpgm_trainer
from cifar_loader import get_train_valid_loader, get_test_loader
from robustbench.data import load_cifar10c
from torch.utils.data import TensorDataset
from torchvision import transforms
from cifar10c import get_loaders
from robustbench.utils import load_model



def train(logdir, args):

    dump_logs(logdir, "Let the games begin")
    device = torch.device("cuda")

    CORRUPTIONS = ["shot_noise", "motion_blur", "snow", "pixelate",
               "gaussian_noise", "defocus_blur", "brightness", "fog",
               "zoom_blur", "frost", "glass_blur", "impulse_noise", "contrast",
               "jpeg_compression", "elastic_transform"]
    
    for i in range(14):
        corruption = CORRUPTIONS[i]
        tr_dataset, val_dataset, te_dataset = get_loaders("cifar10c",".",corruption,5)
      
        trainloader = data.DataLoader(
            tr_dataset,
            batch_size=args.batch_size*args.gpu_per_node,
            num_workers=args.n_workers,
        )

        valloader = data.DataLoader(
            val_dataset,
            batch_size=args.batch_size*args.gpu_per_node, 
            num_workers=args.n_workers,
        )       

        n_classes = args.n_classes


        model = load_model("Standard","models","cifar10","corruptions")
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

        # Setup automatic PGM parameters, optimizer, and scheduler
        if bool(args.proj_freq):
            sampler_tpgm = torch.utils.data.RandomSampler(val_dataset)
            pgmloader = torch.utils.data.DataLoader(
                val_dataset,
                sampler=sampler_tpgm,
                batch_size=args.batch_size,
                num_workers=args.n_workers,
                drop_last=False,
                persistent_workers=True,
            )
            tpgm = tpgm_trainer(
                model,
                pgmloader,
                args.norm_mode,
                args.proj_lr,
                args.max_iters,
                exclude_list=["fc.weight","fc.bias"]
            )
        else:
            tpgm = None

        # Setup optimizer, lr_scheduler and loss function
        optimizer_params = {
            "lr": args.lr,
            "weight_decay": 5.0e-4,
            "momentum": 0.9,
            "nesterov": True,
        }
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_params)
        loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

        start_epoch = 0
        best_acc1 = -100.0
        best_model = model

        # ================================ Training ==========================================
        for epoch in range(start_epoch, args.epoch):
            best_acc1, best_model = train_one_epoch(
                args,
                model,
                loss_fn,
                optimizer,
                scheduler,
                tpgm,
                trainloader,
                valloader,
                device,
                logdir,
                epoch,
                best_acc1,
                best_model,
            )
        # ================================ Testing ==========================================
        print("start testing")

        sites = corruption

        loaders = []

        loaders.append(torch.utils.data.DataLoader(te_dataset,batch_size=args.batch_size*args.gpu_per_node*2))


        best_model.eval()
        with torch.no_grad():
            for site, loader in zip(sites, loaders):
                test_top1 = AverageMeter("Acc@1", ":6.2f")
                test_top5 = AverageMeter("Acc@5", ":6.2f")
                for i, (image, target) in enumerate(loader):
                    image = image.to(device)
                    target = target.to(device)
                    logit = best_model(image)

                    acc1, acc5 = accuracy(logit, target, topk=(1, 5))
                    test_top1.update(acc1[0], image.size(0))
                    test_top5.update(acc5[0], image.size(0))
                    if i % args.print_interval == 0:
                        output = "{} test: [{}/{}]".format(
                            site,
                            i,
                            len(loader),
                        )
                        print(output)

                output = "{site} test results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\t".format(
                    site=site,
                    top1=test_top1,
                    top5=test_top5,
                )

                print(output)
                dump_logs(logdir, output + "\n")

def main(args):
    now = datetime.now()
    if args.output_dir is not None:
        logdir = os.path.join(args.output_dir,"{}_{}".format(args.id, now.strftime("%d_%m_%Y_%H_%M_%S")))
    else:
        logdir = "./log/{}_{}".format(args.id,now.strftime("%d_%m_%Y_%H_%M_%S"))
    args.output_dir = logdir
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    print("RUNDIR: {}".format(logdir))
    train(logdir, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--arch",
        nargs="?",
        type=str,
        default="clip_resnet50",
        help="Backbone Architecture",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Specify save directory",
    )
    parser.add_argument(
        "--load_pretrained",
        type=str,
        default=None,
        help="pretrained model direcotry",
    )
    parser.add_argument(
        "--id",
        nargs="?",
        type=str,
        default=None,
        help="Additional run information",
    )
    parser.add_argument(
        "--epoch",
        default=200,
        type=int,
        help="training epoch",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * gpus",
    )
    parser.add_argument(
        "--gpu_per_node", default=1, type=int, help="Number of gpus per node"
    )
    parser.add_argument(
        "--print_interval",
        default=3,
        type=int,
        help="print interval",
    )
    parser.add_argument(
        "--val_freq",
        default=1,
        type=int,
        help="Validation interval",
    )
    parser.add_argument(
        "--n_workers",
        default=1,
        type=int,
        help="number of workers",
    )
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )

    # dataset parameters
    parser.add_argument(
        "--dataset",
        nargs="?",
        type=str,
        default="domainnet",
        help="Dataset name",
    )
    parser.add_argument(
        "--meta_dir",
        nargs="?",
        type=str,
        default= "./datasets/",
        help="Dataset meta information directory",
    )
    parser.add_argument(
        "--data_dir",
        nargs="?",
        type=str,
        default="/datasets/domainnet",
        help="Image data directory",
    )
    parser.add_argument(
        "--site",
        nargs="?",
        type=str,
        default="real",
        help="DomainNet site",
    )
    parser.add_argument(
        "--percent",
        nargs="?",
        type=str,
        default="5",
        help="DomainNet percentage",
    )
    parser.add_argument(
        "--n_classes",
        nargs="?",
        type=int,
        default=345,
        help="Number of classes",
    )

    # optimizer
    parser.add_argument(
        "--optimizer",
        nargs="?",
        type=str,
        default="sgd",
        help="optimizer type",
    )
    parser.add_argument(
        "--lr",
        default=None,
        type=float,
        help="Custom Learning Rate",
    )

    parser.add_argument(
        "--load_ft",
        nargs="?",
        type=str,
        default=None,
        help="load fine-tuned checkpoint",
    )

    # projection hyperparameters
    parser.add_argument("--proj_freq", default=0, type=int)
    parser.add_argument("--max_iters", default=1, type=int)
    parser.add_argument(
        "--proj_lr",
        default=1e-2,
        type=float,
        help="Custom Project Learning Rate",
    )
    parser.add_argument("--norm_mode", default="mars", type=str)
    parser.add_argument(
        "--mu", default=0.0, type=float, help="Hyperparamter for TV smoothing"
    )

    args = parser.parse_args()
    main(args)
