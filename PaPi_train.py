import argparse
import time
from pathlib import Path
import torch
from TMP_train import print_write
from dataset_loader import load_taglist
import numpy as np
from PaPi.Papi_dataloader import load_data
from PaPi.utils_loss import PaPiLoss
from PaPi.utils import AverageMeter, ProgressMeter, adjust_learning_rate, process
from PaPi.Papi_model import PaPi, PaPiNet
from utils import accuracy

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(description='PaPi')
    parser.add_argument("--dataset", type=str,
                        choices=(
                            "CIFAR100", "tiny-imagenet-200", "stanford_cars", "caltech-101", "food-101"),
                        default="CIFAR100")
    parser.add_argument("--input-size", type=int, default=224, choices=(224, 384))
    # miscellaneous
    parser.add_argument("--output-dir", type=str, default="./PaPi_outputs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)

    parser.add_argument('--alpha_weight', default=1.0, type=float,
                        help='contrastive loss weight')
    parser.add_argument('--alpha_mixup', type=float, default=8.0,
                        help='alpha for beta distribution')
    parser.add_argument('--low-dim', default=128, type=int,
                        help='embedding dimension')
    parser.add_argument('--pseudo_label_weight_range', default='0.95, 0.8', type=str,
                        help='pseudo target updating coefficient')
    parser.add_argument('--pro_weight_range', default='0.9, 0.5', type=str,
                        help='prototype updating coefficient')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-lr_decay_epochs', type=str, default='99,199,299',
                        help='where to decay lr, can be a list')
    parser.add_argument('-lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--cosine', action='store_true', default=False,
                        help='use cosine lr schedule')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--proto_m', default=0.99, type=float,
                        help='momentum for computing the momving average of prototypes')
    parser.add_argument('--tau_proto', type=float, default=0.3,
                        help='temperature for prototype')

    args = parser.parse_args()
    args.pseudo_label_weight_range = [float(item) for item in args.pseudo_label_weight_range.split(',')]
    args.pro_weight_range = [float(item) for item in args.pro_weight_range.split(',')]
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    args.cuda = torch.cuda.is_available()

    # set up output paths
    output_dir = args.output_dir + "/" + args.dataset + "/" + "Train"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pred_file, pr_file, ap_file, summary_file, logit_file = [
        output_dir + "/" + name for name in
        ("pred.txt", "pr.txt", "ap.txt", "summary.txt", "logits.pth")
    ]
    with open(summary_file, "w", encoding="utf-8") as f:
        print_write(f, "****************")
        for key in (
                "dataset", "input_size",
                "output_dir", "batch_size", "num_workers"
        ):
            print_write(f, f"{key}: {getattr(args, key)}")
        print_write(f, "****************")

    info = load_taglist(dataset=args.dataset)
    taglist = info["taglist"]
    num_classes = len(taglist)
    args.num_class = num_classes

    train_loader, train_partialY_matrix, test_loader = load_data(args)

    with open(summary_file, "w", encoding="utf-8") as f:
        print_write(f, f"number of taglist = {num_classes}")
        print_write(f, 'Average candidate num: {}'.format(train_partialY_matrix.sum(1).mean()))

    model = PaPi(args, PaPiNet)
    model.to(device)

    optimizer_fc = torch.optim.SGD(model.encoder.fc.parameters(), lr=args.lr, momentum=args.momentum,
                                   weight_decay=args.weight_decay)
    optimizer_head = torch.optim.SGD(model.encoder.head.parameters(), lr=args.lr, momentum=args.momentum,
                                     weight_decay=args.weight_decay)

    tempY = train_partialY_matrix.sum(dim=1).unsqueeze(1).repeat(1, train_partialY_matrix.shape[1])
    uniform_confidence = train_partialY_matrix.float() / tempY
    uniform_confidence = uniform_confidence.cuda()

    loss_PaPi_func = PaPiLoss(predicted_score_cls=uniform_confidence, pseudo_label_weight=0.99)

    sim_criterion = torch.nn.KLDivLoss(reduction='batchmean').cuda()

    best_acc = 0

    for epoch in range(args.epochs):
        is_best = False
        adjust_learning_rate(args, optimizer_fc, epoch)
        adjust_learning_rate(args, optimizer_head, epoch)
        torch.cuda.empty_cache()
        loss_PaPi_func.set_alpha(epoch, args)
        acc_train_cls, loss_cls_log, loss_PaPi_log = train(train_loader, model, loss_PaPi_func,
                                                           optimizer_fc, optimizer_head, epoch, args,
                                                           sim_criterion, summary_file)
        loss_PaPi_func.set_pseudo_label_weight(epoch, args)
        model.set_prototype_update_weight(epoch, args)

        acc_test, _ = test(model, test_loader, args, epoch, summary_file)

        if acc_test.item() > best_acc:
            best_acc = acc_test.item()
            print('best_top1:', best_acc)
            save_obj = {
                'model': model.state_dict(),
                'epoch': epoch,
                'best_top1': best_acc,
            }
            torch.save(save_obj, os.path.join(output_dir, 'checkpoint_best.pth'))

        with open(summary_file, "w", encoding="utf-8") as f:
            print_write(f, 'Epoch {}: Train_Acc {}, Test_Acc {}, Best_Acc {}.'.format(
                epoch, acc_train_cls.avg, acc_test.item(), best_acc))


def train(train_loader, model, loss_PaPi_func, optimizer_fc, optimizer_head, epoch, args, sim_criterion, summary_file):
    batch_time = AverageMeter('Time', ':1.2f')
    data_time = AverageMeter('Data', ':1.2f')
    acc_cls = AverageMeter('Acc@Cls', ':2.2f')
    loss_cls_log = AverageMeter('Loss@cls', ':2.2f')
    loss_PaPi_log = AverageMeter('Loss@PaPi', ':2.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, acc_cls, loss_cls_log, loss_PaPi_log],
        prefix="Epoch: [{}]".format(epoch)
    )
    model.encoder.encoder.eval()
    model.encoder.fc.train()
    model.encoder.head.train()

    end = time.time()

    for i, (images_1, images_2, labels, true_labels, index) in enumerate(train_loader):

        data_time.update(time.time() - end)

        X_1, X_2, Y, index = images_1.cuda(), images_2.cuda(), labels.cuda(), index.cuda()

        Y_true = true_labels.long().detach().cuda()

        Lambda = np.random.beta(args.alpha_mixup, args.alpha_mixup)
        batch_size = X_1.shape[0]
        # idx_rp = torch.randperm(args.batch_size)
        idx_rp = torch.randperm(batch_size)
        X_1_rp = X_1[idx_rp]
        X_2_rp = X_2[idx_rp]
        Y_rp = Y[idx_rp]

        X_1_mix = Lambda * X_1 + (1 - Lambda) * X_1_rp
        X_2_mix = Lambda * X_2 + (1 - Lambda) * X_2_rp
        Y_mix = Lambda * Y + (1 - Lambda) * Y_rp

        cls_out_1, cls_out_2, logits_prot_1, logits_prot_2, logits_prot_1_mix, logits_prot_2_mix = \
            model(img_q=X_1, img_k=X_2, img_q_mix=X_1_mix, img_k_mix=X_2_mix, partial_Y=Y, Y_true=Y_true, args=args)

        batch_size = cls_out_1.shape[0]

        loss_PaPi_func.update_weight_byclsout1(cls_predicted_score=cls_out_1, batch_index=index, batch_partial_Y=Y,
                                               args=args)

        cls_loss_1, sim_loss_2, alpha_td = loss_PaPi_func(cls_out_1, cls_out_2, logits_prot_1, logits_prot_2,
                                                          logits_prot_1_mix, logits_prot_2_mix, idx_rp, Lambda, index,
                                                          args, sim_criterion)

        loss = cls_loss_1 + alpha_td * sim_loss_2

        loss_cls_log.update(cls_loss_1.item())
        loss_PaPi_log.update(loss.item())

        acc = accuracy(cls_out_1, Y_true)
        acc_cls.update(acc[0].item())

        optimizer_fc.zero_grad()
        optimizer_head.zero_grad()
        loss.backward()
        optimizer_fc.step()
        optimizer_head.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (i % 50 == 0) or ((i + 1) % len(train_loader) == 0):
            with open(summary_file, "w", encoding="utf-8") as f:
                print_write(f, 'Epoch:[{0}][{1}/{2}]\t'
                               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                               'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                               'A_cls {Acc_cls.val:.4f} ({Acc_cls.avg:.4f})\t'
                               'L_cls {Loss_cls.val:.4f} ({Loss_cls.avg:.4f})\t'
                               'L_all {Loss_PaPi.val:.4f} ({Loss_PaPi.avg:.4f})\t'.format(
                    epoch, i + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, Acc_cls=acc_cls,
                    Loss_cls=loss_cls_log, Loss_PaPi=loss_PaPi_log
                )
                            )
    return acc_cls, loss_cls_log, loss_PaPi_log


def test(model, test_loader, args, epoch, summary_file):
    with torch.no_grad():

        print('\n=====> Evaluation...\n')
        model.eval()

        top1_acc = AverageMeter("Top1")
        top5_acc = AverageMeter("Top5")

        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.cuda(), labels.cuda()

            outputs, outputs_pro = model(img_q=images, args=args, eval_only=True)

            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

            top1_acc.update(acc1[0])
            top5_acc.update(acc5[0])

            if (batch_idx % 10 == 0) or ((batch_idx + 1) % len(test_loader) == 0):
                with open(summary_file, "w", encoding="utf-8") as f:
                    print_write(f,
                                'Test:[{0}/{1}]\t'
                                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t' \
                                .format(batch_idx + 1, len(test_loader), top1=top1_acc, top5=top5_acc)
                                )

        # average across all processes
        acc_tensors = torch.Tensor([top1_acc.avg, top5_acc.avg]).cuda()

        # dist.all_reduce(acc_tensors)
        # acc_tensors /= args.world_size

        with open(summary_file, "w", encoding="utf-8") as f:
            print_write(f, 'Top1 Accuracy is %.2f%%, Top5 Accuracy is %.2f%%\n' % (acc_tensors[0], acc_tensors[1]))

    return acc_tensors[0], acc_tensors[1]


if __name__ == '__main__':
    main()
