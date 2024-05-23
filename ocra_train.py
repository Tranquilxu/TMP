import argparse
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
import torch
from train_clip_tf_labels import print_write
from utils import step_lr_schedule
from dataset_loader import load_taglist, load_datasets
from clip import clip
from ocra.utils import cluster_acc, AverageMeter, entropy, MarginLoss, accuracy
from sklearn import metrics
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle
from cls_model import MLP_clip
from ocra.ocra_dataloader import divide_labeled_or_not
import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def train(args, clip_model, model, device, train_label_loader, train_unlabel_loader, optimizer, m, epoch, tf_writer):
    clip_model.eval()
    model.train()
    bce = nn.BCELoss()
    m = min(m, 0.5)
    ce = MarginLoss(m=-1 * m)
    unlabel_loader_iter = cycle(train_unlabel_loader)
    bce_losses = AverageMeter('bce_loss', ':.4e')
    ce_losses = AverageMeter('ce_loss', ':.4e')
    entropy_losses = AverageMeter('entropy_loss', ':.4e')

    for batch_idx, ((x, x2), target, _) in enumerate(train_label_loader):

        ((ux, ux2), _, _) = next(unlabel_loader_iter)

        x = torch.cat([x, ux], 0)
        x2 = torch.cat([x2, ux2], 0)
        # labeled_len = len(target)

        x, x2, target = x.to(device), x2.to(device), target.to(device)
        optimizer.zero_grad()

        image_embeds = clip_model.encode_image(x)
        image_embeds = image_embeds.to(device).squeeze(-1).float()
        output = model(image_embeds)
        image_embeds2 = clip_model.encode_image(x2)
        image_embeds2 = image_embeds2.to(device).squeeze(-1).float()
        output2 = model(image_embeds2)

        # output, feat = model(x)
        # output2, feat2 = model(x2)
        prob = F.softmax(output, dim=1)
        prob2 = F.softmax(output2, dim=1)

        # feat_detach = feat.detach()
        feat_detach = image_embeds.detach()
        feat_norm = feat_detach / torch.norm(feat_detach, 2, 1, keepdim=True)
        cosine_dist = torch.mm(feat_norm, feat_norm.t())
        labeled_len = len(target)

        pos_pairs = []
        target_np = target.cpu().numpy()

        # label part
        for i in range(labeled_len):
            target_i = target_np[i]
            idxs = np.where(target_np == target_i)[0]
            if len(idxs) == 1:
                pos_pairs.append(idxs[0])
            else:
                selec_idx = np.random.choice(idxs, 1)
                while selec_idx == i:
                    selec_idx = np.random.choice(idxs, 1)
                pos_pairs.append(int(selec_idx))

        # unlabel part
        unlabel_cosine_dist = cosine_dist[labeled_len:, :]
        vals, pos_idx = torch.topk(unlabel_cosine_dist, 2, dim=1)
        pos_idx = pos_idx[:, 1].cpu().numpy().flatten().tolist()
        pos_pairs.extend(pos_idx)

        pos_prob = prob2[pos_pairs, :]
        pos_sim = torch.bmm(prob.view(args.batch_size, 1, -1), pos_prob.view(args.batch_size, -1, 1)).squeeze()
        ones = torch.ones_like(pos_sim)
        bce_loss = bce(pos_sim, ones)
        ce_loss = ce(output[:labeled_len], target)
        entropy_loss = entropy(torch.mean(prob, 0))

        loss = - entropy_loss + ce_loss + bce_loss

        bce_losses.update(bce_loss.item(), args.batch_size)
        ce_losses.update(ce_loss.item(), args.batch_size)
        entropy_losses.update(entropy_loss.item(), args.batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    tf_writer.add_scalar('loss/bce', bce_losses.avg, epoch)
    tf_writer.add_scalar('loss/ce', ce_losses.avg, epoch)
    tf_writer.add_scalar('loss/entropy', entropy_losses.avg, epoch)


def test(args, clip_model, model, labeled_num, device, test_loader, epoch, tf_writer):
    clip_model.eval()
    model.eval()
    preds = np.array([])
    targets = np.array([])
    confs = np.array([])
    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(test_loader):
            x, label = x.to(device), label.to(device)
            image_embeds = clip_model.encode_image(x)
            image_embeds = image_embeds.to(device).squeeze(-1).float()
            output = model(image_embeds)
            # output, _ = model(x)
            prob = F.softmax(output, dim=1)
            conf, pred = prob.max(1)
            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())
            confs = np.append(confs, conf.cpu().numpy())
    targets = targets.astype(int)
    preds = preds.astype(int)

    seen_mask = targets < labeled_num
    unseen_mask = ~seen_mask
    overall_acc = cluster_acc(preds, targets)
    seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
    unseen_acc = cluster_acc(preds[unseen_mask], targets[unseen_mask])
    unseen_nmi = metrics.normalized_mutual_info_score(targets[unseen_mask], preds[unseen_mask])
    mean_uncert = 1 - np.mean(confs)
    # print('Test overall acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}'.format(overall_acc, seen_acc, unseen_acc))
    print('Epoch {}: Test overall acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}'.format(epoch, overall_acc, seen_acc, unseen_acc))
    tf_writer.add_scalar('acc/overall', overall_acc, epoch)
    tf_writer.add_scalar('acc/seen', seen_acc, epoch)
    tf_writer.add_scalar('acc/unseen', unseen_acc, epoch)
    tf_writer.add_scalar('nmi/unseen', unseen_nmi, epoch)
    tf_writer.add_scalar('uncert/test', mean_uncert, epoch)
    return mean_uncert, overall_acc


def main():
    parser = argparse.ArgumentParser(description='orca')
    parser.add_argument("--dataset", type=str,
                        choices=("CIFAR100", "tiny-imagenet-200", "stanford_cars", "caltech-101", "food-101"),
                        default="CIFAR100")
    parser.add_argument("--input-size", type=int, default=224, choices=(224, 384))
    # miscellaneous
    parser.add_argument("--output-dir", type=str, default="./ocra_clip_outputs")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

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
    # print("number of taglist = ", num_classes)

    labeled_dataset, unlabeled_dataset = divide_labeled_or_not(dataset=args.dataset, input_size=args.input_size)
    labeled_len = len(labeled_dataset)
    unlabeled_len = len(unlabeled_dataset)
    labeled_batch_size = int(args.batch_size * labeled_len / (labeled_len + unlabeled_len))

    labeled_ratio = labeled_len / (labeled_len + unlabeled_len)
    # print("labeled ratio = ", labeled_ratio)

    label_list = [labeled_dataset.dataset.Y[i] for i in labeled_dataset.indices]
    labeled_classes = list(set(label_list))
    labeled_num = len(labeled_classes)
    all_classes = set(range(num_classes))
    unlabeled_classes = list(all_classes - set(labeled_classes))

    with open(summary_file, "w", encoding="utf-8") as f:
        print_write(f, f"number of taglist = {num_classes}")
        print_write(f, f"labeled ratio = {labeled_ratio}")
        print_write(f, f"number of labeled classes = {labeled_num}")
        print_write(f, f"unlabeled classes : {unlabeled_classes}")

    train_label_loader = DataLoader(labeled_dataset, batch_size=labeled_batch_size, shuffle=True, drop_last=True,
                                    num_workers=args.num_workers)
    train_unlabel_loader = DataLoader(unlabeled_dataset, batch_size=args.batch_size - labeled_batch_size, shuffle=True,
                                      drop_last=True, num_workers=args.num_workers)
    test_loader, _ = load_datasets(
        dataset=args.dataset,
        model_type='clip',
        pattern="val",
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    clip_model, _ = clip.load('ViT-L/14')
    clip_model.to(device).eval()
    for params in clip_model.parameters():
        params.requires_grad = False

    model = MLP_clip(input_dim=768, output_dim=num_classes)
    model.to(device)

    optimizer = torch.optim.AdamW([{'params': model.parameters()}], lr=1e-3, weight_decay=0.05)

    tf_writer = SummaryWriter(log_dir=output_dir)
    acc = 0

    for epoch in range(args.epochs):
        step_lr_schedule(optimizer, epoch, init_lr=1e-3, min_lr=5e-6, decay_rate=0.9)
        torch.cuda.empty_cache()
        mean_uncert, overall_acc = test(args, clip_model, model, labeled_num, device, test_loader, epoch, tf_writer)
        train(args, clip_model, model, device, train_label_loader, train_unlabel_loader, optimizer, mean_uncert, epoch,
              tf_writer)

        if overall_acc > acc:
            acc = overall_acc
            print('best_top1:', acc)
            save_obj = {
                'model': model.state_dict(),
                'epoch': epoch,
                'best_top1': acc,
            }
            torch.save(save_obj, os.path.join(output_dir, 'checkpoint_best.pth'))

    tf_writer.close()


if __name__ == '__main__':
    main()
