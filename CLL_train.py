from CLL.utils_algo import *
from cls_model import Linear
import argparse
import os
from pathlib import Path
from TMP_train import print_write
from dataset_loader import load_taglist, load_datasets
from torch.utils.tensorboard import SummaryWriter
from clip import clip
from utils import step_lr_schedule, accuracy
from tqdm import tqdm

@torch.no_grad()
def test_model(clip_model, model, test_loader, taglist):
    clip_model.eval()
    model.eval()
    final_logits = torch.empty(len(test_loader.dataset), num_classes)
    targs = torch.empty(len(test_loader.dataset))
    pos = 0
    for (imgs, labels) in tqdm(test_loader, desc="Test"):
        imgs = imgs.to(device)
        labels = labels.to(device)
        image_embeds = clip_model.encode_image(imgs).squeeze(-1).float()
        image_embeds = image_embeds.to(device)
        logits = model(image_embeds)

        bs = imgs.shape[0]
        final_logits[pos:pos + bs, :] = logits.cpu()
        # targs[pos:pos + bs, :] = F.one_hot(labels.to(torch.int64), num_classes).cpu() * 2 - 1
        targs[pos:pos + bs] = labels.cpu()
        pos += bs

    # evaluate and record
    top1, top5 = accuracy(final_logits, targs, topk=(1, 5))
    return top1, top5


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='complementary-label learning demo file.',
        usage='Demo with complementary labels.',
        description='CLL with weighted loss',
        epilog='end',
        add_help=True)
    parser.add_argument('-bs', '--batch_size', help='batch_size of ordinary labels.', default=128, type=int)
    parser.add_argument('-me', '--method',
                        help='method type. non_k_softmax: only equation (7) . w_loss: weighted loss.',
                        choices=['w_loss', 'non_k_softmax'], type=str, default='w_loss')
    parser.add_argument('-e', '--epochs', help='number of epochs', type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dataset", type=str,
                        choices=(
                            "CIFAR100", "tiny-imagenet-200", "stanford_cars", "caltech-101", "food-101"),
                        default="CIFAR100")
    parser.add_argument("--input-size", type=int, default=224, choices=(224, 384))
    parser.add_argument("--output-dir", type=str, default="./cll_outputs")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    train_loader, info = load_datasets(
        dataset=args.dataset,
        model_type="clip",
        pattern="train",
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    test_loader, _ = load_datasets(
        dataset=args.dataset,
        model_type="clip",
        pattern="val",
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    clip_model, _ = clip.load('ViT-L/14')
    clip_model.to(device).eval()
    for params in clip_model.parameters():
        params.requires_grad = False
    model = Linear(input_dim=768, output_dim=num_classes)
    model.to(device)
    ce_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([{'params': model.parameters()}], lr=1e-3, weight_decay=0.05)
    tf_writer = SummaryWriter(log_dir=output_dir)

    top1, top5 = 0.0, 0.0
    best_top1 = 0
    best_top5 = 0

    for epoch in range(args.epochs):
        train_loss = 0
        step_lr_schedule(optimizer, epoch, init_lr=1e-3, min_lr=5e-6, decay_rate=0.9)
        model.train()
        for (images, labels, labels_t) in tqdm(train_loader, desc="Train"):
            optimizer.zero_grad()

            mask_T = torch.where(labels_t == 1)[0].to(device)
            mask_F = torch.where(labels_t == 0)[0].to(device)
            imgs = images.to(device)
            labels = labels.to(device)
            image_embeds = clip_model.encode_image(imgs).squeeze(-1).float()
            image_embeds = image_embeds.to(device)
            logits = model(image_embeds)
            logits_T = torch.index_select(logits, dim=0, index=mask_T)
            logits_F = torch.index_select(logits, dim=0, index=mask_F)
            labels_T = torch.index_select(labels, dim=0, index=mask_T)
            labels_F = torch.index_select(labels, dim=0, index=mask_F)

            if len(logits_T) != 0:
                loss_T = ce_loss(logits_T, labels_T)
            else:
                loss_T = 0
            loss_C = chosen_loss_c(f=logits_F, K=num_classes, labels=labels_F, method=args.method)
            loss = loss_T + loss_C
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss.item()

        top1, top5 = test_model(clip_model, model, test_loader, taglist)

        if top1[0] >= best_top1:
            best_top1 = top1[0]
            save_obj = {
                'model': model.state_dict(),
                'epoch': epoch,
                'best_top1': best_top1,
                'top5': top5[0]
            }
            torch.save(save_obj, os.path.join(output_dir, 'checkpoint_best.pth'))
        if top5[0] >= best_top5:
            best_top5 = top5[0]

        with open(summary_file, "a", encoding="utf-8") as f:
            print_write(f,
                        f"Epoch : {epoch}  | top1 : {top1[0]}  | top5 : {top5[0]} | best_top1 : {best_top1}| "
                        f"best_top5 : {best_top5}"
                        )

