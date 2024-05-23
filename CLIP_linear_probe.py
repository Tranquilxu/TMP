import os
from argparse import ArgumentParser
from pathlib import Path
from clip import clip
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from train_clip_tf_labels import print_write
from utils import step_lr_schedule, accuracy
from dataset_loader import load_taglist, divide_labeled_or_not, load_datasets
from cls_model import MLP_clip

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = ArgumentParser()
    # data
    parser.add_argument("--dataset", type=str,
                        choices=("CIFAR100", "tiny-imagenet-200", "stanford_cars", "caltech-101", "food-101"),
                        default="CIFAR100")
    parser.add_argument("--input-size", type=int, default=224, choices=(224, 384))
    # miscellaneous
    parser.add_argument("--output-dir", type=str, default="./lp_outputs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    return args


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

        image_embeds = clip_model.encode_image(imgs)
        image_embeds = image_embeds.to(device).squeeze(-1).float()

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
    args = parse_args()

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

    labeled_dataset, unlabeled_dataset = divide_labeled_or_not(dataset=args.dataset, input_size=args.input_size)
    labeled_loader = DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    test_loader, _ = load_datasets(
        dataset=args.dataset,
        model_type='clip',
        pattern="val",
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    num_classes = len(taglist)
    print("number of taglist = ", num_classes)

    clip_model, _ = clip.load('ViT-L/14')
    clip_model.to(device).eval()
    for params in clip_model.parameters():
        params.requires_grad = False

    model = MLP_clip(input_dim=768, output_dim=num_classes)
    model.to(device)

    ce_loss = torch.nn.CrossEntropyLoss()
    optimizer_model = torch.optim.AdamW([{'params': model.parameters()}], lr=1e-3, weight_decay=0.05)

    top1, top5 = 0.0, 0.0
    best_top1 = 0
    best_top5 = 0

    for epoch in range(args.epochs):
        step_lr_schedule(optimizer_model, epoch, init_lr=1e-3, min_lr=5e-6, decay_rate=0.9)

        torch.cuda.empty_cache()
        model.train()

        for (imgs, labels, _) in tqdm(labeled_loader, desc="Train"):
            optimizer_model.zero_grad()

            imgs = imgs.to(device)
            labels = labels.to(device)

            image_embeds = clip_model.encode_image(imgs)
            image_embeds = image_embeds.to(device).squeeze(-1).float()

            logits = model(image_embeds)
            loss = ce_loss(logits, labels)

            loss.backward()
            optimizer_model.step()

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
