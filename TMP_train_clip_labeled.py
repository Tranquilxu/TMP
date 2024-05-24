import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, TextIO, Tuple
from clip import clip
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import Module
from tqdm import tqdm
from utils import step_lr_schedule, accuracy
from dataset_loader_clip import load_datasets, Generate_data
from cls_model import Linear, image_input

from monai.utils import set_determinism

device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 1
set_determinism(seed=seed)


def parse_args():
    parser = ArgumentParser()
    # data
    parser.add_argument("--dataset", type=str,
                        choices=("CIFAR100", "tiny-imagenet-200", "EuroSAT", "stanford_cars", "caltech-101", "food-101"),
                        default="food-101")
    parser.add_argument("--input-size", type=int, default=224, choices=(224, 384))
    # threshold

    # miscellaneous
    parser.add_argument("--output-dir", type=str, default="./tfl_clip_outputs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)

    args = parser.parse_args()

    # post process and validity check
    args.model_type = 'clip'
    return args


single_template = ["a photo of a {}."]


def article(name):
    return "an" if name[0] in "aeiou" else "a"


def processed_name(name, rm_dot=False):
    # _ for lvis
    # / for obj365
    res = name.replace("_", " ").replace("/", " or ").lower()
    if rm_dot:
        res = res.rstrip(".")
    return res


def load_taglist(
        dataset: str
) -> Tuple[Dict]:
    dataset_root = "./datasets/" + dataset

    tag_file = dataset_root + f"/{dataset}_ram_taglist.txt"

    with open(tag_file, "r", encoding="utf-8") as f:
        taglist_or = [line.strip() for line in f]
    taglist = taglist_or

    info = {"taglist": taglist}
    return info


def build_clip_label_embedding(model, categories):
    # print("Creating pretrained CLIP image model")
    templates = single_template
    run_on_gpu = torch.cuda.is_available()

    with torch.no_grad():
        openset_label_embedding = []
        for category in categories:
            # print("category =", category)
            texts = [
                template.format(
                    processed_name(category, rm_dot=True), article=article(category)
                )
                for template in templates
            ]
            texts = [
                "This is " + text if text.startswith("a") or text.startswith("the") else text
                for text in texts
            ]  
            texts = clip.tokenize(texts)  # tokenize
            # print("texts =", texts.size())
            if run_on_gpu:
                texts = texts.cuda()
                model = model.cuda()
            text_embeddings = model.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            openset_label_embedding.append(text_embedding)
        openset_label_embedding = torch.stack(openset_label_embedding, dim=1)
        if run_on_gpu:
            openset_label_embedding = openset_label_embedding.cuda()

    openset_label_embedding = openset_label_embedding.t()
    return openset_label_embedding


def load_clip() -> Module:
    model, _ = clip.load("ViT-L/14")
    return model.to(device).eval()


def print_write(f: TextIO, s: str):
    print(s)
    f.write(s + "\n")


@torch.no_grad()
def test_model(clip_model, model, test_loader, taglist, x_img, learned_img_epoch):
    clip_model.eval()
    model.eval()
    x_img.eval()

    # inference
    final_logits = torch.empty(len(test_loader.dataset), num_classes)
    targs = torch.empty(len(test_loader.dataset))
    pos = 0
    input_image = torch.ones(1, 1, 224, 224).to(device)
    label_embed_or = build_clip_label_embedding(clip_model, taglist)
    learned_text_epoch = learn_text(text_tag_num, clip_model.encode_image(x_img(input_image)).to(device),
                                    label_embed_or, taglist, clip_model, dataset_name=args.dataset)
    for (imgs, labels) in tqdm(test_loader, desc="Test"):
        imgs = imgs.to(device)
        labels = labels.to(device)
        learned_text = learned_text_epoch.repeat(imgs.size()[0], 1, 1).to(device)
        learned_img = learned_img_epoch.repeat(imgs.size()[0], 1, 1).to(device)
        label_embed_input = clip_model.encode_image(x_img(input_image)).repeat(imgs.size()[0], 1, 1).to(device)

        image_embeds = clip_model.encode_image(imgs).unsqueeze(1)
        image_embeds = image_embeds.to(device)

        feature = torch.flatten(torch.cat((image_embeds, label_embed_input, learned_text, learned_img), dim=1), 1,
                                -1).float()
        logits = model(feature)

        bs = imgs.shape[0]

        final_logits[pos:pos + bs, :] = logits.cpu()
        targs[pos:pos + bs] = labels.cpu()
        pos += bs

    # evaluate and record
    top1, top5 = accuracy(final_logits, targs, topk=(1, 5))
    return top1, top5


@torch.no_grad()
def zero_shot_test_model(clip_model, test_loader, taglist):
    clip_model.eval()

    # inference
    final_logits = torch.empty(len(test_loader.dataset), num_classes)
    targs = torch.empty(len(test_loader.dataset))
    pos = 0
    label_embed_or = build_clip_label_embedding(clip_model, taglist)
    for (imgs, labels) in tqdm(test_loader, desc="Test"):
        imgs = imgs.to(device)
        labels = labels.to(device)

        image_embeds = clip_model.encode_image(imgs).unsqueeze(1)
        image_embeds = image_embeds.to(device)

        label_embed = label_embed_or.repeat(imgs.size()[0], 1, 1)
        label_embed = label_embed.to(device)

        image_to_label = image_embeds.repeat(1, num_classes, 1)

        output = cos(image_to_label, label_embed)

        _, labels_g = torch.max(output, dim=1)

        bs = imgs.shape[0]
        labels_g = F.one_hot(labels_g.to(torch.int64), num_classes=num_classes).cuda() * 2 - 1
        final_logits[pos:pos + bs, :] = labels_g.cpu()

        targs[pos:pos + bs] = labels.cpu()
        pos += bs

    # evaluate and record
    top1, top5 = accuracy(final_logits, targs, topk=(1, 5))
    return top1, top5


def RC_loss(logits_F, labels_F, output):
    m_loss = nn.CrossEntropyLoss(reduction='none').to(device)
    loss = 0.0

    for i in range(num_classes):
        value = m_loss(logits_F, torch.ones(labels_F.size()).long().to(device) * i)

        loss = loss + torch.inner(value.float(), output[:, i].float())

    loss_rf = m_loss(logits_F, labels_F)

    loss_tf = 0
    for j in range(len(labels_F)):
        loss_tf = loss_tf + output[j, labels_F[j]].float() * loss_rf[j]

    loss = loss - loss_tf
    return loss


def RC_train_model(imgs, labels, labels_t, model, image_embed_input, label_embed_label, ce_loss, cos, soft, epoch,
                   learned_text, learned_img):

    mask_T = torch.where(labels_t == 1)[0].to(device)
    mask_F = torch.where(labels_t == 0)[0].to(device)

    imgs = imgs.to(device)
    labels = labels.to(device)

    image_embeds = clip_model.encode_image(imgs).unsqueeze(1)
    image_embeds = image_embeds.to(device)


    feature = torch.flatten(torch.cat((image_embeds, image_embed_input, learned_text, learned_img), dim=1), 1,
                            -1).float()


    logits = model(feature)


    logits_T = torch.index_select(logits, dim=0, index=mask_T)
    logits_F = torch.index_select(logits, dim=0, index=mask_F)
    labels_T = torch.index_select(labels, dim=0, index=mask_T)
    labels_F = torch.index_select(labels, dim=0, index=mask_F)



    image_embeds_F = torch.index_select(image_embeds, dim=0, index=mask_F)
    image_embeds_F = image_embeds_F.to(device)

    image_to_label_F = image_embeds_F.repeat(1, num_classes, 1)
    label_embed_F_l = label_embed_label.repeat(image_embeds_F.size()[0], 1, 1)

    output = cos(image_to_label_F, label_embed_F_l)
    tau = torch.ones(output.size()[1]).to(device) * 0.001
    output = soft(output / tau)
    tau = torch.ones(logits_F.size()[1]).to(device) * 100000
    output_f = logits_F.detach()
    output_f = soft(output_f / tau)
    lambda_a = 0.1
    output = lambda_a * output_f + (1-lambda_a) * output

    if len(logits_T) != 0:
        loss_T = ce_loss(logits_T, labels_T)
    else:
        loss_T = 0
    loss_F = RC_loss(logits_F, labels_F, output)
    loss = loss_T + loss_F
    return loss


def learn_text(text_tag_num, label_embed_or, label_embed_ot, taglist, clip_model, dataset_name):
    with open(f"./datasets/{dataset_name}/learned_taglist.txt", "w") as f:
        f.write("")
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    taglist_new = []

    out = label_embed_or.repeat(label_embed_ot.size()[0], 1)

    output = cos(out, label_embed_ot)
    value, index = torch.topk(output, k=int(num_classes * 0.6))

    for j in range(len(index)):
        taglist_new.append(taglist[index[j]])
    taglist_new = list(set(taglist_new))
    for i in range(text_tag_num):

        file = open(f"./datasets/{dataset_name}/learned_taglist.txt", "a")

        file.write(taglist_new[i])
        if i < len(taglist_new) - 1:
            file.write("\n")
        file.close()
    with open(f"./datasets/{dataset_name}/learned_taglist.txt", "r", encoding="utf-8") as f:
        learned_taglist = [line.strip() for line in f]

    return build_clip_label_embedding(clip_model, learned_taglist)


def learn_img(img_tag_num, label_embed_or, clip_model, input_size, sampler, dataset_name):
    random_num = 1000
    if random_num >= len(sampler):
        random_num = int(len(sampler) * 0.1)
    gen_data = Generate_data(input_size, sampler, random_num, dataset_name).to(device)

    label_embed_gen = clip_model.encode_image(gen_data).to(device)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)


    out = label_embed_or.repeat(gen_data.size()[0], 1)

    output = cos(out, label_embed_gen)
    value, index = torch.topk(output, k=40)

    img_embed = torch.ones(img_tag_num, len(label_embed_gen[1]))
    for j in range(img_tag_num):
        img_embed[j] = label_embed_gen[index[j]]

    return img_embed


if __name__ == "__main__":
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
                "model_type",
                "dataset", "input_size",
                "output_dir", "batch_size", "num_workers"
        ):
            print_write(f, f"{key}: {getattr(args, key)}")
        print_write(f, "****************")

    # prepare data

    info = load_taglist(dataset=args.dataset)
    taglist_label = info["taglist"]

    train_loader, info = load_datasets(
        dataset=args.dataset,
        model_type=args.model_type,
        pattern="train",
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    taglist = info["taglist"]

    test_loader, _ = load_datasets(
        dataset=args.dataset,
        model_type=args.model_type,
        pattern="val",
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    if args.model_type == "clip":
        clip_model = load_clip()

    # freeze
    for params in clip_model.parameters():
        params.requires_grad = False

    num_classes = len(taglist)
    print("number of taglist = ", num_classes)
    sampler = torch.randperm(len(test_loader.dataset))

    text_tag_num = 15
    if text_tag_num >= num_classes:
        text_tag_num = int(num_classes * 0.2)
    img_tag_num = 5

    model = Linear(input_dim=768 * (text_tag_num + img_tag_num + 2), output_dim=num_classes)

    x_img = image_input().to(device)

    for params in model.parameters():
        params.requires_grad = True
    for params in x_img.parameters():
        params.requires_grad = True

    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    soft = nn.Softmax(dim=1)
    ce_loss = torch.nn.CrossEntropyLoss()

    optimizer_img = torch.optim.AdamW([{'params': x_img.parameters()}], lr=1e-3,
                                      weight_decay=0.8)
    optimizer_model = torch.optim.AdamW([{'params': model.parameters()}], lr=1e-3,
                                        weight_decay=0.05)

    clip_model.to(device)
    model.to(device)

    top1, top5 = 0.0, 0.0
    best_top1 = 0
    best_top5 = 0


    learned_label_embed = build_clip_label_embedding(clip_model, taglist)

    label_embed_label = build_clip_label_embedding(clip_model, taglist_label)
    input_image = torch.ones(1, 1, 224, 224).to(device)


    for epoch in range(args.epochs):
        step_lr_schedule(optimizer_model, epoch, init_lr=1e-3, min_lr=5e-6, decay_rate=0.9)
        step_lr_schedule(optimizer_img, epoch, init_lr=8e-2, min_lr=5e-4, decay_rate=0.01)
        torch.cuda.empty_cache()
        model.train()
        x_img.train()
        clip_model.eval()
        learned_text_epoch = learn_text(text_tag_num, clip_model.encode_image(x_img(input_image)).to(device),
                                        learned_label_embed,
                                        taglist, clip_model, dataset_name=args.dataset)
        learned_img_epoch = learn_img(img_tag_num, clip_model.encode_image(x_img(input_image)).to(device), clip_model,
                                      args.input_size, sampler, dataset_name=args.dataset)

        for (imgs, labels, labels_t) in tqdm(train_loader, desc="Train"):
            optimizer_model.zero_grad()
            learned_text = learned_text_epoch.repeat(imgs.size()[0], 1, 1).to(device)
            learned_img = learned_img_epoch.repeat(imgs.size()[0], 1, 1).to(device)
            image_embed_input = clip_model.encode_image(x_img(input_image)).repeat(imgs.size()[0], 1, 1).to(device)


            loss = RC_train_model(imgs, labels, labels_t, model, image_embed_input, label_embed_label, ce_loss, cos,
                                  soft, epoch, learned_text, learned_img)

            loss.backward()
            optimizer_model.step()
            if epoch % 2 == 0:
                optimizer_img.step()

        # test
        top1, top5 = test_model(clip_model, model, test_loader, taglist, x_img, learned_img_epoch)

        if top1[0] >= best_top1:
            best_top1 = top1[0]
            save_obj = {
                'model': model.state_dict(),
                'x_img': x_img.state_dict(),
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
