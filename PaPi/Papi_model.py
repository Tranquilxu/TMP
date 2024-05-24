import torch
import torch.nn as nn
import torch.nn.functional as F
from clip import clip
from cls_model import Linear


class PaPi(nn.Module):
    def __init__(self, args, base_encoder):
        super().__init__()

        pretrained = False

        self.proto_weight = args.proto_m

        self.encoder = base_encoder(head='linear', feat_dim=args.low_dim, num_classes=args.num_class)

        self.register_buffer("prototypes", torch.zeros(args.num_class, args.low_dim))

    def set_prototype_update_weight(self, epoch, args):
        start = args.pro_weight_range[0]
        end = args.pro_weight_range[1]
        self.proto_weight = 1. * epoch / args.epochs * (end - start) + start

    def forward(self, img_q, img_k=None, img_q_mix=None, img_k_mix=None, partial_Y=None, Y_true=None, args=None,
                eval_only=False):

        output_q, q = self.encoder(img_q)

        if eval_only:
            prototypes_eval = self.prototypes.clone().detach()
            logits_prot_test = torch.mm(q, prototypes_eval.t())
            return output_q, logits_prot_test

        output_k, k = self.encoder(img_k)

        output_q_mix, q_mix = self.encoder(img_q_mix)
        output_k_mix, k_mix = self.encoder(img_k_mix)

        predicetd_scores_q = torch.softmax(output_q, dim=1) * partial_Y
        predicetd_scores_q_norm = predicetd_scores_q / predicetd_scores_q.sum(dim=1).repeat(args.num_class,
                                                                                            1).transpose(0, 1)

        predicetd_scores_k = torch.softmax(output_k, dim=1) * partial_Y
        predicetd_scores_k_norm = predicetd_scores_k / predicetd_scores_k.sum(dim=1).repeat(args.num_class,
                                                                                            1).transpose(0, 1)

        max_scores_q, pseudo_labels_q = torch.max(predicetd_scores_q_norm, dim=1)
        max_scores_k, pseudo_labels_k = torch.max(predicetd_scores_k_norm, dim=1)

        prototypes = self.prototypes.clone().detach()

        logits_prot_q = torch.mm(q, prototypes.t())
        logits_prot_k = torch.mm(k, prototypes.t())

        logits_prot_q_mix = torch.mm(q_mix, prototypes.t())
        logits_prot_k_mix = torch.mm(k_mix, prototypes.t())

        for feat_q, label_q in zip(concat_all_gather(q), concat_all_gather(pseudo_labels_q)):
            self.prototypes[label_q] = self.proto_weight * self.prototypes[label_q] + (1 - self.proto_weight) * feat_q

        for feat_k, label_k in zip(concat_all_gather(k), concat_all_gather(pseudo_labels_k)):
            self.prototypes[label_k] = self.proto_weight * self.prototypes[label_k] + (1 - self.proto_weight) * feat_k

        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

        return output_q, output_k, logits_prot_q, logits_prot_k, logits_prot_q_mix, logits_prot_k_mix


class PaPiNet(nn.Module):
    def __init__(self, head, feat_dim, num_classes):
        super(PaPiNet, self).__init__()
        clip_model, _ = clip.load('ViT-L/14')
        clip_model.eval()
        for params in clip_model.parameters():
            params.requires_grad = False
        self.encoder = clip_model

        self.fc = Linear(768, num_classes)
        if head == 'linear':
            self.head = nn.Linear(768, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(768, 768),
                nn.ReLU(inplace=True),
                nn.Linear(768, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder.encode_image(x).squeeze(-1).float()
        feat_c = self.head(feat)
        logits = self.fc(feat)
        return logits, F.normalize(feat_c, dim=1)



# From https://github.com/hbzju/PiCO/blob/main/model.py
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # tensors_gather = [torch.ones_like(tensor)
    #                   for _ in range(torch.distributed.get_world_size())]
    # torch.distributed.all_gather(tensors_gather, tensor, async_op=False)


    tensors_gather = [tensor.clone()]
    output = torch.cat(tensors_gather, dim=0)

    return output


@torch.no_grad()
def get_entropy(logits):
    probs = F.softmax(logits, dim=1)
    log_probs = torch.log(probs + 1e-7)  # avoid NaN
    return -(probs * log_probs).sum(1)
