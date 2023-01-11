import argparse
import os
from dominate import document
import dominate.tags as htags
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from data.datasets import ImageFolder
from models import resnet
from models.slotcon import SlotConEval


def get_cmap(n: int, name: str = "hsv"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def get_color_arr(args, color_map):
    color_arr = []
    for p in range(args.num_prototypes):
        col = color_map(p)[:-1]
        color_arr.append(col)
    color_arr = torch.FloatTensor(color_arr)
    return color_arr


def denorm(img):
    mean, val = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
    img = (img * val[:, None, None] + mean[:, None, None]) * torch.tensor([255, 255, 255])[:, None, None]
    return img.permute(1, 2, 0).cpu().type(torch.uint8)


def denorm_tensor(img):
    mean, val = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
    img = (img * val[:, None, None] + mean[:, None, None])
    return img.cpu()


def get_model(args):
    encoder = resnet.__dict__[args.arch]
    model = SlotConEval(encoder, args)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    weights = {k.replace('module.', ''):v for k, v in checkpoint['model'].items()}
    msg = model.load_state_dict(weights, strict=False)
    print(msg)
    model = model.eval()
    return model


def get_features(model, dataset, bs):
    memory_loader = torch.utils.data.DataLoader(
        dataset, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    bank = []
    for data in tqdm(memory_loader, desc='Feature extracting', leave=False, disable=False):
        feature = model.projector_k(model.encoder_k(data.cuda(non_blocking=True)))#.mean(dim=(-2, -1))
        feature = F.normalize(feature, dim=1)
        bank.append(feature)
    bank = torch.cat(bank, dim=0)
    return bank


def prepare_knn(model, dataset, args):
    prototypes = F.normalize(model.grouping_k.slot_embed.weight, dim=1) # k x d
    memory_bank = get_features(model, dataset, args.batch_size) # n x d x h x w
    dots = torch.einsum('kd,ndhw->nkhw', [prototypes, memory_bank]) # n x k x h x w
    masks = torch.zeros_like(dots).scatter_(1, dots.argmax(1, keepdim=True), 1)
    masks_adder = masks + 1.e-6
    scores = (dots * masks_adder).sum(-1).sum(-1) / masks_adder.sum(-1).sum(-1) # n x k
    _, idxs = scores.t().topk(dim=1, k=args.topk)
    return dots, idxs


def prepare_ktop_counts(model, dataset, args):
    prototypes = F.normalize(model.grouping_k.slot_embed.weight, dim=1) # k x d
    memory_bank = get_features(model, dataset, args.batch_size) # n x d x h x w
    dots = torch.einsum('kd,ndhw->nkhw', [prototypes, memory_bank]) # n x k x h x w
    masks = torch.zeros_like(dots).scatter_(1, dots.argmax(1, keepdim=True), 1)
    counts = masks.sum(-1).sum(-1) # n x k
    _, idxs = counts.t().topk(dim=1, k=args.topk)
    return dots, idxs


def viz_knn(dataset, dots, idxs, slot_idxs, color_arr, args):    
    os.makedirs(os.path.join(args.save_path, f"proto_imgs"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, f"masked_imgs"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, f"full_assgn_imgs"), exist_ok=True)
    
    coverage_avgs = []
    for i, slot_idx in enumerate(tqdm(slot_idxs, desc='KNN retreiving', leave=False, disable=False)):
        proto_color = color_arr[slot_idx]
        proto_color = proto_color.reshape(3, 1, 1)
        torchvision.utils.save_image(proto_color.expand(-1, 14, 14), os.path.join(args.save_path, f"proto_imgs/{slot_idx:03d}.png"))
        top_masked_imgs = []
        full_assgn_imgs = []
        coverage_counts = []
        for j in range(args.topk):
            idx = idxs[slot_idx, j]
            image = denorm_tensor(dataset[idx])
            # import ipdb; ipdb.set_trace()
            pred = F.interpolate(dots[idx].unsqueeze(0), size=image.shape[-2:], mode="bilinear")[0]
            # pred = transforms.functional.resize(dots[idx], image.shape[-2:], TF.InterpolationMode.BILINEAR)
            mask = torch.zeros_like(pred).scatter_(0, pred.argmax(0, keepdim=True), 1)
            mask = mask[slot_idx].unsqueeze(0).cpu()
            image = (args.alpha * (image * mask) + (1 - args.alpha) * proto_color * mask) + (image * (1 - mask))
            assgn = (pred / args.temp).softmax(dim=0)
            # assgn = (pred**3 / args.temp).softmax(dim=0)
            assgn_c = torch.einsum("khw,kc->chw", assgn.cpu(), color_arr)            
            top_masked_imgs.append(image)
            full_assgn_imgs.append(assgn_c)
            coverage_counts.append(mask.sum())

        top_masked_grid = torchvision.utils.make_grid(top_masked_imgs, nrow=int(args.topk/2))
        full_assgn_grid = torchvision.utils.make_grid(full_assgn_imgs, nrow=int(args.topk/2))
        torchvision.utils.save_image(top_masked_grid, os.path.join(args.save_path, f"masked_imgs/{slot_idx:03d}.png"))
        torchvision.utils.save_image(full_assgn_grid, os.path.join(args.save_path, f"full_assgn_imgs/{slot_idx:03d}.png"))
        coverage_avgs.append(np.array(coverage_counts).mean())
    
    create_html(args.save_path, args.mode, coverage_avgs)


def create_html(root, mode, coverage_avgs):
    anchor_dir = os.path.join(root, "proto_imgs")
    img_names = sorted(os.listdir(anchor_dir))
    with document(title=f"Images and Features Viz ({mode})") as doc:
        with doc.head:
            htags.style(
                """
                * {
                    margin: 0;
                    padding: 0;
                }
                .imgbox {
                    display: grid;
                    height: 100%;
                }
                .center-fit {
                    max-width: 100%;
                    max-height: 100vh;
                    margin: auto;
                }
                """
            )
            for i, im in enumerate(img_names):
                htags.h1(f"Prototype ID {im[:-4]}")
                htags.h2(f"Prototype Color {im[:-4]}")
                htags.div(htags.img(src=f"proto_imgs/{im}", _class="center-fit"), _class="imgbox")
                htags.h2(f"Prototype Masked Image {im[:-4]}    Coverage Average {coverage_avgs[i]}")
                htags.div(htags.img(src=f"masked_imgs/{im}", _class="center-fit"), _class="imgbox")
                htags.h2(f"Assignment Image {im[:-4]}")
                htags.div(htags.img(src=f"full_assgn_imgs/{im}", _class="center-fit"), _class="imgbox")
    with open(os.path.join(root, "index.html"), "w") as fh:
        fh.write(doc.render())
        fh.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # viz-related
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--dpi', type=int, default=100)
    parser.add_argument('--sampling', type=int, default=0)
    parser.add_argument('--idxs', type=list, default=[])
    parser.add_argument('--save_path', type=str, default='outputs')
    parser.add_argument('--mode', type=str, choices=["knn", "topk_counts"], default="knn")
    # dataset
    parser.add_argument('--dataset', type=str, default='COCOval', help='dataset type')
    parser.add_argument('--data_dir', type=str, default='./datasets/coco', help='dataset director')
    parser.add_argument('--batch_size', type=int, default=64)
    # Model.
    parser.add_argument('--model_path', type=str, default='output/slotcon_coco_r50_800ep/ckpt_epoch_800.pth')
    parser.add_argument('--dim_hidden', type=int, default=4096)
    parser.add_argument('--dim_out', type=int, default=256)
    parser.add_argument('--arch', type=str, default='resnet50')
    parser.add_argument('--num_prototypes', type=int, default=256)
    parser.add_argument('--temp', default=0.07, type=float)
    args = parser.parse_args()

    mean_vals, std_vals = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=mean_vals, std=std_vals)])

    dataset = ImageFolder(args.dataset, args.data_dir, transform)
    model = get_model(args).cuda()

    if args.mode == "knn":
        args.save_path = os.path.join(args.save_path, "knn")
        dots, idxs = prepare_knn(model, dataset, args)
    elif args.mode == "topk_counts":
        args.save_path = os.path.join(args.save_path, "topk_counts")
        dots, idxs = prepare_ktop_counts(model, dataset, args)
    else:
        raise NotImplementedError()
    
    if args.sampling > 0:
        slot_idxs = np.random.randint(0, args.num_prototypes, args.sampling)
    elif len(args.idxs) > 0:
        slot_idxs = args.idxs
    else:
        slot_idxs = range(args.num_prototypes)
    
    color_map = get_cmap(args.num_prototypes, "rainbow")
    color_arr = get_color_arr(args, color_map)
    viz_knn(dataset, dots, idxs, slot_idxs, color_arr, args)