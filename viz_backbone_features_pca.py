import argparse
import os
import random
import faiss
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

from data.coco_data_module import CocoDataModule
from models.vit import vit_small


class LeopartEval(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        # Init Model
        self.patch_size = args.patch_size
        spatial_res = 224 / args.patch_size
        assert spatial_res.is_integer()
        self.spatial_res = int(spatial_res)
        self.num_prototypes = args.num_prototypes
        self.projection_feat_dim = args.projection_feat_dim
        self.projection_hidden_dim = args.projection_hidden_dim
        self.n_layers_projection_head = args.n_layers_projection_head
        self.proj_bn = args.proj_bn
        if args.arch == "vit-small":
            self.model = vit_small(
                patch_size=self.patch_size,
                output_dim=self.projection_feat_dim,
                hidden_dim=self.projection_hidden_dim,
                nmb_prototypes=self.num_prototypes,
                n_layers_projection_head=self.n_layers_projection_head,
                use_contrastive_projector=False,
                proj_bn=self.proj_bn,
            )
        else:
            raise NotImplementedError()
        
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        B = x.size(0)
        with torch.no_grad():
            tokens = self.model.forward_backbone(x)
            tokens = tokens[:, 1:]
            tokens = tokens.reshape(-1, self.model.embed_dim)
            tokens = tokens.reshape(B, self.spatial_res, self.spatial_res, self.model.embed_dim).permute(0, 3, 1, 2)
            # emb, _ = self.model.forward_head(tokens)
            # emb = emb.reshape(B, self.spatial_res, self.spatial_res, self.projection_feat_dim).permute(0, 3, 1, 2)
            # dots = dots.reshape(B, self.spatial_res, self.spatial_res, self.num_prototypes).permute(0, 3, 1, 2)   
            return tokens


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
    model = LeopartEval(args)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    weights = checkpoint["state_dict"]
    model_weights = {k.replace("teacher", "model"):v for k, v in weights.items() if k.startswith("teacher")}
    # model_weights = {k:v for k, v in weights.items() if k.startswith("model")}
    msg = model.load_state_dict(model_weights, strict=False)
    print(msg)
    model = model.eval()
    del checkpoint
    return model


def get_features(model, data_loader):
    feature_bank = []
    mask_bank = []
    for data in tqdm(data_loader, desc='Feature extracting', leave=False, disable=False):
        img, mask = data
        mask = mask.cuda(non_blocking=True)
        feature = model(img.cuda(non_blocking=True)) #.mean(dim=(-2, -1))
        # feature = F.normalize(feature, dim=1)
        feature_bank.append(feature)
        mask_bank.append(mask)
    feature_bank = torch.cat(feature_bank, dim=0)
    mask_bank = torch.cat(mask_bank, dim=0)
    return feature_bank, mask_bank


def prepare_knn(model, data_loader, args):
    # prototypes = F.normalize(model.model.prototypes.weight, dim=1) # k x d
    feature_bank, mask_bank = get_features(model, data_loader, args.batch_size) # n x d x h x w
    masks_adder = mask_bank.flatten(2, 3) + 1.e-6
    masks_freq = masks_adder.sum(-1)
    cov = torch.cov(masks_freq.T)
    l = 0.01
    cov_inv = torch.inverse(cov + l*torch.eye(args.num_prototypes).to(masks_freq.device))
    scores = (masks_freq @ cov_inv) @ (masks_freq.T)
    _, idxs = scores.t().topk(dim=1, k=args.topk)
    return feature_bank, idxs


def prepare_ktop_counts(model, dataset, args):
    prototypes = F.normalize(model.model.prototypes.weight, dim=1) # k x d
    memory_bank = get_features(model, dataset, args.batch_size) # n x d x h x w
    dots = torch.einsum('kd,ndhw->nkhw', [prototypes, memory_bank]) # n x k x h x w
    masks = torch.zeros_like(dots).scatter_(1, dots.argmax(1, keepdim=True), 1)
    counts = masks.sum(-1).sum(-1) # n x k
    _, idxs = counts.t().topk(dim=1, k=args.topk)
    return dots, idxs


# TODO: Fix this!!
def viz_knn(dataset, feats, idxs, plot_idxs, color_arr, args):    
    os.makedirs(os.path.join(args.save_path, f"masked_imgs"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, f"full_assgn_imgs"), exist_ok=True)
    
    coverage_avgs = []
    for i, plot_idx in enumerate(tqdm(plot_idxs, desc='Mining retreiving', leave=False, disable=False)):
        top_masked_imgs = []
        full_assgn_imgs = []
        coverage_counts = []
        for j in range(args.topk):
            idx = idxs[plot_idx, j]
            image, gt = dataset[idx]
            image = denorm_tensor(image)
            # import ipdb; ipdb.set_trace()
            feat = F.interpolate(feats[idx].unsqueeze(0), size=image.shape[-2:], mode="bilinear")[0]
            assert feat.shape[-2:] == gt.shape[-2:]
            # TODO: PCA1
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
    parser.add_argument('--sampling', type=int, default=20)
    parser.add_argument('--idxs', type=list, default=[])
    parser.add_argument('--save_path', type=str, default='outputs')
    parser.add_argument('--mode', type=str, choices=["knn", "topk_counts"], default="knn")
    # dataset
    parser.add_argument('--dataset', type=str, default='COCOval', help='dataset type')
    parser.add_argument('--data_dir', type=str, default='./datasets/coco', help='dataset director')
    parser.add_argument('--batch_size', type=int, default=64)
    # Model.
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--arch', type=str, default='vit-small')
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--num_prototypes', type=int, default=300)
    parser.add_argument('--projection_feat_dim', type=int, default=256)
    parser.add_argument('--projection_hidden_dim', type=int, default=2048)
    parser.add_argument('--proj_bn', action='store_true')
    parser.add_argument('--n_layers_projection_head', type=int, default=3)
    parser.add_argument('--temp', default=0.07, type=float)
    args = parser.parse_args()

    # mean_vals, std_vals = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=mean_vals, std=std_vals)])

    train_transforms = None
    val_image_transforms = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_target_transforms = transforms.Compose(
        [
            transforms.Resize((256, 256), interpolation=TF.InterpolationMode.NEAREST),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    if "coco" in args.dataset_name:
        assert len(args.dataset_name.split("-")) == 2
        mask_type = args.dataset_name.split("-")[-1]
        assert mask_type in ["stuff", "thing"]
        if mask_type == "thing":
            num_classes = 12
        else:
            num_classes = 15
        args.num_classes = num_classes
        ignore_index = 255
        data_dir = "/scratch/hdd001/home/sroutray/dataset/coco"
        file_list = os.listdir(os.path.join(data_dir, "images", "train2017"))
        file_list_val = os.listdir(os.path.join(data_dir, "images", "val2017"))
        random.shuffle(file_list_val)

        data_module = CocoDataModule(
            batch_size=args.batch_size,
            num_workers=8,
            file_list=file_list,
            data_dir=data_dir,
            file_list_val=file_list_val,
            mask_type=mask_type,
            train_transforms=train_transforms,
            val_transforms=val_image_transforms,
            val_target_transforms=val_target_transforms,
        )
    else:
        raise ValueError(f"{args.dataset_name} not supported")
    
    val_loader = data_module.val_dataloader()
    
    model = get_model(args).cuda()

    if args.mode == "knn":
        args.save_path = os.path.join(args.save_path, "knn")
        feats, idxs = prepare_knn(model, val_loader, args)
    # elif args.mode == "topk_counts":
    #     args.save_path = os.path.join(args.save_path, "topk_counts")
    #     dots, idxs = prepare_ktop_counts(model, val_loader, args)
    else:
        raise NotImplementedError()
    
    if args.sampling > 0:
        plot_idxs = np.random.randint(0, len(file_list_val), args.sampling)
    elif len(args.idxs) > 0:
        plot_idxs = args.idxs
    else:
        plot_idxs = range(len(file_list_val))
    
    color_map = get_cmap(args.num_prototypes, "rainbow")
    color_arr = get_color_arr(args, color_map)
    viz_knn(data_module.coco_val, feats, idxs, plot_idxs, color_arr, args)