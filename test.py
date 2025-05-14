from PIL import Image
from dataset import get_loader
import torch
from torchvision import transforms
from util import save_tensor_img, Logger
from tqdm import tqdm
from torch import nn
import os
from criterion import Eval
import argparse
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main(args):
    # Init model

    device = torch.device("cuda")
    exec('from models import ' + args.model)
    model = eval(args.model+'()')
    model = model.to(device)

    # ginet_dict = torch.load(os.path.join(args.best_epochs, args.one_of_epoch + ".pth"))
    ginet_dict = torch.load("/home/huangjiu/projects/95GCoNet/checkpoint2/best_epochs/ep48_Smeasure0.6656.pth")
    #logger = Logger(os.path.join(args.param_root, "test_log.txt"))

    model.to(device)
    model.ginet.load_state_dict(ginet_dict)

    model.eval()
    model.set_mode('test')

    for testset in ['CoCA', 'CoSOD3k', 'Cosal2015']:
        if testset == 'CoCA':
            test_img_path = '../data/images/CoCA/'
            test_gt_path = '../data/gts/CoCA/'
            saved_root = os.path.join(args.pred_lot, args.one_of_epoch, 'CoCA')
        elif testset == 'CoSOD3k':
            test_img_path = '../data/images/CoSOD3k/'
            test_gt_path = '../data/gts/CoSOD3k/'
            saved_root = os.path.join(args.pred_lot, args.one_of_epoch, 'CoSOD3k')
        elif testset == 'Cosal2015':
            test_img_path = '../data/images/CoSal2015/'
            test_gt_path = '../data/gts/CoSal2015/'
            saved_root = os.path.join(args.pred_lot, args.one_of_epoch, 'CoSal2015')
        else:
            print('Unkonwn test dataset')
            print(args.dataset)
        
        test_loader = get_loader(
            test_img_path, test_gt_path, args.size, 1, istrain=False, shuffle=False, num_workers=8, pin=True)

        for batch in tqdm(test_loader):
            inputs = batch[0].to(device).squeeze(0)
            subpaths = batch[2]
            ori_sizes = batch[3]
            
            scaled_preds = model(inputs)[-1]

            os.makedirs(os.path.join(saved_root, subpaths[0][0].split('/')[0]), exist_ok=True)

            num = len(scaled_preds)
            for inum in range(num):
                subpath = subpaths[inum][0]
                ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
                res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear', align_corners=True)
                save_tensor_img(res, os.path.join(saved_root, subpath))

if __name__ == '__main__':
    # Parameter from command line
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model',
                        default='GCoNet',
                        type=str,
                        help="Options: '', ''")
    #parser.add_argument('--testset',
    #                    default='CoCA',
    #                    type=str,
    #                    help="Options: 'CoCA','CoSal2015','CoSOD3k','iCoseg','MSRC'")
    parser.add_argument('--size',
                        default=224,
                        type=int,
                        help='input size')
    parser.add_argument('--best_epochs', default='./checkpoint2/best_epochs', type=str, help='')
    parser.add_argument('--pred_lot', default='./pred_lot', type=str, help='Output folder')
    parser.add_argument('--one_of_epoch', default=None, type=str, help='')

    args = parser.parse_args()

    main(args)



