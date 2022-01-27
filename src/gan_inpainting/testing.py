import argparse
import logging
from config import Config
from dataset import FaceMaskDataset
from torchvision.utils import save_image
from torchvision import transforms as T
from generator import *
from discriminator import Discriminator
from loss import *
from metrics import TestMetrics
from apex import amp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(netG, netD, dataloader, args):
    netG.eval()
    netD.eval()

    metrics_tester = TestMetrics()

    metrics = {
            'l1': [],
            'accuracy': []
            }

    with torch.no_grad():
        for i, (imgs,masks) in enumerate(dataloader):
            imgs = imgs.to(device)
            masks = masks.to(device)

            imgs = imgs / 127.5 - 1
            masks = masks / 1

            # forward G
            _, coarse_out, refined_out = netG(imgs,masks)
            reconstructed_imgs = refined_out*masks + imgs*(1-masks)

            pos_neg_imgs = torch.cat([imgs,reconstructed_imgs],dim=0)
            dmasks = torch.cat([masks,masks],dim=0)

            # forward D
            pred_pos_neg_imgs = netD(pos_neg_imgs, dmasks)
            pred_pos_imgs, pred_neg_imgs = torch.chunk(pred_pos_neg_imgs, 2, dim=0)

            # Calculate accuracy over the batch
            mean_pos_pred = pred_pos_imgs.clone().detach().mean(dim=1)
            mean_neg_pred = pred_neg_imgs.clone().detach().mean(dim=1)
            mean_pos_pred = torch.where(mean_pos_pred > 0.5, 1, 0).type(torch.FloatTensor)
            mean_neg_pred = torch.where(mean_neg_pred > 0.5, 0, 1).type(torch.FloatTensor)
            accuracyD = torch.sum(mean_pos_pred) + torch.sum(mean_neg_pred)
            tot_elem = mean_pos_pred.shape[0] + mean_neg_pred.shape[0]
            accuracyD /= tot_elem
            metrics['accuracy'].append(accuracyD.item())

            # Calculate L1 loss over the batch
            l1_mean = torch.mean(torch.abs(imgs - reconstructed_imgs))
            metrics['l1'].append(l1_mean)

            output = (reconstructed_imgs + 1) * 127.5

            metrics_tester.update(imgs, reconstructed_imgs)
            for d in range(output.size(0)):
                # needed to calculate FID
                save_image(output[d]/255, f'{args.output_dir}/{i*args.batch_size+d}.png')

        # TODO test_dir is empty
        fid_score = metrics_tester.FID(args.dataset_dir, args.output_dir, args.batch_size,device)
        metrics_dict = metrics_tester.get_metrics()
        metrics_dict['FID'] = fid_score
        for key in metrics:
            metrics_dict[key] = torch.mean(torch.tensor(metrics[key])).item()

    return metrics_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing")
    parser.add_argument("--batch_size", default=2, type=int, help="batch size")
    parser.add_argument("--input_size", default=256, type=int, help="size of the imgs")
    parser.add_argument("--dataset_dir", type=str, help="dataset location", required=True)
    parser.add_argument("--checkpoint_dir", type=str, help="where to load/save checkpoints", required=True)
    # TODO
    parser.add_argument("--output_dir", type=str, help="where to save some test img", required=True)
    args = parser.parse_args()

    # Load dataset
    dataset = FaceMaskDataset(args.dataset_dir, 'maskceleba_test.csv', T.Resize(args.input_size))
    dataloader = dataset.loader(batch_size=args.batch_size)

    netG = MSSAGenerator(input_size=args.input_size)
    netD = Discriminator(input_size=args.input_size)


    netG.to(device)
    netD.to(device)

    netG = torch.nn.DataParallel(netG)
    netD = torch.nn.DataParallel(netD)

    # netG, _ = amp.initialize(netG, None, opt_level='O2')
    # netD, _ = amp.initialize(netD, None, opt_level='O2')

    checkpointG = torch.load(f'{args.checkpoint_dir}/generator.pt', map_location=torch.device('cpu'))
    checkpointD = torch.load(f'{args.checkpoint_dir}/discriminator.pt', map_location=torch.device('cpu'))

    netG.load_state_dict(checkpointG)
    netD.load_state_dict(checkpointD)

    metrics = test(netG, netD, dataloader, args)
    print(metrics)
