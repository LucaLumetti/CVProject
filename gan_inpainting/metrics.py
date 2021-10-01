import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import numpy as np
from skimage.metrics import structural_similarity as ssim

from pytorch_fid.fid_score import calculate_fid_given_paths

import lpips

class TrainingMetrics:

    def __init__(self, dataloader):
        self.losses = None
        self.accuracy = []
        self.dataloader = dataloader
        self.img, self.mask = next(iter(dataloader))
        self.ssim = []
        self.psnr = []
        self.lpips = []
        self.iter = 0

    '''
        Parameters:
            losses: dict with {name: value} name is useful for the graph's legend
            D_result: result of Discriminator for accuracy
            netG: Generator Net for testing
            netD: Discriminator Net for testing
    '''
    def update(self, losses: dict , D_result, netG, netD):

        if losses is None or len(losses.keys()) < 2:
            raise Exception("losses must be at least two")
        self.iter += 1

        if self.losses is None:
            self.losses = losses
        else:
            for (name, value) in losses.items():
                self.losses[name].append(value)

        pred_pos_imgs, pred_neg_imgs = torch.chunk(D_result, 2, dim=0)

        # with torch.inference_mode():
        mean_pos_pred = pred_pos_imgs.mean(dim=1)
        mean_neg_pred = pred_neg_imgs.mean(dim=1)

        mean_pos_pred[mean_pos_pred > 0.5] = 1
        mean_pos_pred[mean_pos_pred <= 0.5] = 0

        mean_neg_pred[mean_neg_pred > 0.5] = 1
        mean_neg_pred[mean_neg_pred <= 0.5] = 0
        mean_neg_pred = torch.Tensor([1 if elem == 0 else 0 for elem in mean_neg_pred])

        accuracyD = torch.sum(mean_pos_pred) + torch.sum(mean_neg_pred)
        accuracyD /= mean_pos_pred.shape[0] + mean_neg_pred.shape[0]
        self.accuracy.append(accuracyD)

        # every 100 img, print losses, update the graph, output an image as example
        if self.iter % 100 == 0:
            print(f"[{self.iter / 100}]\t" + \
                  f"accuracy_d: {accuracy[-1]},")

            fig, axs = plt.subplots(len(self.losses.items()), 1)
            x_axis = range(self.iter)

            for i,(name,value) in enumerate(self.losses):
                print(f"{name}: {value[-1]},")

                # loss i-th
                axs[i].plot(x_axis, value)
                axs[i].set_xlabel('iterations')
                axs[i].set_ylabel(name)

            fig.tight_layout()
            fig.savefig('plots/loss.png', dpi=fig.dpi)
            plt.close(fig)

            #using CPU
            #img = self.img.to(device)
            #mask = self.mask.to(device)

            # change img range from [0,255] to [-1,+1]
            img = self.img / 127.5 - 1

            coarse_out, refined_out = netG(img, self.mask)
            reconstructed_imgs = refined_out * self.mask + imgs * (1 - self.mask)
            checkpoint_recon = ((reconstructed_imgs[0] + 1) * 127.5)
            checkpoint_img = ((img[0] + 1) * 127.5)

            fig,axs = plt.subplots(3,1)

            self.ssim.append(SSIM(self.img, checkpoint_img))
            self.lpips.append(LPIPS(self.img, checkpoint_img))
            self.psnr.append(PSNR(self.img, checkpoint_img))

            x_axis = len(self.ssim)
            # ssim
            axs[0].plot(x_axis, self.ssim)
            axs[0].set_xlabel('iterations')
            axs[0].set_ylabel("SSIM")
            axs[0].title.set_text("SSIM")

            #PSNR
            axs[1].plot(x_axis, self.psnr)
            axs[1].set_xlabel('iterations')
            axs[1].set_ylabel("PSNR")
            axs[0].title.set_text("PSNR")

            #LPIPS
            axs[2].plot(x_axis, self.lpips)
            axs[2].set_xlabel('iterations')
            axs[2].set_ylabel("LPIPS")
            axs[0].title.set_text("PSNR")

            fig.tight_layout()
            fig.savefig('plots/quality_metrics.png', dpi=fig.dpi)
            plt.close(fig)

            save_image(checkpoint_recon / 255, 'plots/recon.png')
            save_image(checkpoint_img / 255, 'plots/orig.png')


'''
    calculate the Structural SIMilarity (SSIM)
    Params:
    --original      : orignal image
    --generate      : generator output
    return:
    --score         : a bigger score indicates better images 
'''
def SSIM(original, generate):
    similarity = ssim(original, generate, data_range=original.max() - original.min())
    return similarity


'''
    Calculate the Peak Signal to Noise Ratio (PSNR) 
    Params:
    --original      : orignal image
    --generate      : generator output
    return:
    --score         : a bigger psnr indicates better images
'''
def PSNR(original, generate):
    mse = np.mean((original - generate) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return psnr

'''
    Calculate Fréchet Inception Distance (FID) from original's dataset and generate's dataset
    Params:
    --data_orig     : path of the dataset with the original images
    --data_gen      : path with the dataset with the generate images
    --batch_size    : batch_size for dataloader
    --device        : it can be like cuda:0 or cpu, it's better if there is a GPU
    --dims          : we can use different layer of the inception network, default is 2048 like paper
    --num_worker    : num_worker fro operations
    
    return:
    --fid_score     : a lower score indicates better-quality images
'''

def FID(data_orig, data_gen, batch_size = 50, device=None, dims=2048, num_worker= 8):
    if device is None:
        dev = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        dev = torch.device(device)

    paths = [data_orig,data_gen]

    fid_value = calculate_fid_given_paths(paths,
                                          batch_size,
                                          dev,
                                          dims,
                                          num_workers)
    print('FID: ', fid_value)
    return fid_value


'''
    calculate Perceptual similarity (LPIPS)
    Params:
    --original      : tansor with original image, size (N,3,H,W)
    --generated     : tensot with generated image, size (N,3,H,W)
    return:
    --result        : average between N LPIPS
'''
def LPIPS(original, generated):
    # change img range from [0,255] to [-1,+1]
    original = original / 127.5 - 1
    generated = generated / 127.5 -1

    loss_alex = lpips.LPIPS(net='alex')

    result = loss_alex(original, generated)

    return result.mean
