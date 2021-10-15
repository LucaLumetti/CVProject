import math

import cv2
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import numpy as np
from skimage.metrics import structural_similarity as ssim

from fid_score import calculate_fid_given_paths

import lpips


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TrainingMetrics:

    def __init__(self, screenshot_step, video_dir, dataset):
        self.losses = dict()
        self.accuracy = []
        self.dataset = dataset
        self.fimg, self.fmask = dataset.__getitem__(0)
        self.ssim = []
        self.psnr = []
        self.lpips = []
        self.iter = 0
        self.screenshot_step = screenshot_step
        self.video_dir = video_dir
        self.save_video = True

    '''
        Parameters:
            loss_list: dict with {name: value} name is useful for the graph's legend
            D_result: result of Discriminator for accuracy
            netG: Generator Net for testing
            netD: Discriminator Net for testing
    '''
    def update(self, loss_list: dict , D_result, netG, netD):
        with torch.inference_mode():
            if loss_list is None or len(loss_list.keys()) < 2:
                raise Exception("losses must be at least two")

            if not self.losses:
                for name in loss_list.keys():
                    self.losses[name] = list()

            for (name, value) in loss_list.items():
                self.losses[name].append(value)

            pred_pos_imgs, pred_neg_imgs = torch.chunk(D_result, 2, dim=0)

            # canculate accuracy of D
            mean_pos_pred = pred_pos_imgs.clone().detach().mean(dim=1)
            mean_neg_pred = pred_neg_imgs.clone().detach().mean(dim=1)
            mean_pos_pred = torch.where(mean_pos_pred > 0.5, 1, 0).type(torch.FloatTensor)
            mean_neg_pred = torch.where(mean_neg_pred > 0.5, 0, 1).type(torch.FloatTensor)
            accuracyD = torch.sum(mean_pos_pred) + torch.sum(mean_neg_pred)
            tot_elem = mean_pos_pred.shape[0] + mean_neg_pred.shape[0]
            accuracyD /= tot_elem
            self.accuracy.append(accuracyD.item())

            # every screenshot_step img, print losses, update the graph, output an image as example
            if self.iter % self.screenshot_step == 0:
                print(f"[{self.iter}]\t" + \
                      f"accuracy_d: {self.accuracy[-1]},")

                fig, axs = plt.subplots(1, 1)

                axs.set_xlabel('iterations')

                for i,key in enumerate(self.losses):
                    name = key
                    value = self.losses[key]
                    x_axis = range(len(self.losses[key]))
                    print(f"{name}: {value[-1]},")

                    # loss i-th
                    axs.plot(x_axis, value, label = name)

                axs.plot(x_axis, self.accuracy, label = "accuracy discr")
                lgd = axs.legend(bbox_to_anchor = (0.3, 1.3), loc="upper center", ncol=((len(self.losses)+1)//2))
                fig.tight_layout()
                fig.savefig(f'plots/loss.png', dpi=fig.dpi, bbox_extra_artists=(lgd,), bbox_inches='tight')
                plt.close(fig)

            # save video frames x10 more frequently
            if self.save_video and self.iter % 5 == 0:
                fimg = self.fimg.to(device)
                fmask = self.fmask.to(device)

                # change img range from [0,255] to [-1,+1]
                fimg = fimg / 127.5 - 1

                coarse_out, refined_out = netG(fimg[None,:,:], fmask[None,:,:])

                # coarse_fimg = coarse_out * fmask + fimg * (1 - fmask)
                reconstructed_fimg = refined_out * fmask + fimg * (1 - fmask)

                # checkpoint_img = ((img[0] + 1) * 127.5)
                # checkpoint_fcoarse = ((coarse_fimg[0] + 1) * 127.5)
                checkpoint_frecon = ((reconstructed_fimg[0] + 1) * 127.5)

                # save_image(checkpoint_img / 255, f'plots/orig_{self.iter}.png')
                # save_image(checkpoint_coarse / 255, f'plots/coarse_{self.iter}.png')
                save_image(checkpoint_frecon/255, f'{self.video_dir}/frame_{self.iter//5}.png')
            self.iter += 1
        return


class TestMetrics:

    def __init__(self):
        self.ssim = []
        self.pnsr = []
        self.lpips = []
        self.loss_alex = lpips.LPIPS(net='alex').to(device)

    def update(self, original, generated):
        if len(original.shape) != 4:
            raise Exception("Unexpected dimension, images must have 4 dimension")

        if len(generated.shape) == 3:
            generated = generated.squeeze(0)
        batch_size = original.shape[0]

        for i in range(batch_size):
            self.ssim.append(self.SSIM(original[i], generated[i]))
            self.lpips.append(self.LPIPS(original[i], generated[i]))
            self.pnsr.append(self.PSNR(original[i], generated[i]))

    '''
        return a dict with metrics
    '''
    def get_metrics(self):
        return dict({"SSIM": np.mean(self.ssim), "PSNR": np.mean(self.pnsr), "LPIPS": np.mean(self.lpips)})

    '''
        plot metrics and save it
    '''
    def save_plot(self):
        fig, axs = plt.subplots(3, 1)

        x_axis = len(self.ssim)
        # ssim
        axs[0].plot(x_axis, self.ssim)
        axs[0].set_xlabel('iterations')
        axs[0].set_ylabel("SSIM")
        axs[0].title.set_text("SSIM")

        # PSNR
        axs[1].plot(x_axis, self.pnsr)
        axs[1].set_xlabel('iterations')
        axs[1].set_ylabel("PSNR")
        axs[0].title.set_text("PSNR")

        # LPIPS
        axs[2].plot(x_axis, self.lpips)
        axs[2].set_xlabel('iterations')
        axs[2].set_ylabel("LPIPS")
        axs[0].title.set_text("PSNR")

        fig.tight_layout()
        fig.savefig('plots/quality_metrics.png', dpi=fig.dpi)
        plt.close(fig)

    '''
        calculate the Structural SIMilarity (SSIM)
        Params:
        --original      : orignal image
        --generate      : generator output
        return:
        --score         : a bigger score indicates better images
    '''
    def SSIM(self, original, generated):

        original = original.cpu()
        generated = generated.cpu()

        original = original.numpy()
        generated = generated.numpy()

        original = np.swapaxes(original, 0,1)
        original = np.swapaxes(original, 1, 2)

        generated = np.swapaxes(generated, 0, 1)
        generated = np.swapaxes(generated, 1, 2)

        ssims = []
        for i in range(3):
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2

            kernel = cv2.getGaussianKernel(11, 1.5)
            window = np.outer(kernel, kernel.transpose())

            mu1 = cv2.filter2D(original, -1, window)[5:-5, 5:-5]  # valid
            mu2 = cv2.filter2D(generated, -1, window)[5:-5, 5:-5]
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = cv2.filter2D(original ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
            sigma2_sq = cv2.filter2D(generated ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
            sigma12 = cv2.filter2D(original * generated, -1, window)[5:-5, 5:-5] - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            ssims.append(ssim_map.mean())
        return np.array(ssims).mean()


    '''
        Calculate the Peak Signal to Noise Ratio (PSNR)
        Params:
        --original      : orignal image
        --generate      : generator output
        return:
        --score         : a bigger psnr indicates better images
    '''
    def PSNR(self, original, generate):
        mse = torch.mean((original - generate) ** 2)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        psnr = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
        return psnr.cpu().numpy()

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
    def FID(self, data_orig, data_gen, batch_size = 50, device=None, dims=2048, num_workers= 2):
        if device is None:
            dev = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
        else:
            dev = torch.device(device)

        paths = [data_orig,data_gen]

        fid_value = calculate_fid_given_paths(paths,
                                              batch_size,
                                              dev,
                                              dims)
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
    def LPIPS(self, original, generated):
        # change img range from [0,255] to [-1,+1]
        original = original / 127.5 - 1
        generated = generated / 127.5 -1

        result = self.loss_alex(original, generated)

        return result.cpu().numpy().mean()
