#!/bin/sh
#SBATCH --partition=students-prod
#SBATCH --gres=gpu:4
#SBATCH --job-name="distr-contra-gin"
#SBATCH --qos=cvcs_bprod

export MASTER_ADDR=localhost
export MASTER_PORT=3141

python training.py \
  --dataset_dir="/nas/softechict-nas-1/llumetti/FFHQ_MASK_GAN"      \
  --checkpoint_dir="/nas/softechict-nas-1/llumetti/checkpoints/gin" \
  --video_dir="/nas/softechict-nas-1/llumetti/video_frames"         \
  --plots_dir="/homes/llumetti/CVProject/src/gan_inpainting/plots"  \
  --batch_size 6                                                    \
  --input_size 256                                                  \
  --epochs 2                                                        \
  --nodes 1                                                         \
  --gpus 4                                                          \
  --nr 0                                                            \
  --checkpoint 1

