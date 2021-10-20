#SBATCH --partition=students-prod
#SBATCH --gres=gpu:4
#SBATCH --job-name="contra-gin"
#SBATCH --qos=cvcs_bprod

export MASTER_ADDR=localhost
export MASTER_PORT=3141

python training.py \
  --dataset_dir = "/nas/softechict-nas-1/llumetti/FFHQ_MASK_GAN"      \
  --checpoint_dir = "/nas/softechict-nas-1/llumetti/checkpoints/gin"  \
  --video_dir = "/nas/softechict-nas-1/llumetti/video_frames"         \
  --batch_size = 4                                                    \
  --input_size = 256                                                  \
  --epochs = 1                                                        \
  --nodes = 1                                                         \
  --gpus = 4                                                          \
  --nr = 0
