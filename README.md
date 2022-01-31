## People
The people involved in the project are:
- Luca Lumetti (244577@studenti.unimore.it)
- Matteo Di Bartolomeo (241469@studenti.unimore.it)
- Federico Silvestri (243938@studenti.unimore.it)

## Installation
Python 3 is required, the main python packages required are: pytorch, torchvision, opencv, mediapipe,
pytorch\_fid, lpips and nvidia-apex. Everything can be installed via pip (`pip
install -r requirements.txt`) but an exception is made for nvidia-apex, which
can be installed via anaconda (`conda install -c conda-forge nvidia-apex`) or by
following the [guide here](https://github.com/NVIDIA/apex#quick-start)

## Run
To run the whole pipeline over a single image or a pair image+reference, execute:
```shellscript
sh run.sh input.jpg [reference.jpg]
```
The outputs will be inside the `output` directory

## Project structure
There are 2 main folders, `pdf` and `src`. The first one contain the whole LaTeX code, images, etc... that were used to produce the [final report](https://github.com/LucaLumetti/CVProject/blob/main/pdf/cvproject.pdf)
In the `src` folder there is the source code of the whole project:
    - `src/mask_detection.py` is responsible for the detection of the surgical mask.
    - `src/warpface.py` is reponsible of applying the TPS over the reference photo.
    - `src/main.py` join the 2 scripts above, it is basically the classical part
      of the pipeline.
Everithing related to the deep neural network can be found inside `src/gan_inpainting/`:
- `run.sh` and `test.sh` are the slurm scripts to start the training and the testing on the AImagelab Servers.
- `layers.py` contains the pytorch implementation various layers used in the networks, so SelfAttention, GatedConv, ResNetSPD, ...
- `loss.py` contains the implementation of different losses used in the
  training.
- `generator.py` and `discriminator.py` are the generator and discriminator classes. `generator.py` actually contains 2 different generator, in the end we used the MSSAGenerator which includes the Multi Scale Self Attention mechanism.
- `dataset.py` contains two dataset classes: FakeDataset, only used during development for testing purposes, and FaceMaskDataset. The datasets used are an extension for FFHQ and CelebA and they can be found on AImagelab server at `/nas/softechict-nas-1/llumetti/FFHQ_MASK_GAN` and `/nas/softechict-nas-1/llumetti/CELEBA_MASK` respectively.

## Run the detection + tps
To execute the classical pipeline alone, run `main.py`:
```bash
python main.py
```

## Run the network
To execute the net over a single image only, it is possible to use `inference.py`:
```bash
python inference.py \
    --input_img face.jpg \
    --input_mask mask.jpg \
    --output output.jpg \
    --checkpoint_dir /nas/softechict-nas-1/llumetti/checkpoints/gin
```

## Train and Testing
To run the training and/or testing, is possible to use `run.sh` and `test.sh` (even without SLURM). For multinode you have to modify the arguments inside the scripts accordingly.
