#!/bin/bash

for i in $(seq -w 00999); do
  echo "processing $i.png"
  python inference.py \
  --input_mask /home/luca/university/cv/project/src/dataset/masked_images/00000/$i.png \
  --input_img /home/luca/university/cv/project/src/dataset/FFHQ/00000/$i.png \
  --output samples/ffhq/$i.jpg \
  --checkpoint_dir ~/gin
done
