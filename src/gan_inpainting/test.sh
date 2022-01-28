#!/bin/bash

for i in $(seq -w 00100); do
  echo "processing $i.jpg"
  python inference.py \
  --input_img /nas/softechict-nas-1/llumetti/CELEBA_MASK/celeba_hq_256/00000/$i.jpg \
  --input_mask /nas/softechict-nas-1/llumetti/CELEBA_MASK/masked_images/00000/$i.jpg \
  --output samples/$i.jpg \
  --checkpoint_dir ~/gin
done
