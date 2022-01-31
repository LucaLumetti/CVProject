[ -e $1 ] || echo "Input image $1 does not exists"
[ -d output ] || mkdir output

if [ -z $2 ]; then
  # classical cv
  python src/main.py \
    --input_front $1 \

  # deep
  python src/gan_inpainting/inference.py \
  --input_img output/face.jpg   \
  --input_mask output/mask.jpg  \
  --output output/result.jpg    \
  --checkpoint_dir ~/gin
else
  # classical cv
  python src/main.py   \
    --input_front $1   \
    --input_lateral $2

  # deep
  python src/gan_inpainting/inference.py \
  --input_img output/face_ref.jpg  \
  --input_mask output/mask_ref.jpg     \
  --output output/result.jpg       \
  --checkpoint_dir ~/gin
fi


