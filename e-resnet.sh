set -x

CORRUPTIONS=(
    'gaussian_noise' 'shot_noise' 'impulse_noise' 'defocus_blur'
    'glass_blur' 'motion_blur' 'zoom_blur' 'snow' 'frost' 'fog'
    'brightness' 'contrast' 'elastic_transform' 'pixelate'
    'jpeg_compression')

for corruption in ${CORRUPTIONS[@]}
do
    CUDA_VISIBLE_DEVICES=4 python extrapolation-resnet.py \
        --corruption $corruption
done