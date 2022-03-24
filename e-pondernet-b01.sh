set -x

CORRUPTIONS=(
    'gaussian_noise' 'shot_noise' 'impulse_noise' 'defocus_blur'
    'glass_blur' 'motion_blur' 'zoom_blur' 'snow' 'frost' 'fog'
    'brightness' 'contrast' 'elastic_transform' 'pixelate'
    'jpeg_compression')

SEVERITY=(1 3 5)

for corruption in ${CORRUPTIONS[@]}
do
    for severity in ${SEVERITY[@]}
    do
        CUDA_VISIBLE_DEVICES=2 python extrapolation.py \
            --corruption $corruption \
            --severity $severity
    done
done