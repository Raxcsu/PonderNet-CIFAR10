set -x

CORRUPTIONS=(
    'gaussian_noise' 'shot_noise' 'impulse_noise' 'defocus_blur'
    'glass_blur' 'motion_blur' 'zoom_blur' 'snow' 'frost' 'fog'
    'brightness' 'contrast' 'elastic_transform' 'pixelate'
    'jpeg_compression')

SEVERITY=(1 2 3 4 5)

for corruption in ${CORRUPTIONS[@]}
do
    for severity in ${SEVERITY[@]}
    do
        CUDA_VISIBLE_DEVICES=3 python extrapolation-b1.py \
            --corruption $corruption \
            --severity $severity
    done
done