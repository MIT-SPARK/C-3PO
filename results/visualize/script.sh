cd ../../c3po/expt_shapenet/

DATASETS="shapenet.real.hard"
DETECTOR_TYPE="point_transformer pointnet"
OBJECT="airplane"

python visualize_model.py \
    --detector "point_transformer" \
    --object "chair" \
    --model "pre" \
    --dataset "shapenet.real.hard"

