cd ../../c3po/expt_shapenet/

python visualize_model.py \
    --detector "point_transformer" \
    --object "cap" \
    --model "pre" \
    --dataset "shapenet.real.hard"

python visualize_model.py \
    --detector "point_transformer" \
    --object "table" \
    --model "pre" \
    --dataset "shapenet.real.hard"

python visualize_model.py \
    --detector "point_transformer" \
    --object "chair" \
    --model "pre" \
    --dataset "shapenet.real.hard"