cd ../../c3po/expt_shapenet/

python evaluate_proposed_model.py \
--detector "point_transformer" \
--object bed \
--model "post" \
--dataset shapenet.real.hard

