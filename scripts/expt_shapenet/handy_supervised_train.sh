cd ../../c3po/expt_shapenet
echo "pre-training: motorcycle"
python training.py "point_transformer" "motorcycle" "supervised"

#echo "pre-training: mug"
#python training.py "point_transformer" "mug" "supervised"

#echo "pre-training: skateboard"
#python training.py "point_transformer" "skateboard" "supervised"

#echo "pre-training: table"
#python training.py "point_transformer" "table" "supervised"

#echo "pre-training: vessel"
#python training.py "point_transformer" "vessel" "supervised"
