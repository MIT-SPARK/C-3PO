
#echo "BOTTLE:: POINT TRANSFORMER"
#python full_self_supervised_training.py "point_transformer" "bottle"

echo "CHAIR:: POINT TRANSFORMER"
#python evaluate_trained_model.py "point_transformer" "chair" "pre" "chair"
#python evaluate_trained_model.py "point_transformer" "chair" "post" "table"
#python evaluate_trained_model.py "point_transformer" "chair" "post" "chair"
#python evaluate_trained_model.py "point_transformer" "chair" "post" "bottle"
#python evaluate_trained_model.py "point_transformer" "chair" "post" "laptop"
python evaluate_trained_model.py "point_transformer" "chair" "post" "skateboard"

#echo "LAPTOP:: POINT TRANSFORMER"
#python full_self_supervised_training.py "point_transformer" "laptop"
#
#echo "SKATEBOARD:: POINT TRANSFORMER"
#python full_self_supervised_training.py "point_transformer" "skateboard"
#
#echo "TABLE:: POINT TRANSFORMER"
#python full_self_supervised_training.py "point_transformer" "table"
#
#
#
#
#echo "BOTTLE:: POINTNET"
#python full_self_supervised_training.py "pointnet" "bottle"
#
#echo "CHAIR:: POINTNET"
#python full_self_supervised_training.py "pointnet" "chair"
#
#echo "LAPTOP:: POINTNET"
#python full_self_supervised_training.py "pointnet" "laptop"
#
#echo "SKATEBOARD:: POINTNET"
#python full_self_supervised_training.py "pointnet" "skateboard"
#
#echo "TABLE:: POINTNET"
#python full_self_supervised_training.py "pointnet" "table"

