
echo "BOTTLE:: POINT TRANSFORMER"
python self_supervised_training.py "point_transformer" "bottle"

echo "CHAIR:: POINT TRANSFORMER"
python self_supervised_training.py "point_transformer" "chair"

echo "LAPTOP:: POINT TRANSFORMER"
python self_supervised_training.py "point_transformer" "laptop"

echo "SKATEBOARD:: POINT TRANSFORMER"
python self_supervised_training.py "point_transformer" "skateboard"

echo "TABLE:: POINT TRANSFORMER"
python self_supervised_training.py "point_transformer" "table"




#echo "BOTTLE:: POINTNET"
#python self_supervised_training.py "pointnet" "bottle"

#echo "CHAIR:: POINTNET"
#python self_supervised_training.py "pointnet" "chair"

#echo "LAPTOP:: POINTNET"
#python self_supervised_training.py "pointnet" "laptop"

#echo "SKATEBOARD:: POINTNET"
#python self_supervised_training.py "pointnet" "skateboard"

#echo "TABLE:: POINTNET"
#python self_supervised_training.py "pointnet" "table"

