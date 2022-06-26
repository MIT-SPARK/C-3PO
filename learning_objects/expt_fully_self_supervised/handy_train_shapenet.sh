
echo "BOTTLE:: POINT TRANSFORMER"
python training.py "point_transformer" "bottle" "" "shapenet"

echo "CHAIR:: POINT TRANSFORMER"
python training.py "point_transformer" "chair" "" "shapenet"

echo "LAPTOP:: POINT TRANSFORMER"
python training.py "point_transformer" "laptop" "" "shapenet"

echo "SKATEBOARD:: POINT TRANSFORMER"
python training.py "point_transformer" "skateboard" "" "shapenet"

echo "TABLE:: POINT TRANSFORMER"
python training.py "point_transformer" "table" "" "shapenet"




echo "BOTTLE:: POINTNET"
python training.py "pointnet" "bottle" "" "shapenet"

echo "CHAIR:: POINTNET"
python training.py "pointnet" "chair" "" "shapenet"

echo "LAPTOP:: POINTNET"
python training.py "pointnet" "laptop" "" "shapenet"

echo "SKATEBOARD:: POINTNET"
python training.py "pointnet" "skateboard" "" "shapenet"

echo "TABLE:: POINTNET"
python training.py "pointnet" "table" "" "shapenet"

