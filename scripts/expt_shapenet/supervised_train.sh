cd ../../c3po/expt_shapenet
##########################################################################

echo "AIRPLANE:: POINT TRANSFORMER"
python training.py "point_transformer" "airplane" "supervised"

echo "AIRPLANE:: POINTNET"
python training.py "pointnet" "airplane" "supervised"

##########################################################################

echo "BATHTUB:: POINT TRANSFORMER"
python training.py "point_transformer" "bathtub" "supervised"

echo "BATHTUB:: POINTNET"
python training.py "pointnet" "bathtub" "supervised"

##########################################################################

echo "BED:: POINT TRANSFORMER"
python training.py "point_transformer" "bed" "supervised"

echo "BED:: POINTNET"
python training.py "pointnet" "bed" "supervised"

##########################################################################

echo "BOTTLE:: POINT TRANSFORMER"
python training.py "point_transformer" "bottle" "supervised"

echo "BOTTLE:: POINTNET"
python training.py "pointnet" "bottle" "supervised"

##########################################################################

echo "CAP:: POINT TRANSFORMER"
python training.py "point_transformer" "cap" "supervised"

echo "CAP:: POINTNET"
python training.py "pointnet" "cap" "supervised"

###############################################

echo "CAR:: POINT TRANSFORMER"
python training.py "point_transformer" "car" "supervised"

echo "CAR:: POINTNET"
python training.py "pointnet" "car" "supervised"

###############################################

echo "CHAIR:: POINT TRANSFORMER"
python training.py "point_transformer" "chair" "supervised"

echo "CHAIR:: POINTNET"
python training.py "pointnet" "chair" "supervised"

###############################################

echo "GUITAR:: POINT TRANSFORMER"
python training.py "point_transformer" "guitar" "supervised"

echo "GUITAR:: POINTNET"
python training.py "pointnet" "guitar" "supervised"

##########################################################################

echo "HELMET:: POINT TRANSFORMER"
python training.py "point_transformer" "helmet" "supervised"

echo "HELMET:: POINTNET"
python training.py "pointnet" "helmet" "supervised"

##########################################################################

echo "KNIFE:: POINT TRANSFORMER"
python training.py "point_transformer" "knife" "supervised"

echo "KNIFE:: POINTNET"
python training.py "pointnet" "knife" "supervised"

##########################################################################

echo "LAPTOP:: POINT TRANSFORMER"
python training.py "point_transformer" "laptop" "supervised"

echo "LAPTOP:: POINTNET"
python training.py "pointnet" "laptop" "supervised"

##########################################################################

echo "MOTORCYCLE:: POINT TRANSFORMER"
python training.py "point_transformer" "motorcycle" "supervised"

echo "MOTORCYCLE:: POINTNET"
python training.py "pointnet" "motorcycle" "supervised"

##########################################################################

echo "MUG:: POINT TRANSFORMER"
python training.py "point_transformer" "mug" "supervised"
#
#echo "MUG:: POINTNET"
#python training.py "pointnet" "mug" "supervised"

##########################################################################

echo "SKATEBOARD:: POINT TRANSFORMER"
python training.py "point_transformer" "skateboard" "supervised"

echo "SKATEBOARD:: POINTNET"
python training.py "pointnet" "skateboard" "supervised"

##########################################################################

echo "TABLE:: POINT TRANSFORMER"
python training.py "point_transformer" "table" "supervised"

echo "TABLE:: POINTNET"
python training.py "pointnet" "table" "supervised"

##########################################################################

echo "VESSEL:: POINT TRANSFORMER"
python training.py "point_transformer" "vessel" "supervised"

echo "VESSEL:: POINTNET"
python training.py "pointnet" "vessel" "supervised"
