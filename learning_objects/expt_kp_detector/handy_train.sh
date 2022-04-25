FILE_NAME="./eval/train_out.txt"

now=$(date +'%c')
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME


#########################################################################

#echo $(date +'%c')
#echo "AIRPLANE:: POINT TRANSFORMER"
#python train.py "point_transformer" "airplane" #>> $FILE_NAME

#echo $(date +'%c')
#echo "AIRPLANE:: POINTNET"
#python train.py "pointnet" "airplane" >> $FILE_NAME

##########################################################################

#echo $(date +'%c')
#echo "BATHTUB:: POINT TRANSFORMER"
#python train.py "point_transformer" "bathtub" #>> $FILE_NAME

#echo $(date +'%c')
#echo "BATHTUB:: POINTNET"
#python train.py "pointnet" "bathtub" >> $FILE_NAME

##########################################################################

#echo $(date +'%c')
#echo "BED:: POINT TRANSFORMER"
#python train.py "point_transformer" "bed" >> $FILE_NAME

#echo $(date +'%c')
#echo "BED:: POINTNET"
#python train.py "pointnet" "bed" >> $FILE_NAME

##########################################################################

echo $(date +'%c')
echo "BOTTLE:: POINT TRANSFORMER"
python train.py "point_transformer" "bottle" #>> $FILE_NAME

#echo $(date +'%c')
#echo "BOTTLE:: POINTNET"
#python train.py "pointnet" "bottle" >> $FILE_NAME

##########################################################################

#echo $(date +'%c')
#echo "CAP:: POINT TRANSFORMER"
#python train.py "point_transformer" "cap" >> $FILE_NAME

#echo $(date +'%c')
#echo "CAP:: POINTNET"
#python train.py "pointnet" "cap" >> $FILE_NAME

###############################################

#echo $(date +'%c')
#echo "CAR:: POINT TRANSFORMER"
#python train.py "point_transformer" "car" >> $FILE_NAME

#echo $(date +'%c')
#echo "CAR:: POINTNET"
#python train.py "pointnet" "car" >> $FILE_NAME

###############################################

echo $(date +'%c')
echo "CHAIR:: POINT TRANSFORMER"
python train.py "point_transformer" "chair" #>> $FILE_NAME

#echo $(date +'%c')
#echo "CHAIR:: POINTNET"
#python train.py "pointnet" "chair" >> $FILE_NAME

###############################################

echo $(date +'%c')
echo "GUITAR:: POINT TRANSFORMER"
python train.py "point_transformer" "guitar" #>> $FILE_NAME

#echo $(date +'%c')
#echo "GUITAR:: POINTNET"
#python train.py "pointnet" "guitar" >> $FILE_NAME

##########################################################################

#echo $(date +'%c')
#echo "HELMET:: POINT TRANSFORMER"
#python train.py "point_transformer" "helmet" >> $FILE_NAME

#echo $(date +'%c')
#echo "HELMET:: POINTNET"
#python train.py "pointnet" "helmet" >> $FILE_NAME

##########################################################################

#echo $(date +'%c')
#echo "KNIFE:: POINT TRANSFORMER"
#python train.py "point_transformer" "knife" #>> $FILE_NAME

#echo $(date +'%c')
#echo "KNIFE:: POINTNET"
#python train.py "pointnet" "knife" >> $FILE_NAME

##########################################################################

echo $(date +'%c')
echo "LAPTOP:: POINT TRANSFORMER"
python train.py "point_transformer" "laptop" #>> $FILE_NAME

#echo $(date +'%c')
#echo "LAPTOP:: POINTNET"
#python train.py "pointnet" "laptop" >> $FILE_NAME

##########################################################################

#echo $(date +'%c')
#echo "MOTORCYCLE:: POINT TRANSFORMER"
#python train.py "point_transformer" "motorcycle" >> $FILE_NAME

#echo $(date +'%c')
#echo "MOTORCYCLE:: POINTNET"
#python train.py "pointnet" "motorcycle" >> $FILE_NAME

##########################################################################

#echo $(date +'%c')
#echo "MUG:: POINT TRANSFORMER"
#python train.py "point_transformer" "mug" >> $FILE_NAME

#echo $(date +'%c')
#echo "MUG:: POINTNET"
#python train.py "pointnet" "mug" >> $FILE_NAME

##########################################################################

echo $(date +'%c')
echo "SKATEBOARD:: POINT TRANSFORMER"
python train.py "point_transformer" "skateboard" #>> $FILE_NAME

#echo $(date +'%c')
#echo "SKATEBOARD:: POINTNET"
#python train.py "pointnet" "skateboard" >> $FILE_NAME

##########################################################################

echo $(date +'%c')
echo "TABLE:: POINT TRANSFORMER"
python train.py "point_transformer" "table" #>> $FILE_NAME

#echo $(date +'%c')
#echo "TABLE:: POINTNET"
#python train.py "pointnet" "table" >> $FILE_NAME

##########################################################################

#echo $(date +'%c')
#echo "VESSEL:: POINT TRANSFORMER"
#python train.py "point_transformer" "vessel" >> $FILE_NAME

#echo $(date +'%c')
#echo "VESSEL:: POINTNET"
#python train.py "pointnet" "vessel" >> $FILE_NAME
