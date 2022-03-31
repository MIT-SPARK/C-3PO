FILE_NAME="./eval/handy_train_baseline_output.txt"

now=$(date)
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>PT>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME

#now=$(date)
#echo "$now"
#echo "pt: airplane"
#python train_baseline.py "point_transformer" "airplane" >> $FILE_NAME
#
#now=$(date)
#echo "$now"
#echo "pt: bathtub"
#python train_baseline.py "point_transformer" "bathtub" >> $FILE_NAME
#
#now=$(date)
#echo "$now"
#echo "pt: bed"
#python train_baseline.py "point_transformer" "bed" >> $FILE_NAME
#
#now=$(date)
#echo "$now"
#echo "pt: bottle"
#python train_baseline.py "point_transformer" "bottle" >> $FILE_NAME
#
#now=$(date)
#echo "$now"
#echo "pt: cap"
#python train_baseline.py "point_transformer" "cap" >> $FILE_NAME
#
#now=$(date)
#echo "$now"
#echo "pt: car"
#python train_baseline.py "point_transformer" "car" >> $FILE_NAME
#
#now=$(date)
#echo "$now"
#echo "pt: chair"
#python train_baseline.py "point_transformer" "chair" >> $FILE_NAME
#
#now=$(date)
#echo "$now"
#echo "pt: guitar"
#python train_baseline.py "point_transformer" "guitar" >> $FILE_NAME
#
#now=$(date)
#echo "$now"
#echo "pt: helmet"
#python train_baseline.py "point_transformer" "helmet" >> $FILE_NAME
#
#now=$(date)
#echo "$now"
#echo "pt: knife"
#python train_baseline.py "point_transformer" "knife" >> $FILE_NAME
#
#now=$(date)
#echo "$now"
#echo "pt: laptop"
#python train_baseline.py "point_transformer" "laptop" >> $FILE_NAME
#
#now=$(date)
#echo "$now"
#echo "pt: motorcycle"
#python train_baseline.py "point_transformer" "motorcycle" >> $FILE_NAME
#
#now=$(date)
#echo "$now"
#echo "pt: mug"
#python train_baseline.py "point_transformer" "mug" >> $FILE_NAME
#
#now=$(date)
#echo "$now"
#echo "pt: skateboard"
#python train_baseline.py "point_transformer" "skateboard" >> $FILE_NAME
#
#now=$(date)
#echo "$now"
#echo "pt: table"
#python train_baseline.py "point_transformer" "table" >> $FILE_NAME
#
#now=$(date)
#echo "$now"
#echo "pt: vessel"
#python train_baseline.py "point_transformer" "vessel" >> $FILE_NAME


now=$(date)
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>POINTNET>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: airplane"
python train_baseline.py "pointnet" "airplane" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: bathtub"
python train_baseline.py "pointnet" "bathtub" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: bed"
python train_baseline.py "pointnet" "bed" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: bottle"
python train_baseline.py "pointnet" "bottle" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: cap"
python train_baseline.py "pointnet" "cap" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: car"
python train_baseline.py "pointnet" "car" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: chair"
python train_baseline.py "pointnet" "chair" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: guitar"
python train_baseline.py "pointnet" "guitar" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: helmet"
python train_baseline.py "pointnet" "helmet" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: knife"
python train_baseline.py "pointnet" "knife" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: laptop"
python train_baseline.py "pointnet" "laptop" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: motorcycle"
python train_baseline.py "pointnet" "motorcycle" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: mug"
python train_baseline.py "pointnet" "mug" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: skateboard"
python train_baseline.py "pointnet" "skateboard" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: table"
python train_baseline.py "pointnet" "table" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: vessel"
python train_baseline.py "pointnet" "vessel" >> $FILE_NAME
