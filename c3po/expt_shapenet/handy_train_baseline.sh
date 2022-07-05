FILE_NAME="./eval/handy_train_baseline_output_main_temp.txt"

now=$(date)
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>PT>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME

#now=$(date)
#echo "$now"
#echo "pt: airplane"
#python training.py "point_transformer" "airplane" "baseline" >> $FILE_NAME
#
#now=$(date)
#echo "$now"
#echo "pt: bathtub"
#python training.py "point_transformer" "bathtub" "baseline" >> $FILE_NAME
#
#now=$(date)
#echo "$now"
#echo "pt: bed"
#python training.py "point_transformer" "bed" "baseline" >> $FILE_NAME
#
#now=$(date)
#echo "$now"
#echo "pt: bottle"
#python training.py "point_transformer" "bottle" "baseline" >> $FILE_NAME
#
#now=$(date)
#echo "$now"
#echo "pt: cap"
#python training.py "point_transformer" "cap" "baseline" >> $FILE_NAME
#
#now=$(date)
#echo "$now"
#echo "pt: car"
#python training.py "point_transformer" "car" "baseline" >> $FILE_NAME
#
#now=$(date)
#echo "$now"
#echo "pt: chair"
#python training.py "point_transformer" "chair" "baseline" >> $FILE_NAME
#
#now=$(date)
#echo "$now"
#echo "pt: guitar"
#python training.py "point_transformer" "guitar" "baseline" >> $FILE_NAME
#
#now=$(date)
#echo "$now"
#echo "pt: helmet"
#python training.py "point_transformer" "helmet" "baseline" >> $FILE_NAME
#
#now=$(date)
#echo "$now"
#echo "pt: knife"
#python training.py "point_transformer" "knife" "baseline" >> $FILE_NAME
#
#now=$(date)
#echo "$now"
#echo "pt: laptop"
#python training.py "point_transformer" "laptop" "baseline" >> $FILE_NAME
#
#now=$(date)
#echo "$now"
#echo "pt: motorcycle"
#python training.py "point_transformer" "motorcycle" "baseline" >> $FILE_NAME
#
#now=$(date)
#echo "$now"
#echo "pt: mug"
#python training.py "point_transformer" "mug" "baseline" >> $FILE_NAME
#
#now=$(date)
#echo "$now"
#echo "pt: skateboard"
#python training.py "point_transformer" "skateboard" "baseline" >> $FILE_NAME
#
#now=$(date)
#echo "$now"
#echo "pt: table"
#python training.py "point_transformer" "table" "baseline" >> $FILE_NAME
#
#now=$(date)
#echo "$now"
#echo "pt: vessel"
#python training.py "point_transformer" "vessel" "baseline" >> $FILE_NAME


now=$(date)
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>POINTNET>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: airplane"
python training.py "pointnet" "airplane" "baseline" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: bathtub"
python training.py "pointnet" "bathtub" "baseline" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: bed"
python training.py "pointnet" "bed" "baseline" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: bottle"
python training.py "pointnet" "bottle" "baseline" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: cap"
python training.py "pointnet" "cap" "baseline" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: car"
python training.py "pointnet" "car" "baseline" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: chair"
python training.py "pointnet" "chair" "baseline" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: guitar"
python training.py "pointnet" "guitar" "baseline" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: helmet"
python training.py "pointnet" "helmet" "baseline" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: knife"
python training.py "pointnet" "knife" "baseline" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: laptop"
python training.py "pointnet" "laptop" "baseline" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: motorcycle"
python training.py "pointnet" "motorcycle" "baseline" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: mug"
python training.py "pointnet" "mug" "baseline" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: skateboard"
python training.py "pointnet" "skateboard" "baseline" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: table"
python training.py "pointnet" "table" "baseline" >> $FILE_NAME

now=$(date)
echo "$now"
echo "pointnet: vessel"
python training.py "pointnet" "vessel" "baseline" >> $FILE_NAME
