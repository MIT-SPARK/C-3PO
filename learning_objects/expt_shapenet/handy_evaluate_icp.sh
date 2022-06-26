FILE_NAME="./eval/eval_icp.txt"

now=$(date +'%c')
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "RANSAC+ICP: no corrector"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME

echo "airplane"
python evaluate_icp.py "airplane" "ransac" "nc" >> $FILE_NAME
echo "bathtub"
python evaluate_icp.py "bathtub" "ransac" "nc" >> $FILE_NAME
echo "bed"
python evaluate_icp.py "bed" "ransac" "nc" >> $FILE_NAME
echo "bottle"
python evaluate_icp.py "bottle" "ransac" "nc" >> $FILE_NAME
echo "cap"
python evaluate_icp.py "cap" "ransac" "nc" >> $FILE_NAME
echo "car"
python evaluate_icp.py "car" "ransac" "nc" >> $FILE_NAME
echo "chair"
python evaluate_icp.py "chair" "ransac" "nc" >> $FILE_NAME
echo "guitar"
python evaluate_icp.py "guitar" "ransac" "nc" >> $FILE_NAME
echo "helmet"
python evaluate_icp.py "helmet" "ransac" "nc" >> $FILE_NAME
echo "knife"
python evaluate_icp.py "knife" "ransac" "nc" >> $FILE_NAME
echo "laptop"
python evaluate_icp.py "laptop" "ransac" "nc" >> $FILE_NAME
echo "motorcycle"
python evaluate_icp.py "motorcycle" "ransac" "nc" >> $FILE_NAME
echo "mug"
python evaluate_icp.py "mug" "ransac" "nc" >> $FILE_NAME
echo "skateboard"
python evaluate_icp.py "skateboard" "ransac" "nc" >> $FILE_NAME
echo "table"
python evaluate_icp.py "table" "ransac" "nc" >> $FILE_NAME
echo "vessel"
python evaluate_icp.py "vessel" "ransac" "nc" >> $FILE_NAME


now=$(date +'%c')
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "Keypoint Registration+ICP: no corrector"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME

echo "airplane"
python evaluate_icp.py "airplane" "none" "nc" >> $FILE_NAME
echo "bathtub"
python evaluate_icp.py "bathtub" "none" "nc" >> $FILE_NAME
echo "bed"
python evaluate_icp.py "bed" "none" "nc" >> $FILE_NAME
echo "bottle"
python evaluate_icp.py "bottle" "none" "nc" >> $FILE_NAME
echo "cap"
python evaluate_icp.py "cap" "none" "nc" >> $FILE_NAME
echo "car"
python evaluate_icp.py "car" "none" "nc" >> $FILE_NAME
echo "chair"
python evaluate_icp.py "chair" "none" "nc" >> $FILE_NAME
echo "guitar"
python evaluate_icp.py "guitar" "none" "nc" >> $FILE_NAME
echo "helmet"
python evaluate_icp.py "helmet" "none" "nc" >> $FILE_NAME
echo "knife"
python evaluate_icp.py "knife" "none" "nc" >> $FILE_NAME
echo "laptop"
python evaluate_icp.py "laptop" "none" "nc" >> $FILE_NAME
echo "motorcycle"
python evaluate_icp.py "motorcycle" "none" "nc" >> $FILE_NAME
echo "mug"
python evaluate_icp.py "mug" "none" "nc" >> $FILE_NAME
echo "skateboard"
python evaluate_icp.py "skateboard" "none" "nc" >> $FILE_NAME
echo "table"
python evaluate_icp.py "table" "none" "nc" >> $FILE_NAME
echo "vessel"
python evaluate_icp.py "vessel" "none" "nc" >> $FILE_NAME


#now=$(date +'%c')
#echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
#echo "TEASER+ICP: no corrector"
#echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
#echo "$now" >> $FILE_NAME
#
#echo "airplane"
#python evaluate_icp.py "airplane" "teaser" "nc" >> $FILE_NAME
#echo "bathtub"
#python evaluate_icp.py "bathtub" "teaser" "nc" >> $FILE_NAME
#echo "bed"
#python evaluate_icp.py "bed" "teaser" "nc" >> $FILE_NAME
#echo "bottle"
#python evaluate_icp.py "bottle" "teaser" "nc" >> $FILE_NAME
#echo "cap"
#python evaluate_icp.py "cap" "teaser" "nc" >> $FILE_NAME
#echo "car"
#python evaluate_icp.py "car" "teaser" "nc" >> $FILE_NAME
#echo "chair"
#python evaluate_icp.py "chair" "teaser" "nc" >> $FILE_NAME
#echo "guitar"
#python evaluate_icp.py "guitar" "teaser" "nc" >> $FILE_NAME
#echo "helmet"
#python evaluate_icp.py "helmet" "teaser" "nc" >> $FILE_NAME
#echo "knife"
#python evaluate_icp.py "knife" "teaser" "nc" >> $FILE_NAME
#echo "laptop"
#python evaluate_icp.py "laptop" "teaser" "nc" >> $FILE_NAME
#echo "motorcycle"
#python evaluate_icp.py "motorcycle" "teaser" "nc" >> $FILE_NAME
#echo "mug"
#python evaluate_icp.py "mug" "teaser" "nc" >> $FILE_NAME
#echo "skateboard"
#python evaluate_icp.py "skateboard" "teaser" "nc" >> $FILE_NAME
#echo "table"
#python evaluate_icp.py "table" "teaser" "nc" >> $FILE_NAME
#echo "vessel"
#python evaluate_icp.py "vessel" "teaser" "nc" >> $FILE_NAME
#
#
now=$(date +'%c')
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "RANSAC+ICP: w corrector"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME

echo "airplane"
python evaluate_icp.py "airplane" "ransac" "c" >> $FILE_NAME
echo "bathtub"
python evaluate_icp.py "bathtub" "ransac" "c" >> $FILE_NAME
echo "bed"
python evaluate_icp.py "bed" "ransac" "c" >> $FILE_NAME
echo "bottle"
python evaluate_icp.py "bottle" "ransac" "c" >> $FILE_NAME
echo "cap"
python evaluate_icp.py "cap" "ransac" "c" >> $FILE_NAME
echo "car"
python evaluate_icp.py "car" "ransac" "c" >> $FILE_NAME
echo "chair"
python evaluate_icp.py "chair" "ransac" "c" >> $FILE_NAME
echo "guitar"
python evaluate_icp.py "guitar" "ransac" "c" >> $FILE_NAME
echo "helmet"
python evaluate_icp.py "helmet" "ransac" "c" >> $FILE_NAME
echo "knife"
python evaluate_icp.py "knife" "ransac" "c" >> $FILE_NAME
echo "laptop"
python evaluate_icp.py "laptop" "ransac" "c" >> $FILE_NAME
echo "motorcycle"
python evaluate_icp.py "motorcycle" "ransac" "c" >> $FILE_NAME
echo "mug"
python evaluate_icp.py "mug" "ransac" "c" >> $FILE_NAME
echo "skateboard"
python evaluate_icp.py "skateboard" "ransac" "c" >> $FILE_NAME
echo "table"
python evaluate_icp.py "table" "ransac" "c" >> $FILE_NAME
echo "vessel"
python evaluate_icp.py "vessel" "ransac" "c" >> $FILE_NAME


now=$(date +'%c')
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "Keypoint Registration+ICP: corrector"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME

echo "airplane"
python evaluate_icp.py "airplane" "none" "c" >> $FILE_NAME
echo "bathtub"
python evaluate_icp.py "bathtub" "none" "c" >> $FILE_NAME
echo "bed"
python evaluate_icp.py "bed" "none" "c" >> $FILE_NAME
echo "bottle"
python evaluate_icp.py "bottle" "none" "c" >> $FILE_NAME
echo "cap"
python evaluate_icp.py "cap" "none" "c" >> $FILE_NAME
echo "car"
python evaluate_icp.py "car" "none" "c" >> $FILE_NAME
echo "chair"
python evaluate_icp.py "chair" "none" "c" >> $FILE_NAME
echo "guitar"
python evaluate_icp.py "guitar" "none" "c" >> $FILE_NAME
echo "helmet"
python evaluate_icp.py "helmet" "none" "c" >> $FILE_NAME
echo "knife"
python evaluate_icp.py "knife" "none" "c" >> $FILE_NAME
echo "laptop"
python evaluate_icp.py "laptop" "none" "c" >> $FILE_NAME
echo "motorcycle"
python evaluate_icp.py "motorcycle" "none" "c" >> $FILE_NAME
echo "mug"
python evaluate_icp.py "mug" "none" "c" >> $FILE_NAME
echo "skateboard"
python evaluate_icp.py "skateboard" "none" "c" >> $FILE_NAME
echo "table"
python evaluate_icp.py "table" "none" "c" >> $FILE_NAME
echo "vessel"
python evaluate_icp.py "vessel" "none" "c" >> $FILE_NAME


#now=$(date +'%c')
#echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
#echo "TEASER+ICP: w corrector"
#echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
#echo "$now" >> $FILE_NAME
#
#echo "airplane"
#python evaluate_icp.py "airplane" "teaser" "c" >> $FILE_NAME
#echo "bathtub"
#python evaluate_icp.py "bathtub" "teaser" "c" >> $FILE_NAME
#echo "bed"
#python evaluate_icp.py "bed" "teaser" "c" >> $FILE_NAME
#echo "bottle"
#python evaluate_icp.py "bottle" "teaser" "c" >> $FILE_NAME
#echo "cap"
#python evaluate_icp.py "cap" "teaser" "c" >> $FILE_NAME
#echo "car"
#python evaluate_icp.py "car" "teaser" "c" >> $FILE_NAME
#echo "chair"
#python evaluate_icp.py "chair" "teaser" "c" >> $FILE_NAME
#echo "guitar"
#python evaluate_icp.py "guitar" "teaser" "c" >> $FILE_NAME
#echo "helmet"
#python evaluate_icp.py "helmet" "teaser" "c" >> $FILE_NAME
#echo "knife"
#python evaluate_icp.py "knife" "teaser" "c" >> $FILE_NAME
#echo "laptop"
#python evaluate_icp.py "laptop" "teaser" "c" >> $FILE_NAME
#echo "motorcycle"
#python evaluate_icp.py "motorcycle" "teaser" "c" >> $FILE_NAME
#echo "mug"
#python evaluate_icp.py "mug" "teaser" "c" >> $FILE_NAME
#echo "skateboard"
#python evaluate_icp.py "skateboard" "teaser" "c" >> $FILE_NAME
#echo "table"
#python evaluate_icp.py "table" "teaser" "c" >> $FILE_NAME
#echo "vessel"
#python evaluate_icp.py "vessel" "teaser" "c" >> $FILE_NAME
#
#
