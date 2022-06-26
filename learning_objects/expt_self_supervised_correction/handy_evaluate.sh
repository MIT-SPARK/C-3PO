FILE_NAME="./eval/model_eval_main.txt"

now=$(date +'%c')
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME

############################################################################################
echo "EVALUATING POINT TRANSFORMER" >> $FILE_NAME
############################################################################################

#echo "Evaluating: pt, airplane, pre"
#python evaluate_trained_model.py "point_transformer" "airplane" "pre" >> $FILE_NAME

echo "Evaluating: pt, airplane, post"
python evaluate_trained_model.py "point_transformer" "airplane" "post" >> $FILE_NAME

############################################################################################

#echo "Evaluating: pt, bathtub, pre"
#python evaluate_trained_model.py "point_transformer" "bathtub" "pre" >> $FILE_NAME

echo "Evaluating: pt, bathtub, post"
python evaluate_trained_model.py "point_transformer" "bathtub" "post" >> $FILE_NAME

############################################################################################

#echo "Evaluating: pt, bed, pre"
#python evaluate_trained_model.py "point_transformer" "bed" "pre" >> $FILE_NAME

echo "Evaluating: pt, bed, post"
python evaluate_trained_model.py "point_transformer" "bed" "post" >> $FILE_NAME

############################################################################################

#echo "Evaluating: pt, bottle, pre"
#python evaluate_trained_model.py "point_transformer" "bottle" "pre" #>> $FILE_NAME
#
echo "Evaluating: pt, bottle, post"
python evaluate_trained_model.py "point_transformer" "bottle" "post" >> $FILE_NAME
#
###########################################################################################

#echo "Evaluating: pt, cap, pre"
#python evaluate_trained_model.py "point_transformer" "cap" "pre" >> $FILE_NAME

echo "Evaluating: pt, cap, post"
python evaluate_trained_model.py "point_transformer" "cap" "post" >> $FILE_NAME
#
###########################################################################################
#
#echo "Evaluating: pt, car, pre"
#python evaluate_trained_model.py "point_transformer" "car" "pre" >> $FILE_NAME

echo "Evaluating: pt, car, post"
python evaluate_trained_model.py "point_transformer" "car" "post" >> $FILE_NAME
#
###########################################################################################

#echo "Evaluating: pt, chair, pre"
#python evaluate_trained_model.py "point_transformer" "chair" "pre" >> $FILE_NAME

echo "Evaluating: pt, chair, post"
python evaluate_trained_model.py "point_transformer" "chair" "post" >> $FILE_NAME

############################################################################################

#echo "Evaluating: pt, guitar, pre"
#python evaluate_trained_model.py "point_transformer" "guitar" "pre" >> $FILE_NAME

echo "Evaluating: pt, guitar, post"
python evaluate_trained_model.py "point_transformer" "guitar" "post" >> $FILE_NAME

############################################################################################

#echo "Evaluating: pt, helmet, pre"
#python evaluate_trained_model.py "point_transformer" "helmet" "pre" >> $FILE_NAME

echo "Evaluating: pt, helmet, post"
python evaluate_trained_model.py "point_transformer" "helmet" "post" >> $FILE_NAME

############################################################################################

#echo "Evaluating: pt, knife, pre"
#python evaluate_trained_model.py "point_transformer" "knife" "pre" >> $FILE_NAME

echo "Evaluating: pt, knife, post"
python evaluate_trained_model.py "point_transformer" "knife" "post" >> $FILE_NAME

###########################################################################################

#echo "Evaluating: pt, laptop, pre"
#python evaluate_trained_model.py "point_transformer" "laptop" "pre" >> $FILE_NAME

echo "Evaluating: pt, laptop, post"
python evaluate_trained_model.py "point_transformer" "laptop" "post" >> $FILE_NAME

############################################################################################

#echo "Evaluating: pt, motorcycle, pre"
#python evaluate_trained_model.py "point_transformer" "motorcycle" "pre" >> $FILE_NAME

echo "Evaluating: pt, motorcycle, post"
python evaluate_trained_model.py "point_transformer" "motorcycle" "post" >> $FILE_NAME

############################################################################################

#echo "Evaluating: pt, mug, pre"
#python evaluate_trained_model.py "point_transformer" "mug" "pre" >> $FILE_NAME

echo "Evaluating: pt, mug, post"
python evaluate_trained_model.py "point_transformer" "mug" "post" >> $FILE_NAME

###########################################################################################

#echo "Evaluating: pt, skateboard, pre"
#python evaluate_trained_model.py "point_transformer" "skateboard" "pre" >> $FILE_NAME

echo "Evaluating: pt, skateboard, post"
python evaluate_trained_model.py "point_transformer" "skateboard" "post" >> $FILE_NAME

###########################################################################################

#echo "Evaluating: pt, table, pre"
#python evaluate_trained_model.py "point_transformer" "table" "pre" >> $FILE_NAME

echo "Evaluating: pt, table, post"
python evaluate_trained_model.py "point_transformer" "table" "post" >> $FILE_NAME

###########################################################################################

#echo "Evaluating: pt, vessel, pre"
#python evaluate_trained_model.py "point_transformer" "vessel" "pre" >> $FILE_NAME

echo "Evaluating: pt, vessel, post"
python evaluate_trained_model.py "point_transformer" "vessel" "post" >> $FILE_NAME
#
#
#
#
###########################################################################################
#echo "EVALUATING POINTNET" >> $FILE_NAME
###########################################################################################

#echo "Evaluating: pointnet, airplane, pre"
#python evaluate_trained_model.py "pointnet" "airplane" "pre" >> $FILE_NAME
#
#echo "Evaluating: pointnet, airplane, post"
#python evaluate_trained_model.py "pointnet" "airplane" "post" >> $FILE_NAME
#
############################################################################################
#
#echo "Evaluating: pointnet, bathtub, pre"
#python evaluate_trained_model.py "pointnet" "bathtub" "pre" >> $FILE_NAME
#
#echo "Evaluating: pointnet, bathtub, post"
#python evaluate_trained_model.py "pointnet" "bathtub" "post" >> $FILE_NAME
#
############################################################################################
#
#echo "Evaluating: pointnet, bed, pre"
#python evaluate_trained_model.py "pointnet" "bed" "pre" >> $FILE_NAME
#
#echo "Evaluating: pointnet, bed, post"
#python evaluate_trained_model.py "pointnet" "bed" "post" >> $FILE_NAME
#
############################################################################################
#
#echo "Evaluating: pointnet, bottle, pre"
#python evaluate_trained_model.py "pointnet" "bottle" "pre" >> $FILE_NAME
#
#echo "Evaluating: pointnet, bottle, post"
#python evaluate_trained_model.py "pointnet" "bottle" "post" >> $FILE_NAME

############################################################################################
#
#echo "Evaluating: pointnet, cap, pre"
#python evaluate_trained_model.py "pointnet" "cap" "pre" >> $FILE_NAME
#
#echo "Evaluating: pointnet, cap, post"
#python evaluate_trained_model.py "pointnet" "cap" "post" >> $FILE_NAME
#
############################################################################################

#echo "Evaluating: pointnet, car, pre"
#python evaluate_trained_model.py "pointnet" "car" "pre" >> $FILE_NAME
#
#echo "Evaluating: pointnet, car, post"
#python evaluate_trained_model.py "pointnet" "car" "post" >> $FILE_NAME

############################################################################################
#
#echo "Evaluating: pointnet, chair, pre"
#python evaluate_trained_model.py "pointnet" "chair" "pre" >> $FILE_NAME
#
#echo "Evaluating: pointnet, chair, post"
#python evaluate_trained_model.py "pointnet" "chair" "post" >> $FILE_NAME
#
############################################################################################
#
#echo "Evaluating: pointnet, guitar, pre"
#python evaluate_trained_model.py "pointnet" "guitar" "pre" >> $FILE_NAME
#
#echo "Evaluating: pointnet, guitar, post"
#python evaluate_trained_model.py "pointnet" "guitar" "post" >> $FILE_NAME
#
############################################################################################
#
#echo "Evaluating: pointnet, helmet, pre"
#python evaluate_trained_model.py "pointnet" "helmet" "pre" >> $FILE_NAME
#
#echo "Evaluating: pointnet, helmet, post"
#python evaluate_trained_model.py "pointnet" "helmet" "post" >> $FILE_NAME
#
############################################################################################
#
#echo "Evaluating: pointnet, knife, pre"
#python evaluate_trained_model.py "pointnet" "knife" "pre" >> $FILE_NAME
#
#echo "Evaluating: pointnet, knife, post"
#python evaluate_trained_model.py "pointnet" "knife" "post" >> $FILE_NAME
#
############################################################################################
#
#echo "Evaluating: pointnet, laptop, pre"
#python evaluate_trained_model.py "pointnet" "laptop" "pre" >> $FILE_NAME
#
#echo "Evaluating: pointnet, laptop, post"
#python evaluate_trained_model.py "pointnet" "laptop" "post" >> $FILE_NAME
#
############################################################################################
#
#echo "Evaluating: pointnet, motorcycle, pre"
#python evaluate_trained_model.py "pointnet" "motorcycle" "pre" >> $FILE_NAME
#
#echo "Evaluating: pointnet, motorcycle, post"
#python evaluate_trained_model.py "pointnet" "motorcycle" "post" >> $FILE_NAME
#
############################################################################################
#
#echo "Evaluating: pointnet, mug, pre"
#python evaluate_trained_model.py "pointnet" "mug" "pre" >> $FILE_NAME
#
#echo "Evaluating: pointnet, mug, post"
#python evaluate_trained_model.py "pointnet" "mug" "post" >> $FILE_NAME
#
############################################################################################
#
#echo "Evaluating: pointnet, skateboard, pre"
#python evaluate_trained_model.py "pointnet" "skateboard" "pre" >> $FILE_NAME
#
#echo "Evaluating: pointnet, skateboard, post"
#python evaluate_trained_model.py "pointnet" "skateboard" "post" >> $FILE_NAME
#
############################################################################################
#
#echo "Evaluating: pointnet, table, pre"
#python evaluate_trained_model.py "pointnet" "table" "pre" >> $FILE_NAME
#
#echo "Evaluating: pointnet, table, post"
#python evaluate_trained_model.py "pointnet" "table" "post" >> $FILE_NAME
#
############################################################################################
#
#echo "Evaluating: pointnet, vessel, pre"
#python evaluate_trained_model.py "pointnet" "vessel" "pre" >> $FILE_NAME
#
#echo "Evaluating: pointnet, vessel, post"
#python evaluate_trained_model.py "pointnet" "vessel" "post" >> $FILE_NAME
#
