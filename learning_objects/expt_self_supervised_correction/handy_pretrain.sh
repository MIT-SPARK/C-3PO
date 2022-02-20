#echo "working on: airplane, pre"
#python evaluate_trained_model.py "pointnet" "airplane" "pre" >> pointnet_eval_pre_models.txt
#
#echo "working on: bathtub, pre"
#python evaluate_trained_model.py "pointnet" "bathtub" "pre" >> pointnet_eval_pre_models.txt
#
#echo "working on: bed, pre"
#python evaluate_trained_model.py "pointnet" "bed" "pre" >> pointnet_eval_pre_models.txt
#
#echo "working on: cap, pre"
#python evaluate_trained_model.py "pointnet" "cap" "pre" >> pointnet_eval_pre_models.txt
#
#echo "working on: helmet, pre"
#python evaluate_trained_model.py "pointnet" "helmet" "pre" >> pointnet_eval_pre_models.txt
#
#echo "working on: knife, pre"
#python evaluate_trained_model.py "pointnet" "knife" "pre" >> pointnet_eval_pre_models.txt
#
#echo "working on: laptop, pre"
#python evaluate_trained_model.py "pointnet" "laptop" "pre" >> pointnet_eval_pre_models.txt
#
#echo "working on: motorcycle, pre"
#python evaluate_trained_model.py "pointnet" "motorcycle" "pre" >> pointnet_eval_pre_models.txt
#
#echo "working on: mug, pre"
#python evaluate_trained_model.py "pointnet" "mug" "pre" >> pointnet_eval_pre_models.txt

echo "working on: skateboard, pre"
python evaluate_trained_model.py "pointnet" "skateboard" "pre" >> pointnet_eval_pre_models.txt

echo "working on: table, pre"
python evaluate_trained_model.py "pointnet" "table" "pre" >> pointnet_eval_pre_models.txt

echo "working on: vessel, pre"
python evaluate_trained_model.py "pointnet" "vessel" "pre" >> pointnet_eval_pre_models.txt
