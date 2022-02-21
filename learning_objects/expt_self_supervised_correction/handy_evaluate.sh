echo "Evaluating: pointnet, motorcycle, post"
python evaluate_trained_model.py "pointnet" "motorcycle" "post" >> pointnet_eval_results.txt

echo "Evaluating: pt, motorcycle, post"
python evaluate_trained_model.py "point_transformer" "motorcycle" "post" >> pt_eval_results.txt

echo "Evaluating: pointnet, table, post"
python evaluate_trained_model.py "pointnet" "table" "post" >> pointnet_eval_results.txt

echo "Evaluating: pt, table, post"
python evaluate_trained_model.py "point_transformer" "table" "post" >> pt_eval_results.txt


#echo "Evaluating: airplane, pre"
#python evaluate_trained_model.py "pointnet" "airplane" "pre" >> pointnet_eval_pre_models2.txt

#echo "Evaluating: bathtub, pre"
#python evaluate_trained_model.py "pointnet" "bathtub" "pre" >> pointnet_eval_pre_models2.txt

#echo "Evaluating: bed, pre"
#python evaluate_trained_model.py "pointnet" "bed" "pre" >> pointnet_eval_pre_models2.txt

#echo "Evaluating: bed, pre"
#python evaluate_trained_model.py "point_transformer" "bed" "pre" >> pointnet_eval_pre_models2.txt

#echo "Evaluating: cap, pre"
#python evaluate_trained_model.py "pointnet" "cap" "pre" >> pointnet_eval_pre_models2.txt

#echo "Evaluating: car, pre"
#python evaluate_trained_model.py "pointnet" "car" "pre" >> pointnet_eval_pre_models2.txt

#echo "Evaluating: helmet, pre"
#python evaluate_trained_model.py "pointnet" "helmet" "pre" >> pointnet_eval_pre_models2.txt
#
#echo "Evaluating: knife, pre"
#python evaluate_trained_model.py "pointnet" "knife" "pre" >> pointnet_eval_pre_models2.txt

#echo "Evaluating: laptop, pre"
#python evaluate_trained_model.py "pointnet" "laptop" "pre" >> pointnet_eval_pre_models2.txt

#echo "Evaluating: motorcycle, pre"
#python evaluate_trained_model.py "pointnet" "motorcycle" "pre" >> pointnet_eval_pre_models2.txt
#
#echo "Evaluating: mug, pre"
#python evaluate_trained_model.py "pointnet" "mug" "pre" >> pointnet_eval_pre_models2.txt

#echo "Evaluating: skateboard, pre"
#python evaluate_trained_model.py "pointnet" "skateboard" "pre" >> pointnet_eval_pre_models2.txt
#
#echo "Evaluating: table, pre"
#python evaluate_trained_model.py "pointnet" "table" "pre" >> pointnet_eval_pre_models2.txt

#echo "Evaluating: vessel, pre"
#python evaluate_trained_model.py "pointnet" "vessel" "pre" >> pointnet_eval_pre_models2.txt
#