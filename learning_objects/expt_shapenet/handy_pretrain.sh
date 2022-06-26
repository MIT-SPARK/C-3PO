echo "pre-training: motorcycle"
python training.py "point_transformer" "motorcycle" "supervised"

#echo "pre-training: mug"
#python training.py "point_transformer" "mug" "supervised"

#echo "pre-training: skateboard"
#python training.py "point_transformer" "skateboard" "supervised"

#echo "pre-training: table"
#python training.py "point_transformer" "table" "supervised"

#echo "pre-training: vessel"
#python training.py "point_transformer" "vessel" "supervised"

# evaluating point_transformer pre-trained models "supervised"

#echo "Evaluating: airplane, pre"
#python evaluate_proposed_model.py "point_transformer" "airplane" "pre" >> pt_eval_results.txt
#
#echo "Evaluating: bathtub, pre"
#python evaluate_proposed_model.py "point_transformer" "bathtub" "pre" >> pt_eval_results.txt
#
#echo "Evaluating: bed, pre"
#python evaluate_proposed_model.py "point_transformer" "bed" "pre" >> pt_eval_results.txt
#
#echo "Evaluating: cap, pre"
#python evaluate_proposed_model.py "point_transformer" "cap" "pre" >> pt_eval_results.txt
#
#echo "Evaluating: car, pre"
#python evaluate_proposed_model.py "point_transformer" "car" "pre" >> pt_eval_results.txt
#
#echo "Evaluating: helmet, pre"
#python evaluate_proposed_model.py "point_transformer" "helmet" "pre" >> pt_eval_results.txt
#
#echo "Evaluating: knife, pre"
#python evaluate_proposed_model.py "point_transformer" "knife" "pre" >> pt_eval_results.txt
#
#echo "Evaluating: laptop, pre"
#python evaluate_proposed_model.py "point_transformer" "laptop" "pre" >> pt_eval_results.txt
#
#echo "Evaluating: motorcycle, pre"
#python evaluate_proposed_model.py "point_transformer" "motorcycle" "pre" >> pt_eval_results.txt
#
#echo "Evaluating: mug, pre"
#python evaluate_proposed_model.py "point_transformer" "mug" "pre" >> pt_eval_results.txt
#
#echo "Evaluating: skateboard, pre"
#python evaluate_proposed_model.py "point_transformer" "skateboard" "pre" >> pt_eval_results.txt
#
#echo "Evaluating: table, pre"
#python evaluate_proposed_model.py "point_transformer" "table" "pre" >> pt_eval_results.txt

#echo "Evaluating: vessel, pre"
#python evaluate_proposed_model.py "point_transformer" "vessel" "pre" >> pt_eval_results.txt

