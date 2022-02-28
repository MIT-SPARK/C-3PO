FILE_NAME="pt_eval_2pcent."

echo "Evaluating: pt, airplane, pre"
python evaluate_trained_model.py "point_transformer" "airplane" "pre" >> $FILE_NAME

echo "Evaluating: pt, airplane, post"
python evaluate_trained_model.py "point_transformer" "airplane" "post" >> $FILE_NAME

############################################################################################

echo "Evaluating: pt, bathtub, pre"
python evaluate_trained_model.py "point_transformer" "bathtub" "pre" >> $FILE_NAME

echo "Evaluating: pt, bathtub, post"
python evaluate_trained_model.py "point_transformer" "bathtub" "post" >> $FILE_NAME

############################################################################################

echo "Evaluating: pt, bed, pre"
python evaluate_trained_model.py "point_transformer" "bed" "pre" >> $FILE_NAME

echo "Evaluating: pt, bed, post"
python evaluate_trained_model.py "point_transformer" "bed" "post" >> $FILE_NAME

############################################################################################

echo "Evaluating: pt, bottle, pre"
python evaluate_trained_model.py "point_transformer" "bottle" "pre" >> $FILE_NAME

echo "Evaluating: pt, bottle, post"
python evaluate_trained_model.py "point_transformer" "bottle" "post" >> $FILE_NAME

############################################################################################

echo "Evaluating: pt, cap, pre"
python evaluate_trained_model.py "point_transformer" "cap" "pre" >> $FILE_NAME

echo "Evaluating: pt, cap, post"
python evaluate_trained_model.py "point_transformer" "cap" "post" >> $FILE_NAME

############################################################################################

echo "Evaluating: pt, car, pre"
python evaluate_trained_model.py "point_transformer" "car" "pre" >> $FILE_NAME

echo "Evaluating: pt, car, post"
python evaluate_trained_model.py "point_transformer" "car" "post" >> $FILE_NAME

############################################################################################

echo "Evaluating: pt, chair, pre"
python evaluate_trained_model.py "point_transformer" "chair" "pre" >> $FILE_NAME

echo "Evaluating: pt, chair, post"
python evaluate_trained_model.py "point_transformer" "chair" "post" >> $FILE_NAME

############################################################################################

echo "Evaluating: pt, guitar, pre"
python evaluate_trained_model.py "point_transformer" "guitar" "pre" >> $FILE_NAME

echo "Evaluating: pt, guitar, post"
python evaluate_trained_model.py "point_transformer" "guitar" "post" >> $FILE_NAME

############################################################################################

echo "Evaluating: pt, helmet, pre"
python evaluate_trained_model.py "point_transformer" "helmet" "pre" >> $FILE_NAME

echo "Evaluating: pt, helmet, post"
python evaluate_trained_model.py "point_transformer" "helmet" "post" >> $FILE_NAME

############################################################################################

echo "Evaluating: pt, knife, pre"
python evaluate_trained_model.py "point_transformer" "knife" "pre" >> $FILE_NAME

echo "Evaluating: pt, knife, post"
python evaluate_trained_model.py "point_transformer" "knife" "post" >> $FILE_NAME

############################################################################################

echo "Evaluating: pt, laptop, pre"
python evaluate_trained_model.py "point_transformer" "laptop" "pre" >> $FILE_NAME

echo "Evaluating: pt, laptop, post"
python evaluate_trained_model.py "point_transformer" "laptop" "post" >> $FILE_NAME

############################################################################################

echo "Evaluating: pt, motorcycle, pre"
python evaluate_trained_model.py "point_transformer" "motorcycle" "pre" >> $FILE_NAME

echo "Evaluating: pt, motorcycle, post"
python evaluate_trained_model.py "point_transformer" "motorcycle" "post" >> $FILE_NAME

############################################################################################

echo "Evaluating: pt, mug, pre"
python evaluate_trained_model.py "point_transformer" "mug" "pre" >> $FILE_NAME

echo "Evaluating: pt, mug, post"
python evaluate_trained_model.py "point_transformer" "mug" "post" >> $FILE_NAME

############################################################################################

echo "Evaluating: pt, skateboard, pre"
python evaluate_trained_model.py "point_transformer" "skateboard" "pre" >> $FILE_NAME

echo "Evaluating: pt, skateboard, post"
python evaluate_trained_model.py "point_transformer" "skateboard" "post" >> $FILE_NAME

############################################################################################

echo "Evaluating: pt, table, pre"
python evaluate_trained_model.py "point_transformer" "table" "pre" >> $FILE_NAME

echo "Evaluating: pt, table, post"
python evaluate_trained_model.py "point_transformer" "table" "post" >> $FILE_NAME

############################################################################################

echo "Evaluating: pt, vessel, pre"
python evaluate_trained_model.py "point_transformer" "vessel" "pre" >> $FILE_NAME

echo "Evaluating: pt, vessel, post"
python evaluate_trained_model.py "point_transformer" "vessel" "post" >> $FILE_NAME

