BASELINE_SUPERVISED_FILE="./eval/sim_trained_model_eval.txt"

echo "EVALUATING BASELINE: POINTNET, AIRPLANE"
python evaluate_sim_supervised_model.py "pointnet" "airplane" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINTNET, BATHTUB"
python evaluate_sim_supervised_model.py "pointnet" "bathtub" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINTNET, BED"
python evaluate_sim_supervised_model.py "pointnet" "bed" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINTNET, BOTTLE"
python evaluate_sim_supervised_model.py "pointnet" "bottle" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINTNET, CAP"
python evaluate_sim_supervised_model.py "pointnet" "cap" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINTNET, CAR"
python evaluate_sim_supervised_model.py "pointnet" "car" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINTNET, CHAIR"
python evaluate_sim_supervised_model.py "pointnet" "chair" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINTNET, GUITAR"
python evaluate_sim_supervised_model.py "pointnet" "guitar" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINTNET, HELMET"
python evaluate_sim_supervised_model.py "pointnet" "helmet" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINTNET, KNIFE"
python evaluate_sim_supervised_model.py "pointnet" "knife" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINTNET, LAPTOP"
python evaluate_sim_supervised_model.py "pointnet" "laptop" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINTNET, MOTORCYCLE"
python evaluate_sim_supervised_model.py "pointnet" "motorcycle" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINTNET, MUG"
python evaluate_sim_supervised_model.py "pointnet" "mug" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINTNET, SKATEBOARD"
python evaluate_sim_supervised_model.py "pointnet" "skateboard" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINTNET, TABLE"
python evaluate_sim_supervised_model.py "pointnet" "table" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINTNET, VESSEL"
python evaluate_sim_supervised_model.py "pointnet" "vessel" >> $BASELINE_SUPERVISED_FILE



echo "EVALUATING BASELINE: POINT_TRANSFORMER, AIRPLANE"
python evaluate_sim_supervised_model.py "point_transformer" "airplane" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, BATHTUB"
python evaluate_sim_supervised_model.py "point_transformer" "bathtub" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, BED"
python evaluate_sim_supervised_model.py "point_transformer" "bed" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, BOTTLE"
python evaluate_sim_supervised_model.py "point_transformer" "bottle" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, CAP"
python evaluate_sim_supervised_model.py "point_transformer" "cap" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, CAR"
python evaluate_sim_supervised_model.py "point_transformer" "car" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, CHAIR"
python evaluate_sim_supervised_model.py "point_transformer" "chair" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, GUITAR"
python evaluate_sim_supervised_model.py "point_transformer" "guitar" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, HELMET"
python evaluate_sim_supervised_model.py "point_transformer" "helmet" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, KNIFE"
python evaluate_sim_supervised_model.py "point_transformer" "knife" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, LAPTOP"
python evaluate_sim_supervised_model.py "point_transformer" "laptop" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, MOTORCYCLE"
python evaluate_sim_supervised_model.py "point_transformer" "motorcycle" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, MUG"
python evaluate_sim_supervised_model.py "point_transformer" "mug" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, SKATEBOARD"
python evaluate_sim_supervised_model.py "point_transformer" "skateboard" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, TABLE"
python evaluate_sim_supervised_model.py "point_transformer" "table" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, VESSEL"
python evaluate_sim_supervised_model.py "point_transformer" "vessel" >> $BASELINE_SUPERVISED_FILE
