SIM_TRAINED_FILE="./eval/sim_trained_model_eval_main.txt"

now=$(date +'%c')
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $SIM_TRAINED_FILE
echo "$now" >> $SIM_TRAINED_FILE
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $SIM_TRAINED_FILE

echo "supervised evaluation: POINTNET, AIRPLANE"
python evaluate_proposed_model.py "pointnet" "airplane" "pre" >> $SIM_TRAINED_FILE

echo "supervised evaluation: POINTNET, BATHTUB"
python evaluate_proposed_model.py "pointnet" "bathtub" "pre" >> $SIM_TRAINED_FILE

echo "supervised evaluation: POINTNET, BED"
python evaluate_proposed_model.py "pointnet" "bed" "pre" >> $SIM_TRAINED_FILE

echo "supervised evaluation: POINTNET, BOTTLE"
python evaluate_proposed_model.py "pointnet" "bottle" "pre" >> $SIM_TRAINED_FILE
#
#echo "supervised evaluation: POINTNET, CAP"
#python evaluate_proposed_model.py "pointnet" "cap" "pre" >> $SIM_TRAINED_FILE
#
#echo "supervised evaluation: POINTNET, CAR"
#python evaluate_proposed_model.py "pointnet" "car" "pre" >> $SIM_TRAINED_FILE
#
#echo "supervised evaluation: POINTNET, CHAIR"
#python evaluate_proposed_model.py "pointnet" "chair" "pre" >> $SIM_TRAINED_FILE
#
#echo "supervised evaluation: POINTNET, GUITAR"
#python evaluate_proposed_model.py "pointnet" "guitar" "pre" >> $SIM_TRAINED_FILE
#
#echo "supervised evaluation: POINTNET, HELMET"
#python evaluate_proposed_model.py "pointnet" "helmet" "pre" >> $SIM_TRAINED_FILE
#
#echo "supervised evaluation: POINTNET, KNIFE"
#python evaluate_proposed_model.py "pointnet" "knife" "pre" >> $SIM_TRAINED_FILE
#
#echo "supervised evaluation: POINTNET, LAPTOP"
#python evaluate_proposed_model.py "pointnet" "laptop" "pre" >> $SIM_TRAINED_FILE
#
#echo "supervised evaluation: POINTNET, MOTORCYCLE"
#python evaluate_proposed_model.py "pointnet" "motorcycle" "pre" >> $SIM_TRAINED_FILE
#
#echo "supervised evaluation: POINTNET, MUG"
#python evaluate_proposed_model.py "pointnet" "mug" "pre" >> $SIM_TRAINED_FILE
#
#echo "supervised evaluation: POINTNET, SKATEBOARD"
#python evaluate_proposed_model.py "pointnet" "skateboard" "pre" >> $SIM_TRAINED_FILE
#
#echo "supervised evaluation: POINTNET, TABLE"
#python evaluate_proposed_model.py "pointnet" "table" "pre" >> $SIM_TRAINED_FILE
#
#echo "supervised evaluation: POINTNET, VESSEL"
#python evaluate_proposed_model.py "pointnet" "vessel" "pre" >> $SIM_TRAINED_FILE



echo "supervised evaluation: POINT_TRANSFORMER, AIRPLANE"
python evaluate_proposed_model.py "point_transformer" "airplane" "pre" >> $SIM_TRAINED_FILE

echo "supervised evaluation: POINT_TRANSFORMER, BATHTUB"
python evaluate_proposed_model.py "point_transformer" "bathtub" "pre" >> $SIM_TRAINED_FILE

echo "supervised evaluation: POINT_TRANSFORMER, BED"
python evaluate_proposed_model.py "point_transformer" "bed" "pre" >> $SIM_TRAINED_FILE

echo "supervised evaluation: POINT_TRANSFORMER, BOTTLE"
python evaluate_proposed_model.py "point_transformer" "bottle" "pre" >> $SIM_TRAINED_FILE

echo "supervised evaluation: POINT_TRANSFORMER, CAP"
python evaluate_proposed_model.py "point_transformer" "cap" "pre" >> $SIM_TRAINED_FILE

echo "supervised evaluation: POINT_TRANSFORMER, CAR"
python evaluate_proposed_model.py "point_transformer" "car" "pre" >> $SIM_TRAINED_FILE

echo "supervised evaluation: POINT_TRANSFORMER, CHAIR"
python evaluate_proposed_model.py "point_transformer" "chair" "pre" >> $SIM_TRAINED_FILE

echo "supervised evaluation: POINT_TRANSFORMER, GUITAR"
python evaluate_proposed_model.py "point_transformer" "guitar" "pre" >> $SIM_TRAINED_FILE

echo "supervised evaluation: POINT_TRANSFORMER, HELMET"
python evaluate_proposed_model.py "point_transformer" "helmet" "pre" >> $SIM_TRAINED_FILE

echo "supervised evaluation: POINT_TRANSFORMER, KNIFE"
python evaluate_proposed_model.py "point_transformer" "knife" "pre" >> $SIM_TRAINED_FILE

echo "supervised evaluation: POINT_TRANSFORMER, LAPTOP"
python evaluate_proposed_model.py "point_transformer" "laptop" "pre" >> $SIM_TRAINED_FILE

echo "supervised evaluation: POINT_TRANSFORMER, MOTORCYCLE"
python evaluate_proposed_model.py "point_transformer" "motorcycle" "pre" >> $SIM_TRAINED_FILE

echo "supervised evaluation: POINT_TRANSFORMER, MUG"
python evaluate_proposed_model.py "point_transformer" "mug" "pre" >> $SIM_TRAINED_FILE

echo "supervised evaluation: POINT_TRANSFORMER, SKATEBOARD"
python evaluate_proposed_model.py "point_transformer" "skateboard" "pre" >> $SIM_TRAINED_FILE

echo "supervised evaluation: POINT_TRANSFORMER, TABLE"
python evaluate_proposed_model.py "point_transformer" "table" "pre" >> $SIM_TRAINED_FILE

echo "supervised evaluation: POINT_TRANSFORMER, VESSEL"
python evaluate_proposed_model.py "point_transformer" "vessel" "pre" >> $SIM_TRAINED_FILE
