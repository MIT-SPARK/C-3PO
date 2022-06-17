BASELINE_SUPERVISED_FILE="./eval/baseline_supervised_eval.txt"

now=$(date)
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $BASELINE_SUPERVISED_FILE
echo "$now" >> $BASELINE_SUPERVISED_FILE
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $BASELINE_SUPERVISED_FILE

echo "POINT_TRANSFORMER" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, 001_chips_can"
python evaluate_baseline.py "point_transformer" "001_chips_can" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, 002_master_chef_can"
python evaluate_baseline.py "point_transformer" "002_master_chef_can" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, 003_cracker_box"
python evaluate_baseline.py "point_transformer" "003_cracker_box" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, 004_sugar_box"
python evaluate_baseline.py "point_transformer" "004_sugar_box" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, 005_tomato_soup_can"
python evaluate_baseline.py "point_transformer" "005_tomato_soup_can" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, 006_mustard_bottle"
python evaluate_baseline.py "point_transformer" "006_mustard_bottle" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, 007_tuna_fish_can"
python evaluate_baseline.py "point_transformer" "007_tuna_fish_can" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, 008_pudding_box"
python evaluate_baseline.py "point_transformer" "008_pudding_box" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, 009_gelatin_box"
python evaluate_baseline.py "point_transformer" "009_gelatin_box" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, 010_potted_meat_can"
python evaluate_baseline.py "point_transformer" "010_potted_meat_can" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, 011_banana"
python evaluate_baseline.py "point_transformer" "011_banana" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, 019_pitcher_base"
python evaluate_baseline.py "point_transformer" "019_pitcher_base" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, 021_bleach_cleanser"
python evaluate_baseline.py "point_transformer" "021_bleach_cleanser" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, 035_power_drill"
python evaluate_baseline.py "point_transformer" "035_power_drill" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, 036_wood_block"
python evaluate_baseline.py "point_transformer" "036_wood_block" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, 037_scissors"
python evaluate_baseline.py "point_transformer" "037_scissors" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, 051_large_clamp"
python evaluate_baseline.py "point_transformer" "051_large_clamp" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, 052_extra_large_clamp"
python evaluate_baseline.py "point_transformer" "052_extra_large_clamp" >> $BASELINE_SUPERVISED_FILE

echo "EVALUATING BASELINE: POINT_TRANSFORMER, 061_foam_brick"
python evaluate_baseline.py "point_transformer" "061_foam_brick" >> $BASELINE_SUPERVISED_FILE
