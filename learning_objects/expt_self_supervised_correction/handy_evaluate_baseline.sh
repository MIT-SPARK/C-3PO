echo "EVALUATING BASELINE: POINTNET, AIRPLANE"
python evaluate_baseline.py "pointnet" "airplane" >> supervised_baseline_eval.txt

echo "EVALUATING BASELINE: POINTNET, BATHTUB"
python evaluate_baseline.py "pointnet" "bathtub" >> supervised_baseline_eval.txt

echo "EVALUATING BASELINE: POINTNET, BED"
python evaluate_baseline.py "pointnet" "bed" >> supervised_baseline_eval.txt

echo "EVALUATING BASELINE: POINTNET, BOTTLE"
python evaluate_baseline.py "pointnet" "bottle" >> supervised_baseline_eval.txt

echo "EVALUATING BASELINE: POINTNET, CAP"
python evaluate_baseline.py "pointnet" "cap" >> supervised_baseline_eval.txt

echo "EVALUATING BASELINE: POINTNET, CAR"
python evaluate_baseline.py "pointnet" "car" >> supervised_baseline_eval.txt

echo "EVALUATING BASELINE: POINTNET, CHAIR"
python evaluate_baseline.py "pointnet" "chair" >> supervised_baseline_eval.txt

echo "EVALUATING BASELINE: POINTNET, GUITAR"
python evaluate_baseline.py "pointnet" "guitar" >> supervised_baseline_eval.txt

echo "EVALUATING BASELINE: POINTNET, HELMET"
python evaluate_baseline.py "pointnet" "helmet" >> supervised_baseline_eval.txt

echo "EVALUATING BASELINE: POINTNET, KNIFE"
python evaluate_baseline.py "pointnet" "knife" >> supervised_baseline_eval.txt

echo "EVALUATING BASELINE: POINTNET, LAPTOP"
python evaluate_baseline.py "pointnet" "laptop" >> supervised_baseline_eval.txt

echo "EVALUATING BASELINE: POINTNET, MOTORCYCLE"
python evaluate_baseline.py "pointnet" "motorcycle" >> supervised_baseline_eval.txt

echo "EVALUATING BASELINE: POINTNET, MUG"
python evaluate_baseline.py "pointnet" "mug" >> supervised_baseline_eval.txt

echo "EVALUATING BASELINE: POINTNET, SKATEBOARD"
python evaluate_baseline.py "pointnet" "skateboard" >> supervised_baseline_eval.txt

echo "EVALUATING BASELINE: POINTNET, TABLE"
python evaluate_baseline.py "pointnet" "table" >> supervised_baseline_eval.txt

echo "EVALUATING BASELINE: POINTNET, VESSEL"
python evaluate_baseline.py "pointnet" "vessel" >> supervised_baseline_eval.txt
