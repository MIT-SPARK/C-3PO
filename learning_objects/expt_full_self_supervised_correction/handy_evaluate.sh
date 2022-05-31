FILE_NAME="./eval/model_eval.txt"

now=$(date +'%c')
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME

echo "$now"
echo "BOTTLE:: POINT TRANSFORMER"
echo "BOTTLE:: POINT TRANSFORMER" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
python evaluate_trained_model.py "point_transformer" "bottle" "pre" "bottle" >> $FILE_NAME
python evaluate_trained_model.py "point_transformer" "bottle" "post" "table" >> $FILE_NAME
python evaluate_trained_model.py "point_transformer" "bottle" "post" "chair" >> $FILE_NAME
python evaluate_trained_model.py "point_transformer" "bottle" "post" "bottle" >> $FILE_NAME
python evaluate_trained_model.py "point_transformer" "bottle" "post" "laptop" >> $FILE_NAME
python evaluate_trained_model.py "point_transformer" "bottle" "post" "skateboard" >> $FILE_NAME

echo "$now"
echo "CHAIR:: POINT TRANSFORMER"
echo "CHAIR:: POINT TRANSFORMER" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
python evaluate_trained_model.py "point_transformer" "chair" "pre" "chair" >> $FILE_NAME
python evaluate_trained_model.py "point_transformer" "chair" "post" "table" >> $FILE_NAME
python evaluate_trained_model.py "point_transformer" "chair" "post" "chair" >> $FILE_NAME
python evaluate_trained_model.py "point_transformer" "chair" "post" "bottle" >> $FILE_NAME
python evaluate_trained_model.py "point_transformer" "chair" "post" "laptop" >> $FILE_NAME
python evaluate_trained_model.py "point_transformer" "chair" "post" "skateboard" >> $FILE_NAME

echo "$now"
echo "LAPTOP:: POINT TRANSFORMER"
echo "LAPTOP:: POINT TRANSFORMER" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
python evaluate_trained_model.py "point_transformer" "laptop" "pre" "laptop" >> $FILE_NAME
python evaluate_trained_model.py "point_transformer" "laptop" "post" "table" >> $FILE_NAME
python evaluate_trained_model.py "point_transformer" "laptop" "post" "chair" >> $FILE_NAME
python evaluate_trained_model.py "point_transformer" "laptop" "post" "bottle" >> $FILE_NAME
python evaluate_trained_model.py "point_transformer" "laptop" "post" "laptop" >> $FILE_NAME
python evaluate_trained_model.py "point_transformer" "laptop" "post" "skateboard" >> $FILE_NAME

echo "$now"
echo "SKATEBOARD:: POINT TRANSFORMER"
echo "SKATEBOARD:: POINT TRANSFORMER" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
python evaluate_trained_model.py "point_transformer" "skateboard" "pre" "skateboard" >> $FILE_NAME
python evaluate_trained_model.py "point_transformer" "skateboard" "post" "table" >> $FILE_NAME
python evaluate_trained_model.py "point_transformer" "skateboard" "post" "chair" >> $FILE_NAME
python evaluate_trained_model.py "point_transformer" "skateboard" "post" "bottle" >> $FILE_NAME
python evaluate_trained_model.py "point_transformer" "skateboard" "post" "laptop" >> $FILE_NAME
python evaluate_trained_model.py "point_transformer" "skateboard" "post" "skateboard" >> $FILE_NAME

echo "$now"
echo "TABLE:: POINT TRANSFORMER"
echo "TABLE:: POINT TRANSFORMER" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
python evaluate_trained_model.py "point_transformer" "table" "pre" "table" >> $FILE_NAME
python evaluate_trained_model.py "point_transformer" "table" "post" "table" >> $FILE_NAME
python evaluate_trained_model.py "point_transformer" "table" "post" "chair" >> $FILE_NAME
python evaluate_trained_model.py "point_transformer" "table" "post" "bottle" >> $FILE_NAME
python evaluate_trained_model.py "point_transformer" "table" "post" "laptop" >> $FILE_NAME
python evaluate_trained_model.py "point_transformer" "table" "post" "skateboard" >> $FILE_NAME



echo "$now"
echo "BOTTLE:: POINTNET"
echo "BOTTLE:: POINTNET" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
python evaluate_trained_model.py "pointnet" "bottle" "pre" "bottle" >> $FILE_NAME
python evaluate_trained_model.py "pointnet" "bottle" "post" "table" >> $FILE_NAME
python evaluate_trained_model.py "pointnet" "bottle" "post" "chair" >> $FILE_NAME
python evaluate_trained_model.py "pointnet" "bottle" "post" "bottle" >> $FILE_NAME
python evaluate_trained_model.py "pointnet" "bottle" "post" "laptop" >> $FILE_NAME
python evaluate_trained_model.py "pointnet" "bottle" "post" "skateboard" >> $FILE_NAME

echo "$now"
echo "CHAIR:: POINTNET"
echo "CHAIR:: POINTNET" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
python evaluate_trained_model.py "pointnet" "chair" "pre" "chair" >> $FILE_NAME
python evaluate_trained_model.py "pointnet" "chair" "post" "table" >> $FILE_NAME
python evaluate_trained_model.py "pointnet" "chair" "post" "chair" >> $FILE_NAME
python evaluate_trained_model.py "pointnet" "chair" "post" "bottle" >> $FILE_NAME
python evaluate_trained_model.py "pointnet" "chair" "post" "laptop" >> $FILE_NAME
python evaluate_trained_model.py "pointnet" "chair" "post" "skateboard" >> $FILE_NAME

echo "$now"
echo "LAPTOP:: POINTNET"
echo "LAPTOP:: POINTNET" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
python evaluate_trained_model.py "pointnet" "laptop" "pre" "laptop" >> $FILE_NAME
python evaluate_trained_model.py "pointnet" "laptop" "post" "table" >> $FILE_NAME
python evaluate_trained_model.py "pointnet" "laptop" "post" "chair" >> $FILE_NAME
python evaluate_trained_model.py "pointnet" "laptop" "post" "bottle" >> $FILE_NAME
python evaluate_trained_model.py "pointnet" "laptop" "post" "laptop" >> $FILE_NAME
python evaluate_trained_model.py "pointnet" "laptop" "post" "skateboard" >> $FILE_NAME

echo "$now"
echo "SKATEBOARD:: POINTNET"
echo "SKATEBOARD:: POINTNET" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
python evaluate_trained_model.py "pointnet" "skateboard" "pre" "skateboard" >> $FILE_NAME
python evaluate_trained_model.py "pointnet" "skateboard" "post" "table" >> $FILE_NAME
python evaluate_trained_model.py "pointnet" "skateboard" "post" "chair" >> $FILE_NAME
python evaluate_trained_model.py "pointnet" "skateboard" "post" "bottle" >> $FILE_NAME
python evaluate_trained_model.py "pointnet" "skateboard" "post" "laptop" >> $FILE_NAME
python evaluate_trained_model.py "pointnet" "skateboard" "post" "skateboard" >> $FILE_NAME

echo "$now"
echo "TABLE:: POINTNET"
echo "TABLE:: POINTNET" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
python evaluate_trained_model.py "pointnet" "table" "pre" "table" >> $FILE_NAME
python evaluate_trained_model.py "pointnet" "table" "post" "table" >> $FILE_NAME
python evaluate_trained_model.py "pointnet" "table" "post" "chair" >> $FILE_NAME
python evaluate_trained_model.py "pointnet" "table" "post" "bottle" >> $FILE_NAME
python evaluate_trained_model.py "pointnet" "table" "post" "laptop" >> $FILE_NAME
python evaluate_trained_model.py "pointnet" "table" "post" "skateboard" >> $FILE_NAME

