FILE_NAME="./eval/model_eval_main_temp.txt"

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
python evaluate_model.py "point_transformer" "bottle" "pre" "bottle" "shapenet" >> $FILE_NAME
python evaluate_model.py "point_transformer" "bottle" "post" "table" "shapenet" >> $FILE_NAME
python evaluate_model.py "point_transformer" "bottle" "post" "chair" "shapenet" >> $FILE_NAME
python evaluate_model.py "point_transformer" "bottle" "post" "bottle" "shapenet" >> $FILE_NAME
python evaluate_model.py "point_transformer" "bottle" "post" "laptop" "shapenet" >> $FILE_NAME
python evaluate_model.py "point_transformer" "bottle" "post" "skateboard" "shapenet" >> $FILE_NAME

echo "$now"
echo "CHAIR:: POINT TRANSFORMER"
echo "CHAIR:: POINT TRANSFORMER" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
python evaluate_model.py "point_transformer" "chair" "pre" "chair" "shapenet" >> $FILE_NAME
python evaluate_model.py "point_transformer" "chair" "post" "table" "shapenet" >> $FILE_NAME
python evaluate_model.py "point_transformer" "chair" "post" "chair" "shapenet" >> $FILE_NAME
python evaluate_model.py "point_transformer" "chair" "post" "bottle" "shapenet" >> $FILE_NAME
python evaluate_model.py "point_transformer" "chair" "post" "laptop" "shapenet" >> $FILE_NAME
python evaluate_model.py "point_transformer" "chair" "post" "skateboard" "shapenet" >> $FILE_NAME

echo "$now"
echo "LAPTOP:: POINT TRANSFORMER"
echo "LAPTOP:: POINT TRANSFORMER" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
python evaluate_model.py "point_transformer" "laptop" "pre" "laptop" "shapenet" >> $FILE_NAME
python evaluate_model.py "point_transformer" "laptop" "post" "table" "shapenet" >> $FILE_NAME
python evaluate_model.py "point_transformer" "laptop" "post" "chair" "shapenet" >> $FILE_NAME
python evaluate_model.py "point_transformer" "laptop" "post" "bottle" "shapenet" >> $FILE_NAME
python evaluate_model.py "point_transformer" "laptop" "post" "laptop" "shapenet" >> $FILE_NAME
python evaluate_model.py "point_transformer" "laptop" "post" "skateboard" "shapenet" >> $FILE_NAME

echo "$now"
echo "SKATEBOARD:: POINT TRANSFORMER"
echo "SKATEBOARD:: POINT TRANSFORMER" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
python evaluate_model.py "point_transformer" "skateboard" "pre" "skateboard" "shapenet" >> $FILE_NAME
python evaluate_model.py "point_transformer" "skateboard" "post" "table" "shapenet" >> $FILE_NAME
python evaluate_model.py "point_transformer" "skateboard" "post" "chair" "shapenet" >> $FILE_NAME
python evaluate_model.py "point_transformer" "skateboard" "post" "bottle" "shapenet" >> $FILE_NAME
python evaluate_model.py "point_transformer" "skateboard" "post" "laptop" "shapenet" >> $FILE_NAME
python evaluate_model.py "point_transformer" "skateboard" "post" "skateboard" "shapenet" >> $FILE_NAME

echo "$now"
echo "TABLE:: POINT TRANSFORMER"
echo "TABLE:: POINT TRANSFORMER" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
python evaluate_model.py "point_transformer" "table" "pre" "table" "shapenet" >> $FILE_NAME
python evaluate_model.py "point_transformer" "table" "post" "table" "shapenet" >> $FILE_NAME
python evaluate_model.py "point_transformer" "table" "post" "chair" "shapenet" >> $FILE_NAME
python evaluate_model.py "point_transformer" "table" "post" "bottle" "shapenet" >> $FILE_NAME
python evaluate_model.py "point_transformer" "table" "post" "laptop" "shapenet" >> $FILE_NAME
python evaluate_model.py "point_transformer" "table" "post" "skateboard" "shapenet" >> $FILE_NAME



echo "$now"
echo "BOTTLE:: POINTNET"
echo "BOTTLE:: POINTNET" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
python evaluate_model.py "pointnet" "bottle" "pre" "bottle" "shapenet" >> $FILE_NAME
python evaluate_model.py "pointnet" "bottle" "post" "table" "shapenet" >> $FILE_NAME
python evaluate_model.py "pointnet" "bottle" "post" "chair" "shapenet" >> $FILE_NAME
python evaluate_model.py "pointnet" "bottle" "post" "bottle" "shapenet" >> $FILE_NAME
python evaluate_model.py "pointnet" "bottle" "post" "laptop" "shapenet" >> $FILE_NAME
python evaluate_model.py "pointnet" "bottle" "post" "skateboard" "shapenet" >> $FILE_NAME

echo "$now"
echo "CHAIR:: POINTNET"
echo "CHAIR:: POINTNET" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
python evaluate_model.py "pointnet" "chair" "pre" "chair" "shapenet" >> $FILE_NAME
python evaluate_model.py "pointnet" "chair" "post" "table" "shapenet" >> $FILE_NAME
python evaluate_model.py "pointnet" "chair" "post" "chair" "shapenet" >> $FILE_NAME
python evaluate_model.py "pointnet" "chair" "post" "bottle" "shapenet" >> $FILE_NAME
python evaluate_model.py "pointnet" "chair" "post" "laptop" "shapenet" >> $FILE_NAME
python evaluate_model.py "pointnet" "chair" "post" "skateboard" "shapenet" >> $FILE_NAME

echo "$now"
echo "LAPTOP:: POINTNET"
echo "LAPTOP:: POINTNET" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
python evaluate_model.py "pointnet" "laptop" "pre" "laptop" "shapenet" >> $FILE_NAME
python evaluate_model.py "pointnet" "laptop" "post" "table" "shapenet" >> $FILE_NAME
python evaluate_model.py "pointnet" "laptop" "post" "chair" "shapenet" >> $FILE_NAME
python evaluate_model.py "pointnet" "laptop" "post" "bottle" "shapenet" >> $FILE_NAME
python evaluate_model.py "pointnet" "laptop" "post" "laptop" "shapenet" >> $FILE_NAME
python evaluate_model.py "pointnet" "laptop" "post" "skateboard" "shapenet" >> $FILE_NAME

echo "$now"
echo "SKATEBOARD:: POINTNET"
echo "SKATEBOARD:: POINTNET" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
python evaluate_model.py "pointnet" "skateboard" "pre" "skateboard" "shapenet" >> $FILE_NAME
python evaluate_model.py "pointnet" "skateboard" "post" "table" "shapenet" >> $FILE_NAME
python evaluate_model.py "pointnet" "skateboard" "post" "chair" "shapenet" >> $FILE_NAME
python evaluate_model.py "pointnet" "skateboard" "post" "bottle" "shapenet" >> $FILE_NAME
python evaluate_model.py "pointnet" "skateboard" "post" "laptop" "shapenet" >> $FILE_NAME
python evaluate_model.py "pointnet" "skateboard" "post" "skateboard" "shapenet" >> $FILE_NAME

echo "$now"
echo "TABLE:: POINTNET"
echo "TABLE:: POINTNET" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
python evaluate_model.py "pointnet" "table" "pre" "table" "shapenet" >> $FILE_NAME
python evaluate_model.py "pointnet" "table" "post" "table" "shapenet" >> $FILE_NAME
python evaluate_model.py "pointnet" "table" "post" "chair" "shapenet" >> $FILE_NAME
python evaluate_model.py "pointnet" "table" "post" "bottle" "shapenet" >> $FILE_NAME
python evaluate_model.py "pointnet" "table" "post" "laptop" "shapenet" >> $FILE_NAME
python evaluate_model.py "pointnet" "table" "post" "skateboard" "shapenet" >> $FILE_NAME

