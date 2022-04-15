FILE_NAME="./eval/model_eval_across.txt"

now=$(date +'%c')
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME

##########################################################################################
echo "EVALUATING POINT TRANSFORMER" >> $FILE_NAME
###########################################################################################

echo "EVALUATING PT(laptop) ON DATASET(laptop)"
echo "EVALUATING PT(laptop) ON DATASET(laptop)" >> $FILE_NAME
python evaluate_trained_model_across.py "point_transformer" laptop "pre" laptop >> $FILE_NAME
python evaluate_trained_model_across.py "point_transformer" laptop "post" laptop >> $FILE_NAME

echo "EVALUATING PT(laptop) ON DATASET(skateboard)"
echo "EVALUATING PT(laptop) ON DATASET(skateboard)" >> $FILE_NAME
python evaluate_trained_model_across.py "point_transformer" laptop "pre" skateboard >> $FILE_NAME
python evaluate_trained_model_across.py "point_transformer" laptop "post" skateboard >> $FILE_NAME


for model_type in "skateboard"
#for model_type in "table"
do
  for data_type in "table" "chair" "bottle" "laptop" "skateboard"
  do
    echo "EVALUATING PT($model_type) ON DATASET($data_type)"
    echo "EVALUATING PT($model_type) ON DATASET($data_type)" >> $FILE_NAME
    python evaluate_trained_model_across.py "point_transformer" $model_type "pre" $data_type >> $FILE_NAME
    python evaluate_trained_model_across.py "point_transformer" $model_type "post" $data_type >> $FILE_NAME
#    if [ $model_type != $data_type ]
#    then
#      echo "EVALUATING PT($model_type) ON DATASET($data_type)"
#      echo "EVALUATING PT($model_type) ON DATASET($data_type)" >> $FILE_NAME
#      python evaluate_trained_model_across.py "point_transformer" $model_type "post" $data_type >> $FILE_NAME
#    fi
  done
done


##########################################################################################
echo "EVALUATING POINTNET" >> $FILE_NAME
###########################################################################################

for model_type in "table" "chair" "bottle" "laptop" "skateboard"
#for model_type in "table"
do
  for data_type in "table" "chair" "bottle" "laptop" "skateboard"
  do
    echo "EVALUATING POINTNET($model_type) ON DATASET($data_type)"
    echo "EVALUATING POINTNET($model_type) ON DATASET($data_type)" >> $FILE_NAME
    python evaluate_trained_model_across.py "pointnet" $model_type "pre" $data_type >> $FILE_NAME
    python evaluate_trained_model_across.py "pointnet" $model_type "post" $data_type >> $FILE_NAME
#    if [ $model_type != $data_type ]
#    then
#      echo "EVALUATING POINTNET($model_type) ON DATASET($data_type)"
#      echo "EVALUATING POINTNET($model_type) ON DATASET($data_type)" >> $FILE_NAME
#      python evaluate_trained_model_across.py "pointnet" $model_type "post" $data_type >> $FILE_NAME
#    fi
  done
done
