FILE_NAME="./eval/model_eval_across.txt"

now=$(date +'%c')
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME

##########################################################################################
echo "EVALUATING POINT TRANSFORMER" >> $FILE_NAME
###########################################################################################

for model_type in "table" "chair" "bottle" "laptop" "skateboard"
do
  for data_type in "table" "chair" "bottle" "laptop" "skateboard"
  do
    if [ $model_type != $data_type ]
    then
      echo "EVALUATING PT($model_type) ON DATASET($data_type)"
      echo "EVALUATING PT($model_type) ON DATASET($data_type)" >> $FILE_NAME
      python evaluate_trained_model_across.py "point_transformer" $model_type "post" $data_type >> $FILE_NAME
    fi
  done
done


##########################################################################################
echo "EVALUATING POINTNET" >> $FILE_NAME
###########################################################################################

for model_type in "table" "chair" "bottle" "laptop" "skateboard"
do
  for data_type in "table" "chair" "bottle" "laptop" "skateboard"
  do
    if [ $model_type != $data_type ]
    then
      echo "EVALUATING POINTNET($model_type) ON DATASET($data_type)"
      echo "EVALUATING POINTNET($model_type) ON DATASET($data_type)" >> $FILE_NAME
      python evaluate_trained_model_across.py "pointnet" $model_type "post" $data_type >> $FILE_NAME
    fi
  done
done
