FILE_NAME="./eval/model_eval_ycb.txt"

now=$(date +'%c')
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
#
#echo "$now"
#echo "002_master_chef_can:: POINT TRANSFORMER"
#echo "002_master_chef_can:: POINT TRANSFORMER" >> $FILE_NAME
#echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
#echo "$now" >> $FILE_NAME
#echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
#python evaluate_model.py "point_transformer" "002_master_chef_can" "pre" "002_master_chef_can" "ycb" >> $FILE_NAME
#python evaluate_model.py "point_transformer" "002_master_chef_can" "post" "002_master_chef_can" "ycb" >> $FILE_NAME
#python evaluate_model.py "point_transformer" "002_master_chef_can" "post" "006_mustard_bottle" "ycb" >> $FILE_NAME
#python evaluate_model.py "point_transformer" "002_master_chef_can" "post" "011_banana" "ycb" >> $FILE_NAME
#python evaluate_model.py "point_transformer" "002_master_chef_can" "post" "037_scissors" "ycb" >> $FILE_NAME
#python evaluate_model.py "point_transformer" "002_master_chef_can" "post" "052_extra_large_clamp" "ycb" >> $FILE_NAME

echo "$now"
echo "006_mustard_bottle:: POINT TRANSFORMER"
echo "006_mustard_bottle:: POINT TRANSFORMER" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
python evaluate_model.py "point_transformer" "006_mustard_bottle" "pre" "006_mustard_bottle" "ycb" >> $FILE_NAME
python evaluate_model.py "point_transformer" "006_mustard_bottle" "post" "002_master_chef_can" "ycb" >> $FILE_NAME
python evaluate_model.py "point_transformer" "006_mustard_bottle" "post" "006_mustard_bottle" "ycb" >> $FILE_NAME
python evaluate_model.py "point_transformer" "006_mustard_bottle" "post" "011_banana" "ycb" >> $FILE_NAME
python evaluate_model.py "point_transformer" "006_mustard_bottle" "post" "037_scissors" "ycb" >> $FILE_NAME
python evaluate_model.py "point_transformer" "006_mustard_bottle" "post" "052_extra_large_clamp" "ycb" >> $FILE_NAME

echo "$now"
echo "011_banana:: POINT TRANSFORMER"
echo "011_banana:: POINT TRANSFORMER" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
python evaluate_model.py "point_transformer" "011_banana" "pre" "011_banana" "ycb" >> $FILE_NAME
python evaluate_model.py "point_transformer" "011_banana" "post" "002_master_chef_can" "ycb" >> $FILE_NAME
python evaluate_model.py "point_transformer" "011_banana" "post" "006_mustard_bottle" "ycb" >> $FILE_NAME
python evaluate_model.py "point_transformer" "011_banana" "post" "011_banana" "ycb" >> $FILE_NAME
python evaluate_model.py "point_transformer" "011_banana" "post" "037_scissors" "ycb" >> $FILE_NAME
python evaluate_model.py "point_transformer" "011_banana" "post" "052_extra_large_clamp" "ycb" >> $FILE_NAME

echo "$now"
echo "037_scissors:: POINT TRANSFORMER"
echo "037_scissors:: POINT TRANSFORMER" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
python evaluate_model.py "point_transformer" "037_scissors" "pre" "037_scissors" "ycb" >> $FILE_NAME
python evaluate_model.py "point_transformer" "037_scissors" "post" "002_master_chef_can" "ycb" >> $FILE_NAME
python evaluate_model.py "point_transformer" "037_scissors" "post" "006_mustard_bottle" "ycb" >> $FILE_NAME
python evaluate_model.py "point_transformer" "037_scissors" "post" "011_banana" "ycb" >> $FILE_NAME
python evaluate_model.py "point_transformer" "037_scissors" "post" "037_scissors" "ycb" >> $FILE_NAME
python evaluate_model.py "point_transformer" "037_scissors" "post" "052_extra_large_clamp" "ycb" >> $FILE_NAME

echo "$now"
echo "052_extra_large_clamp:: POINT TRANSFORMER"
echo "052_extra_large_clamp:: POINT TRANSFORMER" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
python evaluate_model.py "point_transformer" "052_extra_large_clamp" "pre" "052_extra_large_clamp" "ycb" >> $FILE_NAME
python evaluate_model.py "point_transformer" "052_extra_large_clamp" "post" "002_master_chef_can" "ycb" >> $FILE_NAME
python evaluate_model.py "point_transformer" "052_extra_large_clamp" "post" "006_mustard_bottle" "ycb" >> $FILE_NAME
python evaluate_model.py "point_transformer" "052_extra_large_clamp" "post" "011_banana" "ycb" >> $FILE_NAME
python evaluate_model.py "point_transformer" "052_extra_large_clamp" "post" "037_scissors" "ycb" >> $FILE_NAME
python evaluate_model.py "point_transformer" "052_extra_large_clamp" "post" "052_extra_large_clamp" "ycb" >> $FILE_NAME
