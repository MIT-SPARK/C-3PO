cd ../../c3po/expt_ycb

FILE_NAME="./eval/handy_train_baseline_output.txt"

now=$(date)
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>PT>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME


echo "pt: 001_chips_can"
python training.py "point_transformer" "001_chips_can" "baseline" >> $FILE_NAME
echo "pt: 002_master_chef_can"
python training.py "point_transformer" "002_master_chef_can" "baseline" >> $FILE_NAME
echo "pt: 003_cracker_box"
python training.py "point_transformer" "003_cracker_box" "baseline" >> $FILE_NAME
echo "pt: 004_sugar_box"
python training.py "point_transformer" "004_sugar_box" "baseline" >> $FILE_NAME
echo "pt: 005_tomato_soup_can"
python training.py "point_transformer" "005_tomato_soup_can" "baseline" >> $FILE_NAME
echo "pt: 006_mustard_bottle"
python training.py "point_transformer" "006_mustard_bottle" "baseline" >> $FILE_NAME
echo "pt: 007_tuna_fish_can"
python training.py "point_transformer" "007_tuna_fish_can" "baseline" >> $FILE_NAME
echo "pt: 008_pudding_box"
python training.py "point_transformer" "008_pudding_box" "baseline" >> $FILE_NAME
echo "pt: 009_gelatin_box"
python training.py "point_transformer" "009_gelatin_box" "baseline" >> $FILE_NAME
echo "pt: 010_potted_meat_can"
python training.py "point_transformer" "010_potted_meat_can" "baseline" >> $FILE_NAME
echo "pt: 011_banana"
python training.py "point_transformer" "011_banana" "baseline" >> $FILE_NAME
echo "pt: 019_pitcher_base"
python training.py "point_transformer" "019_pitcher_base" "baseline" >> $FILE_NAME
echo "pt: 021_bleach_cleanser"
python training.py "point_transformer" "021_bleach_cleanser" "baseline" >> $FILE_NAME
echo "pt: 035_power_drill"
python training.py "point_transformer" "035_power_drill" "baseline" >> $FILE_NAME
echo "pt: 036_wood_block"
python training.py "point_transformer" "036_wood_block" "baseline" >> $FILE_NAME
echo "pt: 037_scissors"
python training.py "point_transformer" "037_scissors" "baseline" >> $FILE_NAME
echo "pt: 051_large_clamp"
python training.py "point_transformer" "051_large_clamp" "baseline" >> $FILE_NAME
echo "pt: 052_extra_large_clamp"
python training.py "point_transformer" "052_extra_large_clamp" "baseline" >> $FILE_NAME
echo "pt: 061_foam_brick"
python training.py "point_transformer" "061_foam_brick" "baseline" >> $FILE_NAME


