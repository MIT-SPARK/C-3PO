FILE_NAME="./eval/handy_train_baseline_output.txt"

echo "pt: 001_chips_can"
python train_baseline.py "point_transformer" "001_chips_can" >> $FILE_NAME
echo "pt: 002"
python train_baseline.py "point_transformer" "002_master_chef_can" >> $FILE_NAME
echo "pt: 003"
python train_baseline.py "point_transformer" "003_cracker_box" >> $FILE_NAME
echo "pt: 004"
python train_baseline.py "point_transformer" "004_sugar_box" >> $FILE_NAME
echo "pt: 005"
python train_baseline.py "point_transformer" "005_tomato_soup_can" >> $FILE_NAME
echo "pt: 006"
python train_baseline.py "point_transformer" "006_mustard_bottle" >> $FILE_NAME
echo "pt: 007"
python train_baseline.py "point_transformer" "007_tuna_fish_can" >> $FILE_NAME
echo "pt: 008"
python train_baseline.py "point_transformer" "008_pudding_box" >> $FILE_NAME
echo "pt: 009"
python train_baseline.py "point_transformer" "009_gelatin_box" >> $FILE_NAME
echo "pt: 010"
python train_baseline.py "point_transformer" "010_potted_meat_can" >> $FILE_NAME
echo "pt: 011"
python train_baseline.py "point_transformer" "011_banana" >> $FILE_NAME
echo "pt: 019"
python train_baseline.py "point_transformer" "019_pitcher_base" >> $FILE_NAME
echo "pt: 021_bleach_cleanser"
python train_baseline.py "point_transformer" "021_bleach_cleanser" >> $FILE_NAME
echo "pt: 035_power_drill"
python train_baseline.py "point_transformer" "035_power_drill" >> $FILE_NAME
echo "pt: 036_wood_block"
python train_baseline.py "point_transformer" "036_wood_block" >> $FILE_NAME
echo "pt: 037_scissors"
python train_baseline.py "point_transformer" "037_scissors" >> $FILE_NAME
echo "pt: 051_large_clamp"
python train_baseline.py "point_transformer" "051_large_clamp" >> $FILE_NAME
echo "pt: 052_extra_large_clamp"
python train_baseline.py "point_transformer" "052_extra_large_clamp" >> $FILE_NAME
echo "pt: 061_foam_brick"
python train_baseline.py "point_transformer" "061_foam_brick" >> $FILE_NAME


