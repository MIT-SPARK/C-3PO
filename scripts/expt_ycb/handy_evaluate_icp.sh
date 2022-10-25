cd ../../c3po/expt_ycb

FILE_NAME="./eval/eval_icp.txt"

now=$(date +'%c')
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "RANSAC+ICP: no corrector"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME

echo "001_chips_can"
python evaluate_icp.py "001_chips_can" "ransac" "nc" >> $FILE_NAME
echo "002_master_chef_can"
python evaluate_icp.py "002_master_chef_can" "ransac" "nc" >> $FILE_NAME
echo "003_cracker_box"
python evaluate_icp.py "003_cracker_box" "ransac" "nc" >> $FILE_NAME
echo "004_sugar_box"
python evaluate_icp.py "004_sugar_box" "ransac" "nc" >> $FILE_NAME
echo "005_tomato_soup_can"
python evaluate_icp.py "005_tomato_soup_can" "ransac" "nc" >> $FILE_NAME
echo "006_mustard_bottle"
python evaluate_icp.py "006_mustard_bottle" "ransac" "nc" >> $FILE_NAME
echo "007_tuna_fish_can"
python evaluate_icp.py "007_tuna_fish_can" "ransac" "nc" >> $FILE_NAME
echo "008_pudding_box"
python evaluate_icp.py "008_pudding_box" "ransac" "nc" >> $FILE_NAME
echo "009_gelatin_box"
python evaluate_icp.py "009_gelatin_box" "ransac" "nc" >> $FILE_NAME
echo "010_potted_meat_can"
python evaluate_icp.py "010_potted_meat_can" "ransac" "nc" >> $FILE_NAME
echo "011_banana"
python evaluate_icp.py "011_banana" "ransac" "nc" >> $FILE_NAME
echo "019_pitcher_base"
python evaluate_icp.py "019_pitcher_base" "ransac" "nc" >> $FILE_NAME
echo "021_bleach_cleanser"
python evaluate_icp.py "021_bleach_cleanser" "ransac" "nc" >> $FILE_NAME
echo "035_power_drill"
python evaluate_icp.py "035_power_drill" "ransac" "nc" >> $FILE_NAME
echo "036_wood_block"
python evaluate_icp.py "036_wood_block" "ransac" "nc" >> $FILE_NAME
echo "037_scissors"
python evaluate_icp.py "037_scissors" "ransac" "nc" >> $FILE_NAME
echo "051_large_clamp"
python evaluate_icp.py "051_large_clamp" "ransac" "nc" >> $FILE_NAME
echo "052_extra_large_clamp"
python evaluate_icp.py "052_extra_large_clamp" "ransac" "nc" >> $FILE_NAME
echo "061_foam_brick"
python evaluate_icp.py "061_foam_brick" "ransac" "nc" >> $FILE_NAME


now=$(date +'%c')
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "Keypoint Registration+ICP: no corrector"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME

echo "001_chips_can"
python evaluate_icp.py "001_chips_can" "none" "nc" >> $FILE_NAME
echo "002_master_chef_can"
python evaluate_icp.py "002_master_chef_can" "none" "nc" >> $FILE_NAME
echo "003_cracker_box"
python evaluate_icp.py "003_cracker_box" "none" "nc" >> $FILE_NAME
echo "004_sugar_box"
python evaluate_icp.py "004_sugar_box" "none" "nc" >> $FILE_NAME
echo "005_tomato_soup_can"
python evaluate_icp.py "005_tomato_soup_can" "none" "nc" >> $FILE_NAME
echo "006_mustard_bottle"
python evaluate_icp.py "006_mustard_bottle" "none" "nc" >> $FILE_NAME
echo "007_tuna_fish_can"
python evaluate_icp.py "007_tuna_fish_can" "none" "nc" >> $FILE_NAME
echo "008_pudding_box"
python evaluate_icp.py "008_pudding_box" "none" "nc" >> $FILE_NAME
echo "009_gelatin_box"
python evaluate_icp.py "009_gelatin_box" "none" "nc" >> $FILE_NAME
echo "010_potted_meat_can"
python evaluate_icp.py "010_potted_meat_can" "none" "nc" >> $FILE_NAME
echo "011_banana"
python evaluate_icp.py "011_banana" "none" "nc" >> $FILE_NAME
echo "019_pitcher_base"
python evaluate_icp.py "019_pitcher_base" "none" "nc" >> $FILE_NAME
echo "021_bleach_cleanser"
python evaluate_icp.py "021_bleach_cleanser" "none" "nc" >> $FILE_NAME
echo "035_power_drill"
python evaluate_icp.py "035_power_drill" "none" "nc" >> $FILE_NAME
echo "036_wood_block"
python evaluate_icp.py "036_wood_block" "none" "nc" >> $FILE_NAME
echo "037_scissors"
python evaluate_icp.py "037_scissors" "none" "nc" >> $FILE_NAME
echo "051_large_clamp"
python evaluate_icp.py "051_large_clamp" "none" "nc" >> $FILE_NAME
echo "052_extra_large_clamp"
python evaluate_icp.py "052_extra_large_clamp" "none" "nc" >> $FILE_NAME
echo "061_foam_brick"
python evaluate_icp.py "061_foam_brick" "none" "nc" >> $FILE_NAME



##now=$(date +'%c')
##echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
##echo "TEASER+ICP: no corrector"
##echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
##echo "$now" >> $FILE_NAME
##
##echo "airplane"
##python evaluate_icp.py "airplane" "teaser" "nc" >> $FILE_NAME
##echo "bathtub"
##python evaluate_icp.py "bathtub" "teaser" "nc" >> $FILE_NAME
##echo "bed"
##python evaluate_icp.py "bed" "teaser" "nc" >> $FILE_NAME
##echo "bottle"
##python evaluate_icp.py "bottle" "teaser" "nc" >> $FILE_NAME
##echo "cap"
##python evaluate_icp.py "cap" "teaser" "nc" >> $FILE_NAME
##echo "car"
##python evaluate_icp.py "car" "teaser" "nc" >> $FILE_NAME
##echo "chair"
##python evaluate_icp.py "chair" "teaser" "nc" >> $FILE_NAME
##echo "guitar"
##python evaluate_icp.py "guitar" "teaser" "nc" >> $FILE_NAME
##echo "helmet"
##python evaluate_icp.py "helmet" "teaser" "nc" >> $FILE_NAME
##echo "knife"
##python evaluate_icp.py "knife" "teaser" "nc" >> $FILE_NAME
##echo "laptop"
##python evaluate_icp.py "laptop" "teaser" "nc" >> $FILE_NAME
##echo "motorcycle"
##python evaluate_icp.py "motorcycle" "teaser" "nc" >> $FILE_NAME
##echo "mug"
##python evaluate_icp.py "mug" "teaser" "nc" >> $FILE_NAME
##echo "skateboard"
##python evaluate_icp.py "skateboard" "teaser" "nc" >> $FILE_NAME
##echo "table"
##python evaluate_icp.py "table" "teaser" "nc" >> $FILE_NAME
##echo "vessel"
##python evaluate_icp.py "vessel" "teaser" "nc" >> $FILE_NAME
##
#
now=$(date +'%c')
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "RANSAC+ICP: w corrector"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME

echo "001_chips_can"
python evaluate_icp.py "001_chips_can" "ransac" "c" >> $FILE_NAME
echo "002_master_chef_can"
python evaluate_icp.py "002_master_chef_can" "ransac" "c" >> $FILE_NAME
echo "003_cracker_box"
python evaluate_icp.py "003_cracker_box" "ransac" "c" >> $FILE_NAME
echo "004_sugar_box"
python evaluate_icp.py "004_sugar_box" "ransac" "c" >> $FILE_NAME
echo "005_tomato_soup_can"
python evaluate_icp.py "005_tomato_soup_can" "ransac" "c" >> $FILE_NAME
echo "006_mustard_bottle"
python evaluate_icp.py "006_mustard_bottle" "ransac" "c" >> $FILE_NAME
echo "007_tuna_fish_can"
python evaluate_icp.py "007_tuna_fish_can" "ransac" "c" >> $FILE_NAME
echo "008_pudding_box"
python evaluate_icp.py "008_pudding_box" "ransac" "c" >> $FILE_NAME
echo "009_gelatin_box"
python evaluate_icp.py "009_gelatin_box" "ransac" "c" >> $FILE_NAME
echo "010_potted_meat_can"
python evaluate_icp.py "010_potted_meat_can" "ransac" "c" >> $FILE_NAME
echo "011_banana"
python evaluate_icp.py "011_banana" "ransac" "c" >> $FILE_NAME
echo "019_pitcher_base"
python evaluate_icp.py "019_pitcher_base" "ransac" "c" >> $FILE_NAME
echo "021_bleach_cleanser"
python evaluate_icp.py "021_bleach_cleanser" "ransac" "c" >> $FILE_NAME
echo "035_power_drill"
python evaluate_icp.py "035_power_drill" "ransac" "c" >> $FILE_NAME
echo "036_wood_block"
python evaluate_icp.py "036_wood_block" "ransac" "c" >> $FILE_NAME
echo "037_scissors"
python evaluate_icp.py "037_scissors" "ransac" "c" >> $FILE_NAME
echo "051_large_clamp"
python evaluate_icp.py "051_large_clamp" "ransac" "c" >> $FILE_NAME
echo "052_extra_large_clamp"
python evaluate_icp.py "052_extra_large_clamp" "ransac" "c" >> $FILE_NAME
echo "061_foam_brick"
python evaluate_icp.py "061_foam_brick" "ransac" "c" >> $FILE_NAME


now=$(date +'%c')
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "Keypoint Registration+ICP: corrector"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
#
echo "001_chips_can"
python evaluate_icp.py "001_chips_can" "none" "c" >> $FILE_NAME
echo "002_master_chef_can"
python evaluate_icp.py "002_master_chef_can" "none" "c" >> $FILE_NAME
echo "003_cracker_box"
python evaluate_icp.py "003_cracker_box" "none" "c" >> $FILE_NAME
echo "004_sugar_box"
python evaluate_icp.py "004_sugar_box" "none" "c" >> $FILE_NAME
echo "005_tomato_soup_can"
python evaluate_icp.py "005_tomato_soup_can" "none" "c" >> $FILE_NAME
echo "006_mustard_bottle"
python evaluate_icp.py "006_mustard_bottle" "none" "c" >> $FILE_NAME
echo "007_tuna_fish_can"
python evaluate_icp.py "007_tuna_fish_can" "none" "c" >> $FILE_NAME
echo "008_pudding_box"
python evaluate_icp.py "008_pudding_box" "none" "c" >> $FILE_NAME
echo "009_gelatin_box"
python evaluate_icp.py "009_gelatin_box" "none" "c" >> $FILE_NAME
echo "010_potted_meat_can"
python evaluate_icp.py "010_potted_meat_can" "none" "c" >> $FILE_NAME
echo "011_banana"
python evaluate_icp.py "011_banana" "none" "c" >> $FILE_NAME
echo "019_pitcher_base"
python evaluate_icp.py "019_pitcher_base" "none" "c" >> $FILE_NAME
echo "021_bleach_cleanser"
python evaluate_icp.py "021_bleach_cleanser" "none" "c" >> $FILE_NAME
echo "035_power_drill"
python evaluate_icp.py "035_power_drill" "none" "c" >> $FILE_NAME
echo "036_wood_block"
python evaluate_icp.py "036_wood_block" "none" "c" >> $FILE_NAME
echo "037_scissors"
python evaluate_icp.py "037_scissors" "none" "c" >> $FILE_NAME
echo "051_large_clamp"
python evaluate_icp.py "051_large_clamp" "none" "c" >> $FILE_NAME
echo "052_extra_large_clamp"
python evaluate_icp.py "052_extra_large_clamp" "none" "c" >> $FILE_NAME
echo "061_foam_brick"
python evaluate_icp.py "061_foam_brick" "none" "c" >> $FILE_NAME


#now=$(date +'%c')
#echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
#echo "TEASER+ICP: w corrector"
#echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
#echo "$now" >> $FILE_NAME
#
#echo "airplane"
#python evaluate_icp.py "airplane" "teaser" "c" >> $FILE_NAME
#echo "bathtub"
#python evaluate_icp.py "bathtub" "teaser" "c" >> $FILE_NAME
#echo "bed"
#python evaluate_icp.py "bed" "teaser" "c" >> $FILE_NAME
#echo "bottle"
#python evaluate_icp.py "bottle" "teaser" "c" >> $FILE_NAME
#echo "cap"
#python evaluate_icp.py "cap" "teaser" "c" >> $FILE_NAME
#echo "car"
#python evaluate_icp.py "car" "teaser" "c" >> $FILE_NAME
#echo "chair"
#python evaluate_icp.py "chair" "teaser" "c" >> $FILE_NAME
#echo "guitar"
#python evaluate_icp.py "guitar" "teaser" "c" >> $FILE_NAME
#echo "helmet"
#python evaluate_icp.py "helmet" "teaser" "c" >> $FILE_NAME
#echo "knife"
#python evaluate_icp.py "knife" "teaser" "c" >> $FILE_NAME
#echo "laptop"
#python evaluate_icp.py "laptop" "teaser" "c" >> $FILE_NAME
#echo "motorcycle"
#python evaluate_icp.py "motorcycle" "teaser" "c" >> $FILE_NAME
#echo "mug"
#python evaluate_icp.py "mug" "teaser" "c" >> $FILE_NAME
#echo "skateboard"
#python evaluate_icp.py "skateboard" "teaser" "c" >> $FILE_NAME
#echo "table"
#python evaluate_icp.py "table" "teaser" "c" >> $FILE_NAME
#echo "vessel"
#python evaluate_icp.py "vessel" "teaser" "c" >> $FILE_NAME
#
#
