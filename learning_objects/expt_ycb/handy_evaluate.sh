FILE_NAME="./eval/model_eval_main.txt"

now=$(date +'%c')
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
############################################################################################
echo "EVALUATING POINT TRANSFORMER, post" >> $FILE_NAME
############################################################################################

echo "001_chips_can self-supervised evaluation:: POINT TRANSFORMER, post"
python evaluate_proposed_model.py "point_transformer" "001_chips_can" "post" >> $FILE_NAME

##########################################################################

echo "002_master_chef_can self-supervised evaluation:: POINT TRANSFORMER, post"
python evaluate_proposed_model.py "point_transformer" "002_master_chef_can" "post" >> $FILE_NAME

#######################################################################

echo "003_cracker_box self-supervised evaluation:: POINT TRANSFORMER, post"
python evaluate_proposed_model.py "point_transformer" "003_cracker_box" "post" >> $FILE_NAME

#########################################################################

echo "004_sugar_box self-supervised evaluation:: POINT TRANSFORMER, post"
python evaluate_proposed_model.py "point_transformer" "004_sugar_box" "post" >> $FILE_NAME

########################################################################

echo "005_tomato_soup_can self-supervised evaluation:: POINT TRANSFORMER, post"
python evaluate_proposed_model.py "point_transformer" "005_tomato_soup_can" "post" >> $FILE_NAME

########################################################################

echo "006_mustard_bottle self-supervised evaluation:: POINT TRANSFORMER, post"
python evaluate_proposed_model.py "point_transformer" "006_mustard_bottle" "post" >> $FILE_NAME

########################################################################

echo "007_tuna_fish_can self-supervised evaluation:: POINT TRANSFORMER, post"
python evaluate_proposed_model.py "point_transformer" "007_tuna_fish_can" "post" >> $FILE_NAME

#########################################################################

echo "008_pudding_box self-supervised evaluation:: POINT TRANSFORMER, post"
python evaluate_proposed_model.py "point_transformer" "008_pudding_box" "post" >> $FILE_NAME

##########################################################################

echo "009_gelatin_box self-supervised evaluation:: POINT TRANSFORMER, post"
python evaluate_proposed_model.py "point_transformer" "009_gelatin_box" "post" >> $FILE_NAME

#########################################################################

echo "010_potted_meat_can self-supervised evaluation:: POINT TRANSFORMER, post"
python evaluate_proposed_model.py "point_transformer" "010_potted_meat_can" "post" >> $FILE_NAME

##########################################################################

echo "011_banana self-supervised evaluation:: POINT TRANSFORMER, post"
python evaluate_proposed_model.py "point_transformer" "011_banana" "post" >> $FILE_NAME

##########################################################################

echo "019_pitcher_base self-supervised evaluation:: POINT TRANSFORMER, post"
python evaluate_proposed_model.py "point_transformer" "019_pitcher_base" "post" >> $FILE_NAME

###########################################################################

echo "021_bleach_cleanser self-supervised evaluation:: POINT TRANSFORMER, post"
python evaluate_proposed_model.py "point_transformer" "021_bleach_cleanser" "post" >> $FILE_NAME

#########################################################################

echo "035_power_drill self-supervised evaluation:: POINT TRANSFORMER, post"
python evaluate_proposed_model.py "point_transformer" "035_power_drill" "post" >> $FILE_NAME

##########################################################################

echo "036_wood_block self-supervised evaluation:: POINT TRANSFORMER, post"
python evaluate_proposed_model.py "point_transformer" "036_wood_block" "post" >> $FILE_NAME

##########################################################################

echo "037_scissors self-supervised evaluation:: POINT TRANSFORMER, post"
python evaluate_proposed_model.py "point_transformer" "037_scissors" "post" >> $FILE_NAME

#########################################################################

echo "051_large_clamp self-supervised evaluation:: POINT TRANSFORMER, post"
python evaluate_proposed_model.py "point_transformer" "051_large_clamp" "post" >> $FILE_NAME

#########################################################################

echo "052_extra_large_clamp self-supervised evaluation:: POINT TRANSFORMER, post"
python evaluate_proposed_model.py "point_transformer" "052_extra_large_clamp" "post" >> $FILE_NAME

###########################################################################

echo "061_foam_brick self-supervised evaluation:: POINT TRANSFORMER, post"
python evaluate_proposed_model.py "point_transformer" "061_foam_brick" "post" >> $FILE_NAME

##########################################################################

