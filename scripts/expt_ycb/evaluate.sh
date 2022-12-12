cd ../../c3po/expt_ycb

DETECTOR_TYPE="point_transformer"
YCB_OBJECTS="001_chips_can 002_master_chef_can 003_cracker_box 004_sugar_box 005_tomato_soup_can 006_mustard_bottle \
007_tuna_fish_can 008_pudding_box 009_gelatin_box 010_potted_meat_can 011_banana 019_pitcher_base 021_bleach_cleanser \
035_power_drill 036_wood_block 037_scissors 040_large_marker 051_large_clamp 052_extra_large_clamp 061_foam_brick"

# KeyPoSim
for object in $YCB_OBJECTS
do
  python evaluate_sim_supervised_model.py $DETECTOR_TYPE $object
done
echo "--------------------------------------------------------------------------"

# KeyPoSimICP
for object in $YCB_OBJECTS
do
  echo $object
  python evaluate_icp.py $object "none" "nc"
done
echo "--------------------------------------------------------------------------"

# KeyPoSimRANSACICP
for object in $YCB_OBJECTS
do
  echo $object
  python evaluate_icp.py $object "ransac" "nc"
done
echo "--------------------------------------------------------------------------"

# KeyPoSimCor
for object in $YCB_OBJECTS
do
  echo $object
  python evaluate_proposed_model.py $DETECTOR_TYPE $object "pre"
done
echo "--------------------------------------------------------------------------"

# KeyPoSimCorICP
for object in $YCB_OBJECTS
do
  echo $object
  python evaluate_icp.py $object "none" "c"
done
echo "--------------------------------------------------------------------------"

# KeyPoSimCorRANSACICP
for object in $YCB_OBJECTS
do
  echo $object
  python evaluate_icp.py $object "ransac" "c"
done
echo "--------------------------------------------------------------------------"

# c3po
for object in $YCB_OBJECTS
do
  echo $object
  python evaluate_proposed_model.py $DETECTOR_TYPE $object "post"
done
echo "--------------------------------------------------------------------------"

# KeyPoReal
for object in $YCB_OBJECTS
do
  echo $object
  python evaluate_baseline.py $DETECTOR_TYPE $object
done
echo "--------------------------------------------------------------------------"


