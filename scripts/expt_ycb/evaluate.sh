cd ../../c3po/expt_ycb

#DATASETS="ycb ycb.sim ycb.real"
DATASETS="ycb.sim ycb.real"
DETECTOR_TYPE="point_transformer"
YCB_OBJECTS="001_chips_can 002_master_chef_can 003_cracker_box 004_sugar_box 005_tomato_soup_can 006_mustard_bottle \
007_tuna_fish_can 008_pudding_box 009_gelatin_box 010_potted_meat_can 011_banana 019_pitcher_base 021_bleach_cleanser \
035_power_drill 036_wood_block 037_scissors 051_large_clamp 052_extra_large_clamp 061_foam_brick"

DATASES="ycb.real"
DETECTOR_TYPE="point_transformer"
SHAPENET_OBJECTS="008_pudding_box"

for dset in $DATASES
do
  echo $dset
  for detector in $DETECTOR_TYPE
  do

    echo $detector

    # KeyPoSim
    for object in $SHAPENET_OBJECTS
    do
      echo $object
      python evaluate_sim_supervised_model.py \
      --detector $detector \
      --object $object \
      --dataset $dset
    done
    echo "--------------------------------------------------------------------------"

    # KeyPoSimICP
    for object in $SHAPENET_OBJECTS
    do
      echo $object
      python evaluate_icp.py \
      --object $object \
      --gr "none" \
      --c "nc" \
      --detector $detector \
      --dataset $dset
    done
    echo "--------------------------------------------------------------------------"

    # KeyPoSimRANSACICP
    for object in $SHAPENET_OBJECTS
    do
      echo $object
      python evaluate_icp.py \
      --object $object \
      --gr "ransac" \
      --c "nc" \
      --detector $detector \
      --dataset $dset
    done
    echo "--------------------------------------------------------------------------"

    # KeyPoSimCor
    for object in $SHAPENET_OBJECTS
    do
      echo $object
      python evaluate_proposed_model.py \
      --detector $detector \
      --object $object \
      --model "pre" \
      --dataset $dset
    done
    echo "--------------------------------------------------------------------------"

    # KeyPoSimCorICP
    for object in $SHAPENET_OBJECTS
    do
      echo $object
      python evaluate_icp.py \
      --object $object \
      --gr "none" \
      --c "c" \
      --detector $detector \
      --dataset $dset
    done
    echo "--------------------------------------------------------------------------"

    # KeyPoSimCorRANSACICP
    for object in $SHAPENET_OBJECTS
    do
      echo $object
      python evaluate_icp.py \
      --object $object \
      --gr "ransac" \
      --c "c" \
      --detector $detector \
      --dataset $dset
    done
    echo "--------------------------------------------------------------------------"

    # c3po
    for object in $SHAPENET_OBJECTS
    do
      echo $object
      python evaluate_proposed_model.py \
      --detector $detector \
      --object $object \
      --model "post" \
      --dataset $dset
    done
    echo "--------------------------------------------------------------------------"

    # KeyPoReal
    for object in $SHAPENET_OBJECTS
    do
      echo $object
      python evaluate_baseline.py \
      --detector $detector \
      --object $object \
      --dataset $dset
    done
    echo "--------------------------------------------------------------------------"



  done
done