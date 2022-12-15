cd ../../c3po/expt_shapenet/

#DATASETS="shapenet.sim.easy shapenet.sim.hard"
DATASETS="shapenet.sim.easy"
DETECTOR_TYPE="point_transformer pointnet"
SHAPENET_OBJECTS='airplane bathtub bed bottle cap car chair guitar helmet knife laptop motorcycle mug skateboard table vessel'


for dset in $DATASETS
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

#    # c3po
#    for object in $SHAPENET_OBJECTS
#    do
#      echo $object
#      python evaluate_proposed_model.py \
#      --detector $detector \
#      --object $object \
#      --model "post" \
#      --dataset $dset
#    done
#    echo "--------------------------------------------------------------------------"
#
#    # KeyPoReal
#    for object in $SHAPENET_OBJECTS
#    do
#      echo $object
#      python evaluate_baseline.py \
#      --detector $detector \
#      --object $object \
#      --dataset $dset
#    done
#    echo "--------------------------------------------------------------------------"

  done
done
