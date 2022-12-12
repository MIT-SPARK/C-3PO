cd ../../c3po/expt_shapenet/

DETECTOR_TYPE="point_transformer"
SHAPENET_OBJECTS='airplane bathtub bed bottle cap car chair guitar helmet knife laptop motorcycle mug skateboard table vessel'


## KeyPoSim
#for object in $SHAPENET_OBJECTS
#do
#  echo $object
#  python evaluate_sim_supervised_model.py $DETECTOR_TYPE $object
#done
#echo "--------------------------------------------------------------------------"

# KeyPoSimICP
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_icp.py $object "none" "nc"
done
echo "--------------------------------------------------------------------------"

# KeyPoSimRANSACICP
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_icp.py $object "ransac" "nc"
done
echo "--------------------------------------------------------------------------"

## KeyPoSimCor
#for object in $SHAPENET_OBJECTS
#do
#  echo $object
#  python evaluate_proposed_model.py $DETECTOR_TYPE $object "pre"
#done
#echo "--------------------------------------------------------------------------"

# KeyPoSimCorICP
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_icp.py $object "none" "c"
done
echo "--------------------------------------------------------------------------"

# KeyPoSimCorRANSACICP
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_icp.py $object "ransac" "c"
done
echo "--------------------------------------------------------------------------"

## c3po
#for object in $SHAPENET_OBJECTS
#do
#  echo $object
#  python evaluate_proposed_model.py $DETECTOR_TYPE $object "post"
#done
#echo "--------------------------------------------------------------------------"
#
## KeyPoReal
#for object in $SHAPENET_OBJECTS
#do
#  echo $object
#  python evaluate_baseline.py $DETECTOR_TYPE $object
#done
#echo "--------------------------------------------------------------------------"

echo "--------------------------------------------------------------------------"
DETECTOR_TYPE="pointnet"

## KeyPoSim
#for object in $SHAPENET_OBJECTS
#do
#  echo $object
#  python evaluate_sim_supervised_model.py $DETECTOR_TYPE $object
#done
#echo "--------------------------------------------------------------------------"

# KeyPoSimICP
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_icp.py $object "none" "nc"
done
echo "--------------------------------------------------------------------------"

# KeyPoSimRANSACICP
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_icp.py $object "ransac" "nc"
done
echo "--------------------------------------------------------------------------"

## KeyPoSimCor
#for object in $SHAPENET_OBJECTS
#do
#  echo $object
#  python evaluate_proposed_model.py $DETECTOR_TYPE $object "pre"
#done
#echo "--------------------------------------------------------------------------"

# KeyPoSimCorICP
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_icp.py $object "none" "c"
done
echo "--------------------------------------------------------------------------"

# KeyPoSimCorRANSACICP
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_icp.py $object "ransac" "c"
done
echo "--------------------------------------------------------------------------"

## c3po
#for object in $SHAPENET_OBJECTS
#do
#  echo $object
#  python evaluate_proposed_model.py $DETECTOR_TYPE $object "post"
#done
#echo "--------------------------------------------------------------------------"
#
## KeyPoReal
#for object in $SHAPENET_OBJECTS
#do
#  echo $object
#  python evaluate_baseline.py $DETECTOR_TYPE $object
#done
#echo "--------------------------------------------------------------------------"
