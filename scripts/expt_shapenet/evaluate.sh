cd ../../c3po/expt_shapenet/

DETECTOR_TYPE="point_transformer"
SHAPENET_OBJECTS='airplane bathtub bed bottle cap car chair guitar helmet knife laptop motorcycle mug skateboard table vessel'


# KeyPoSim
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_sim_supervised_model.py $DETECTOR_TYPE $object
done
echo "--------------------------------------------------------------------------"

# KeyPoSimICP
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_icp.py $object "none" "nc" $DETECTOR_TYPE
done
echo "--------------------------------------------------------------------------"

# KeyPoSimRANSACICP
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_icp.py $object "ransac" "nc" $DETECTOR_TYPE
done
echo "--------------------------------------------------------------------------"

# KeyPoSimCor
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_proposed_model.py \
  --detector $DETECTOR_TYPE \
  --object $object \
  --model "pre" \
  --dataset "shapenet"
done
echo "--------------------------------------------------------------------------"

# KeyPoSimCorICP
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_icp.py $object "none" "c" $DETECTOR_TYPE
done
echo "--------------------------------------------------------------------------"

# KeyPoSimCorRANSACICP
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_icp.py $object "ransac" "c" $DETECTOR_TYPE
done
echo "--------------------------------------------------------------------------"

# c3po
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_proposed_model.py \
  --detector $DETECTOR_TYPE \
  --object $object \
  --model "post" \
  --dataset "shapenet"
done
echo "--------------------------------------------------------------------------"

# KeyPoReal
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_baseline.py $DETECTOR_TYPE $object
done
echo "--------------------------------------------------------------------------"

echo "--------------------------------------------------------------------------"
DETECTOR_TYPE="pointnet"

# KeyPoSim
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_sim_supervised_model.py $DETECTOR_TYPE $object
done
echo "--------------------------------------------------------------------------"

# KeyPoSimICP
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_icp.py $object "none" "nc" $DETECTOR_TYPE
done
echo "--------------------------------------------------------------------------"

# KeyPoSimRANSACICP
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_icp.py $object "ransac" "nc" $DETECTOR_TYPE
done
echo "--------------------------------------------------------------------------"

# KeyPoSimCor
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_proposed_model.py \
  --detector $DETECTOR_TYPE \
  --object $object \
  --model "pre" \
  --dataset "shapenet"
done
echo "--------------------------------------------------------------------------"

# KeyPoSimCorICP
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_icp.py $object "none" "c" $DETECTOR_TYPE
done
echo "--------------------------------------------------------------------------"

# KeyPoSimCorRANSACICP
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_icp.py $object "ransac" "c" $DETECTOR_TYPE
done
echo "--------------------------------------------------------------------------"

# c3po
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_proposed_model.py \
  --detector $DETECTOR_TYPE \
  --object $object \
  --model "post" \
  --dataset "shapenet"
done
echo "--------------------------------------------------------------------------"

# KeyPoReal
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_baseline.py $DETECTOR_TYPE $object
done
echo "--------------------------------------------------------------------------"
