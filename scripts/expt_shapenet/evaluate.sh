cd ../../c3po/expt_shapenet/

DETECTOR_TYPE="point_transformer"
SHAPENET_OBJECTS='airplane bathtub bed bottle cap car chair guitar helmet knife laptop motorcycle mug skateboard table vessel'


# KeyPoSim
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_sim_supervised_model.py $DETECTOR_TYPE $object
done

# KeyPoSimCor
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_proposed_model.py $DETECTOR_TYPE $object "pre"
done

# c3po
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_proposed_model.py $DETECTOR_TYPE $object "post"
done

# KeyPoReal
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_baseline.py $DETECTOR_TYPE $object
done



DETECTOR_TYPE="pointnet"

# KeyPoSim
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_sim_supervised_model.py $DETECTOR_TYPE $object
done

# KeyPoSimCor
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_proposed_model.py $DETECTOR_TYPE $object "pre"
done

# c3po
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_proposed_model.py $DETECTOR_TYPE $object "post"
done

# KeyPoReal
for object in $SHAPENET_OBJECTS
do
  echo $object
  python evaluate_baseline.py $DETECTOR_TYPE $object
done

