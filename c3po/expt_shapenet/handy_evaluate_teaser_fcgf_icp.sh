
OBJECT_LIST='airplane bathtub bed bottle cap car chair guitar helmet knife laptop motorcycle mug skateboard table vessel'
FILE_NAME="./eval/eval_teaser_fcgf_icp_v8.log"
MODEL_FILE="../baselines/fcgf/data/output_shapenet/realv8/real/best_val_checkpoint.pth"
MODEL_DIR="../baselines/fcgf/data/output_shapenet/realv8/real"


# Computes ADD-S, ADD-S AUC, and % certifiable for each object category
FOLDER_NAME="./temp"
mkdir $FOLDER_NAME

now=$(date +'%c')
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "TEASER + FCGF + ICP: baseline" >> $FILE_NAME
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME

for OBJECT in $OBJECT_LIST
do
  echo $OBJECT
  python -u evaluate_teaser_fcgf_icp.py \
  --object $OBJECT \
  --model $MODEL_FILE \
  --pre y \
  --folder $FOLDER_NAME >> $FILE_NAME

  # --pre is 'y' if we are using the pre-trained model either as initialization or for evaluation.
  # The default is 'y'.
done

python print_eval_results.py --folder $FOLDER_NAME --objects "${OBJECT_LIST}" --filename "results"

rm -r $FOLDER_NAME

# Plots ADD-S, ADD-S AUC, and Training Loss as a function of #epoch
python teaser_fcgf_icp_plot_loss.py --object 'helmet' --model_dir $MODEL_DIR --epochs 10
