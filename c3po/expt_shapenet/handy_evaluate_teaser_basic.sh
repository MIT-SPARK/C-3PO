FILE_NAME="./eval/eval_teaser_basic.txt"

now=$(date +'%c')
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "TEASER: baseline"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME

# teaser with no features, just point clouds
python evaluate_teaser.py --object airplane >> $FILE_NAME
python evaluate_teaser.py --object bathtub >> $FILE_NAME
python evaluate_teaser.py --object bed >> $FILE_NAME
python evaluate_teaser.py --object bottle >> $FILE_NAME
python evaluate_teaser.py --object cap >> $FILE_NAME
python evaluate_teaser.py --object car >> $FILE_NAME
python evaluate_teaser.py --object chair >> $FILE_NAME
python evaluate_teaser.py --object guitar >> $FILE_NAME
python evaluate_teaser.py --object helmet >> $FILE_NAME
python evaluate_teaser.py --object knife >> $FILE_NAME
python evaluate_teaser.py --object laptop >> $FILE_NAME
python evaluate_teaser.py --object motorcycle >> $FILE_NAME
python evaluate_teaser.py --object mug >> $FILE_NAME
python evaluate_teaser.py --object skateboard >> $FILE_NAME
python evaluate_teaser.py --object table >> $FILE_NAME
python evaluate_teaser.py --object vessel >> $FILE_NAME