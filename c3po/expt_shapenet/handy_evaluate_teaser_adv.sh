FILE_NAME="./eval/eval_teaser_adv.txt"

now=$(date +'%c')
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "TEASER + FPFH + ICP: baseline"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME

# teaser + fpfh + icp 
python evaluate_teaser_fpfh_icp.py --object airplane >> $FILE_NAME
python evaluate_teaser_fpfh_icp.py --object bathtub >> $FILE_NAME
python evaluate_teaser_fpfh_icp.py --object bed >> $FILE_NAME
python evaluate_teaser_fpfh_icp.py --object bottle >> $FILE_NAME
python evaluate_teaser_fpfh_icp.py --object cap >> $FILE_NAME
python evaluate_teaser_fpfh_icp.py --object car >> $FILE_NAME
python evaluate_teaser_fpfh_icp.py --object chair >> $FILE_NAME
python evaluate_teaser_fpfh_icp.py --object guitar >> $FILE_NAME
python evaluate_teaser_fpfh_icp.py --object helmet >> $FILE_NAME
python evaluate_teaser_fpfh_icp.py --object knife >> $FILE_NAME
python evaluate_teaser_fpfh_icp.py --object laptop >> $FILE_NAME
python evaluate_teaser_fpfh_icp.py --object motorcycle >> $FILE_NAME
python evaluate_teaser_fpfh_icp.py --object mug >> $FILE_NAME
python evaluate_teaser_fpfh_icp.py --object skateboard >> $FILE_NAME
python evaluate_teaser_fpfh_icp.py --object table >> $FILE_NAME
python evaluate_teaser_fpfh_icp.py --object vessel >> $FILE_NAME

now=$(date +'%c')
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "TEASER + FCGF + ICP: baseline"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $FILE_NAME
echo "$now" >> $FILE_NAME

# teaser + pre-trained fcgf (on 3DMatch) + icp
python evaluate_teaser_fcgf_icp.py --object airplane >> $FILE_NAME
python evaluate_teaser_fcgf_icp.py --object bathtub >> $FILE_NAME
python evaluate_teaser_fcgf_icp.py --object bed >> $FILE_NAME
python evaluate_teaser_fcgf_icp.py --object bottle >> $FILE_NAME
python evaluate_teaser_fcgf_icp.py --object cap >> $FILE_NAME
python evaluate_teaser_fcgf_icp.py --object car >> $FILE_NAME
python evaluate_teaser_fcgf_icp.py --object chair >> $FILE_NAME
python evaluate_teaser_fcgf_icp.py --object guitar >> $FILE_NAME
python evaluate_teaser_fcgf_icp.py --object helmet >> $FILE_NAME
python evaluate_teaser_fcgf_icp.py --object knife >> $FILE_NAME
python evaluate_teaser_fcgf_icp.py --object laptop >> $FILE_NAME
python evaluate_teaser_fcgf_icp.py --object motorcycle >> $FILE_NAME
python evaluate_teaser_fcgf_icp.py --object mug >> $FILE_NAME
python evaluate_teaser_fcgf_icp.py --object skateboard >> $FILE_NAME
python evaluate_teaser_fcgf_icp.py --object table >> $FILE_NAME
python evaluate_teaser_fcgf_icp.py --object vessel >> $FILE_NAME
