
#!/bin/bash
#ONFIG_FILE="/home/alper/Desktop/Works/Tez/mmdetection3d/my_mmdet3d/3dssd_4xb4_kitti-3d-3class.py"
CONFIG_FILE="/home/alper/Desktop/Works/Tez/mmdetection3d/my_mmdet3d/centerpoint_005voxel_second_secfpn_4x8_cyclic_80e_kitti.py"
#CONFIG_FILE="/home/alper/Desktop/Works/Tez/mmdetection3d/my_mmdet3d/point-rcnn_8xb2_kitti-3d-3class.py"

GPU_NUM=2
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} --resume auto
#python ./tools/train.py ${CONFIG_FILE} 