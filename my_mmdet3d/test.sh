
#!/bin/bash
CHECKPOINT_FILE="configs/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth"
CONFIG_FILE="configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py"
GPU_NUM=2
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} --task lidar_det