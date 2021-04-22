# RotationNet
python network/train.py --config config_rotnet.yml --obj_config obj_info_sapien.yml \
                        --pose_perturb/r=5 --pose_perturb/t=0.02 --pose_perturb/s=0.02 \
                        --batch_size=4 \
                        --obj_category=glasses \
                        --experiment_dir=../runs/glasses_rot_new

# CoordinateNet
python network/train.py --config config_coordnet.yml --obj_config obj_info_sapien.yml \
                        --pose_perturb/r=5 --pose_perturb/t=0.02 --pose_perturb/s=0.02 \
                        --batch_size=12 \
                        --obj_category=glasses \
                        --experiment_dir=../runs/glasses_coord_new
