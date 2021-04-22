# RotationNet
python network/train.py --config config_rotnet.yml --obj_config obj_info_sapien.yml \
                        --pose_perturb/r=3 --pose_perturb/t=0.01 --pose_perturb/s=0.01 \
                        --batch_size=6 \
                        --obj_category=scissors \
                        --experiment_dir=../runs/scissors_rot_new

# CoordinateNet
python network/train.py --config config_coordnet.yml --obj_config obj_info_sapien.yml \
                        --pose_perturb/r=3 --pose_perturb/t=0.01 --pose_perturb/s=0.01 \
                        --batch_size=12 \
                        --obj_category=scissors \
                        --experiment_dir=../runs/scissors_coord_new
