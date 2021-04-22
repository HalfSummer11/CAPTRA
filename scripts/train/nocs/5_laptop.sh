# RotationNet
python network/train.py --config=config_rotnet.yml --obj_config=obj_info_nocs.yml \
                        --pose_perturb/r=5.0 --pose_perturb/t=0.03 --pose_perturb/s=0.02 \
                        --batch_size=12 \
                        --obj_category=5 \
                        --experiment_dir=../runs/5_laptop_rot_new \
                        --use_val=real_test \
                        --num_workers=2
# CoordinateNet
python network/train.py --config=config_coordnet.yml --obj_config=obj_info_nocs.yml \
                        --pose_perturb/r=5.0 --pose_perturb/t=0.03 --pose_perturb/s=0.02 \
                        --batch_size=12 \
                        --obj_category=5 \
                        --experiment_dir=../runs/5_laptop_coord_new \
                        --use_val=real_test \
                        --num_workers=2

