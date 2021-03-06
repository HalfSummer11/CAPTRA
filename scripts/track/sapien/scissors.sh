python network/test.py --config=config_track.yml --obj_config=obj_info_sapien.yml \
                       --mode_name=test_seq \
                       --pose_perturb/r=3 --pose_perturb/t=0.01 --pose_perturb/s=0.01 \
                       --batch_size=6 \
                       --obj_category=scissors \
                       --experiment_dir=../runs/scissors_rot \
                       --coord_exp/dir=../runs/scissors_coord \
                       --save --no_eval
