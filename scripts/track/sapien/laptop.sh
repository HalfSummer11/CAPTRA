python network/test.py --config=config_track.yml --obj_config=obj_info_sapien.yml \
                       --mode_name=test_seq \
                       --pose_perturb/r=3 --pose_perturb/t=0.02 --pose_perturb/s=0.015 \
                       --batch_size=6 \
                       --obj_category=laptop \
                       --experiment_dir=../runs/laptop_rot \
                       --coord_exp/dir=../runs/laptop_coord \
                       --save --no_eval
