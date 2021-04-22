python network/test.py --config=config_track.yml --obj_config=obj_info_sapien.yml \
                       --mode_name=test_seq \
                       --pose_perturb/r=5 --pose_perturb/t=0.02 --pose_perturb/s=0.02 \
                       --batch_size=4 \
                       --obj_category=glasses \
                       --experiment_dir=../runs/glasses_rot \
                       --coord_exp/dir=../runs/glasses_coord \
                       --save --no_eval
