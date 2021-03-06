python network/test.py --config=config_track.yml --obj_config=obj_info_sapien.yml \
                       --mode_name=test_seq \
                       --pose_perturb/r=3 --pose_perturb/t=0.02 --pose_perturb/s=0.02 \
                       --batch_size=3 \
                       --obj_category=drawers \
                       --experiment_dir=../runs/drawers_rot \
                       --coord_exp/dir=../runs/drawers_coord \
                       --save --no_eval
