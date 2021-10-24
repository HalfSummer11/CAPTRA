# nocs
python misc/visualize/visualize_tracking_nocs.py --img_path ../data/nocs_data/nocs_full/real_test --exp_path ../runs --output_path ../nocs_viz --save_fig

# glasses
python misc/visualize/visualize_tracking_sapien.py --config config_track.yml --obj_config obj_info_sapien.yml --obj_category=glasses --experiment_dir=../runs/glasses_rot --save_fig

# laptop
python misc/visualize/visualize_tracking_sapien.py --config config_track.yml --obj_config obj_info_sapien.yml --obj_category=laptop --experiment_dir=../runs/laptop_rot --save_fig

# scissors
python misc/visualize/visualize_tracking_sapien.py --config config_track.yml --obj_config obj_info_sapien.yml --obj_category=scissors --experiment_dir=../runs/scissors_rot --save_fig

# drawers
python misc/visualize/visualize_tracking_sapien.py --config config_track.yml --obj_config obj_info_sapien.yml --obj_category=drawers --experiment_dir=../runs/drawers_rot --save_fig
