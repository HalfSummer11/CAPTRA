# 1_bottle
python misc/eval/eval.py --config config_track.yml --obj_config obj_info_nocs.yml --obj_category=1 --experiment_dir=../runs/1_bottle_rot
# 2_bowl
python misc/eval/eval.py --config config_track.yml --obj_config obj_info_nocs.yml --obj_category=2 --experiment_dir=../runs/2_bowl_rot
# 3_camera
python misc/eval/eval.py --config config_track.yml --obj_config obj_info_nocs.yml --obj_category=3 --experiment_dir=../runs/3_camera_rot
# 4 can
python misc/eval/eval.py --config config_track.yml --obj_config obj_info_nocs.yml --obj_category=4 --experiment_dir=../runs/4_can_rot
# 5 laptop
python misc/eval/eval.py --config config_track.yml --obj_config obj_info_nocs.yml --obj_category=5 --experiment_dir=../runs/5_laptop_rot
# 6 mug
python misc/eval/eval.py --config config_track.yml --obj_config obj_info_nocs.yml --obj_category=6 --experiment_dir=../runs/6_mug_rot

# glasses
python misc/eval/eval.py --config config_track.yml --obj_config obj_info_sapien.yml --obj_category=glasses --experiment_dir=../runs/glasses_rot

# laptop
python misc/eval/eval.py --config config_track.yml --obj_config obj_info_sapien.yml --obj_category=laptop --experiment_dir=../runs/laptop_rot

# scissors
python misc/eval/eval.py --config config_track.yml --obj_config obj_info_sapien.yml --obj_category=scissors --experiment_dir=../runs/scissors_rot

# drawers
python misc/eval/eval.py --config config_track.yml --obj_config obj_info_sapien.yml --obj_category=drawers --experiment_dir=../runs/drawers_rot
