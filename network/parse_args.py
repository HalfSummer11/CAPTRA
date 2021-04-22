from utils import boolean_string


def add_args(parser):
    parser.add_argument('--config', type=str, default='config.yml')
    parser.add_argument('--obj_config', type=str, default=None)
    parser.add_argument('--obj_category', type=str, default=None)
    parser.add_argument('--experiment_dir', type=str, default=None)
    parser.add_argument('--resume_epoch', type=int, default=-1)

    parser.add_argument('--coord_exp/dir', type=str, default=None)
    parser.add_argument('--coord_exp/resume_epoch', type=int, default=None)

    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--cuda_id', type=int, default=None)
    parser.add_argument('--total_epoch',  default=None, type=int)
    parser.add_argument('--optimizer', type=str, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--lr_policy', type=str, default=None)
    parser.add_argument('--lr_gamma', type=float, default=None)
    parser.add_argument('--lr_step_size', type=int, default=None)
    parser.add_argument('--lr_clip', type=float, default=None)

    parser.add_argument('--num_workers', type=int,  default=0, help='num_workers in data_loader')

    parser.add_argument('--num_points', type=int,  default=None)
    parser.add_argument('--data_radius', type=float,  default=None)
    parser.add_argument('--dataset_length', type=int,  default=None, help='truncate dataset to check the code quicker')

    parser.add_argument('--freq/save', type=int,  default=None, help='ckpt saving frequency in epochs')

    parser.add_argument('--pointnet_cfg/camera', type=str, default=None)

    parser.add_argument('--network/type', type=str, default=None)
    parser.add_argument('--network/nocs_head_dims', type=int, default=None)
    parser.add_argument('--network/backbone_out_dim', type=int, default=None)
    parser.add_argument('--network/pwm_num', type=int, default=None)

    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--eval_train', action='store_true', default=False)
    parser.add_argument('--no_eval', action='store_true', default=False)

    parser.add_argument('--init_frame/gt', type=boolean_string, default=None)

    parser.add_argument('--loss_weight/rloss', type=float, default=None)
    parser.add_argument('--loss_weight/tloss', type=float, default=None)
    parser.add_argument('--loss_weight/sloss', type=float, default=None)
    parser.add_argument('--loss_weight/corner_loss', type=float, default=None)
    parser.add_argument('--loss_weight/nocs_loss', type=float, default=None)
    parser.add_argument('--loss_weight/nocs_dist_loss', type=float, default=None)
    parser.add_argument('--loss_weight/nocs_pwm_loss', type=float, default=None)
    parser.add_argument('--loss_weight/seg_loss', type=float, default=None)

    parser.add_argument('--pose_loss_type/r', type=str, default=None)
    parser.add_argument('--pose_loss_type/s', type=str, default=None)
    parser.add_argument('--pose_loss_type/t', type=str, default=None)
    parser.add_argument('--pose_loss_type/point', type=str, default=None)

    parser.add_argument('--pose_perturb/type', type=str, default=None)
    parser.add_argument('--pose_perturb/r', type=float, default=None)
    parser.add_argument('--pose_perturb/s', type=float, default=None)
    parser.add_argument('--pose_perturb/t', type=float, default=None)

    parser.add_argument('--nocs_otf', type=boolean_string, default=None)

    parser.add_argument('--track_cfg/gt_label', type=boolean_string, default=None)
    parser.add_argument('--track_cfg/nocs2d_label', type=boolean_string, default=None)
    parser.add_argument('--track_cfg/nocs2d_path', type=str, default=None)

    return parser

