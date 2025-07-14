import argparse
from experiment.run import basic_run
from experiment.twin_verification_run import twin_verification_run
from utils.setup_logging import get_logger
from utils.misc import set_seed,load_yaml,override_args_with_yaml
import time

logger = get_logger("Prompt_CAM")


def main():
    args = setup_parser().parse_args()

    if args.config:
        yaml_config = load_yaml(args.config)
        if yaml_config:
            args = override_args_with_yaml(args, yaml_config)

    # Handle the --no_stage1_training flag to skip Stage 1
    if hasattr(args, 'no_stage1_training') and args.no_stage1_training:
        args.stage1_training = False
    
    # If stage1_training is None and we have a resume checkpoint, default to False
    if args.stage1_training is None:
        args.stage1_training = not bool(args.resume_checkpoint)

    set_seed(args.random_seed)
    start = time.time()
    args.vis_attn= False
    
    # Route to appropriate training function
    if args.train_type == 'twin_verification' or (hasattr(args, 'data') and args.data.startswith('twin')):
        twin_verification_run(args)
    else:
        basic_run(args)
        
    end = time.time()
    logger.info(f'----------- Total Run time : {(end - start) / 60} mins-----------')


def setup_parser():
    parser = argparse.ArgumentParser(description='Prompt_CAM')

    ########################Pretrained Model#########################
    parser.add_argument('--pretrained_weights', type=str, default='vit_base_patch16_224_in21k',
                        choices=['vit_base_patch16_224_in21k', 'vit_base_mae', 'vit_base_patch14_dinov2','vit_base_patch16_dino',
                                 'vit_base_patch16_clip_224'],
                        help='pretrained weights name')
    parser.add_argument('--drop_path_rate', default=0.1,
                        type=float,
                        help='Drop Path Rate (default: %(default)s)')
    parser.add_argument('--model', type=str, default='dinov2', choices=['vit', 'dino', 'dinov2'],
                        help='pretrained model name')
    
    parser.add_argument('--train_type', type=str, default='vpt', choices=['vpt', 'prompt_cam', 'linear', 'twin_verification'],
                        help='pretrained model name')
                 
    ########################Optimizer Scheduler#########################
    parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam', 'adamw'],
                        help='Optimizer (default: %(default)s)')
    parser.add_argument('--lr', default=0.005,
                        type=float,
                        help='Learning rate (default: %(default)s)')
    parser.add_argument('--epoch', default=100,
                        type=int,
                        help='The number of total epochs used. (default: %(default)s)')
    parser.add_argument('--warmup_epoch', default=20,
                        type=int,
                        help='warnup epoch in scheduler. (default: %(default)s)')
    parser.add_argument('--lr_min', type=float, default=1e-5,
                        help='lr_min for scheduler (default: %(default)s)')
    parser.add_argument('--warmup_lr_init', type=float, default=1e-6,
                        help='warmup_lr_init for scheduler (default: %(default)s)')
    parser.add_argument('--batch_size', default=16,
                        type=int,
                        help='Batch size (default: %(default)s)')
    parser.add_argument('--test_batch_size', default=32,
                        type=int,
                        help='Test batch size (default: %(default)s)')
    parser.add_argument('--wd', type=float, default=0.001,
                        help='weight_decay (default: %(default)s)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum used in sgd (default: %(default)s)')
    parser.add_argument('--early_patience', type=int, default=101,
                        help='early stop patience (default: %(default)s)')

    ########################Data#########################
    parser.add_argument('--data', default="processed_vtab-dtd",
                        help='data name. (default: %(default)s)')
    parser.add_argument('--data_path', default="data_folder/vtab_processed",
                        help='Path to the dataset. (default: %(default)s)')
    parser.add_argument('--crop_size', default=224,
                        type=int,
                        help='Crop size of the input image (default: %(default)s)')
    parser.add_argument('--final_run', action='store_false',
                        help='If final_run is True, use train+val as train data else, use train only')
    parser.add_argument('--normalized', action='store_false',
                        help='If imagees are normalized using ImageNet mean and variance ')

    ########################VPT#########################
    parser.add_argument('--vpt_mode', type=str, default=None, choices=['deep', 'shallow'],
                        help='VPT mode, deep or shallow')
    parser.add_argument('--vpt_num', default=10, type=int,
                        help='Number of prompts (default: %(default)s)')
    parser.add_argument('--vpt_layer', default=None, type=int,
                        help='Number of layers to add prompt, start from the last layer (default: %(default)s)')
    parser.add_argument('--vpt_dropout', default=0.1, type=float,
                        help='VPT dropout rate for deep mode. (default: %(default)s)')


    ########################full#########################
    parser.add_argument('--full', action='store_true',
                        help='whether turn on full finetune')

    ########################Misc#########################
    parser.add_argument('--gpu_num', default=1,
                        type=int,
                        help='Number of GPU (default: %(default)s)')
    parser.add_argument('--debug', action='store_false',
                        help='Debug mode to show more information (default: %(default)s)')
    parser.add_argument('--random_seed', default=42,
                        type=int,
                        help='Random seed (default: %(default)s)')
    parser.add_argument('--eval_freq', default=10,
                        type=int,
                        help='eval frequency(epoch) testset (default: %(default)s)')
    parser.add_argument('--store_ckp', action='store_true',
                        help='whether store checkpoint')
    parser.add_argument('--final_acc_hp', action='store_false',
                        help='if true, use the best acc during all epochs as criteria to select HP, if false, use the acc at final epoch as criteria to select HP ')
    
    ######################## YAML Config #########################
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')

    ########################Twin Verification Specific#########################
    parser.add_argument('--stage1_training', action='store_true', default=None,
                        help='Enable Stage 1 training (identity classification)')
    parser.add_argument('--no_stage1_training', action='store_true',
                        help='Disable Stage 1 training (skip to Stage 2)')
    parser.add_argument('--stage1_epochs', default=30, type=int,
                        help='Number of epochs for Stage 1 training')
    parser.add_argument('--stage2_epochs', default=50, type=int,
                        help='Number of epochs for Stage 2 training')
    parser.add_argument('--num_persons', default=356, type=int,
                        help='Number of persons in the dataset')
    parser.add_argument('--positive_ratio', default=0.5, type=float,
                        help='Ratio of positive pairs in verification dataset')
    parser.add_argument('--hard_negative_ratio', default=0.7, type=float,
                        help='Ratio of hard negatives (twin pairs) among negative pairs')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision training')
    parser.add_argument('--tracking_mode', default='none', choices=['mlflow', 'wandb', 'none'],
                        help='Experiment tracking mode')
    parser.add_argument('--wandb_entity', default='', type=str,
                        help='WandB entity name')
    parser.add_argument('--save_freq', default=5, type=int,
                        help='Checkpoint saving frequency (epochs)')
    parser.add_argument('--data_dir', default='data', type=str,
                        help='Directory containing dataset files')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to checkpoint file to resume training from')
                 
    return parser


if __name__ == '__main__':
    main()
