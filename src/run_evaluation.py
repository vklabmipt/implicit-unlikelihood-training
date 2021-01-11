import subprocess
import argparse
import logging
import os
from pathlib import Path
import torch

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPONT_PATH = './checkpoint/gpt2/'
TOPP_PARAMETERS = [0.3, 0.9]
TOPK_PARAMETERS = [1, 3, 8]
SINGLETOKEN_TOPP = [(0.9, 2e-6), (1.0, 0.0)]
SINGLETOKEN_TOPK = [(1, 2e-5), (10, 8e-6)]


def main():
    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument('--path_to_script',
                        type=str,
                        default='run_gpt2.py',
                        help='script to run')
    parser.add_argument('--checkpoint_folder',
                        type=str)
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0')
    parser.add_argument('--algorithm',
                        type=str,
                        default='ul')
    parser.add_argument('--eval_mode', type=str, choices=['completion', 
                                                          'singletoken', 
                                                          'singletoken_sampling', 
                                                          'all'], default='all')
    parser.add_argument('--adress', type=str, default='tcp://127.0.0.1:4444')
    parser.add_argument('--model-name', type=str, default='gpt2')
    
    parser.add_argument('--alpha_entmax', action='store_true')
    parser.add_argument('--alpha', type=float, default=1.2)
    parser.add_argument('--laplas_eps', type=float, default=1e-6)
    parser.add_argument('--eval_split', type=str, default='valid', choices=['valid', 'test'])
    args = parser.parse_args()

    device = args.device
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))

    full_path_to_checkpoint = os.path.join(
        CHECKPONT_PATH, args.checkpoint_folder)
    if not Path(full_path_to_checkpoint).exists():
        raise ValueError(f'path {full_path_to_checkpoint} doesn\'t exists')
    elif not Path(full_path_to_checkpoint).is_dir():
        raise ValueError(f'{full_path_to_checkpoint} is not a directory')

    if not Path(args.path_to_script).exists():
        raise ValueError(f'file {args.path_to_script} doesn\'t exists')
    if not args.path_to_script.endswith('.py'):
        raise ValueError(
            f'file {args.path_to_script} should be a python script')
        
    if args.alpha_entmax is True:
        subprocess.call(['./evaluate_singletoken_alpha_entmax.sh',
                                 args.path_to_script,
                                 str(full_path_to_checkpoint),
                                 str(device),
                                 args.algorithm,
                                 str(args.adress),
                                 str(args.model_name),
                                 str(args.alpha),
                                 str(args.laplas_eps),
                        str(args.eval_split)])
        
        subprocess.call(['./evaluate_completion_alpha_entmax.sh',
                                 args.path_to_script,
                                 str(full_path_to_checkpoint),
                                 str(device),
                                 args.algorithm,
                                 str(args.adress),
                                 str(args.model_name),
                                 str(args.alpha),
                                 str(args.laplas_eps),
                        str(args.eval_split)])
        
    else:
        if args.eval_mode == 'singletoken_argmax' or args.eval_mode == 'all':
            subprocess.call(['./evaluate_singletoken_argmax.sh',
                             args.path_to_script,
                             str(full_path_to_checkpoint),
                             str(device),
                             str(args.algorithm),
                             str(args.adress),
                             str(args.model_name),
                            str(args.eval_split)])

        if args.eval_mode == 'singletoken' or args.eval_mode == 'all':
            for (p, laplas_eps) in SINGLETOKEN_TOPP:
                subprocess.call(['./evaluate_singletoken_softmax.sh',
                                 args.path_to_script,
                                 str(full_path_to_checkpoint),
                                 str(device),
                                 str(args.algorithm),
                                 str(args.adress),
                                 str(args.model_name),
                                 '0',
                                 str(p),
                                 str(laplas_eps),
                                str(args.eval_split)])

            for (k, laplas_eps) in SINGLETOKEN_TOPK:
                subprocess.call(['./evaluate_singletoken_softmax.sh',
                                 args.path_to_script,
                                 str(full_path_to_checkpoint),
                                 str(device),
                                 str(args.algorithm),
                                 str(args.adress),
                                 str(args.model_name),
                                 str(k),
                                 '0.0',
                                 str(laplas_eps),
                                str(args.eval_split)])

        if args.eval_mode == 'completion' or args.eval_mode == 'all':
            for k in TOPK_PARAMETERS:
                subprocess.call(['./evaluate_completion_softmax.sh',
                                 args.path_to_script,
                                 str(full_path_to_checkpoint),
                                 str(device),
                                 str(k),
                                 '0.',
                                 args.algorithm,
                                 str(args.adress),
                                 str(args.model_name),
                                str(args.eval_split)])
            for p in TOPP_PARAMETERS:
                subprocess.call(['./evaluate_completion_softmax.sh',
                                 args.path_to_script,
                                 str(full_path_to_checkpoint),
                                 str(device),
                                 '0',
                                 str(p),
                                 args.algorithm,
                                 str(args.adress),
                                 str(args.model_name),
                                str(args.eval_split)])


if __name__ == '__main__':
    main()
