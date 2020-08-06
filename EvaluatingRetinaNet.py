
from Archs_2D.build_model import build_model
import logging
from Archs_2D.DefaultEvaluator import DefaultEvaluator
import sys
import os
import torch
import argparse

def arg_parser():
    parser = argparse.ArgumentParser(description="Retina Training")
    parser.add_argument("--model-name", default="MOBI-V2-RETINA-FPN", help="specific model name")
    return parser

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    logging.info('Cuda available: ', torch.cuda.is_available())
    args = arg_parser().parse_args()
    model, backbone, cfg = build_model(args.model_name)
    evaluator = DefaultEvaluator(model, cfg)
    f1_score,precision, recall = evaluator.runing()
    sys.stdout.write(f'\rF1_score:{f1_score}, Precision:{precision}, Recall:{recall} in kitti trainning dataset\n')

