from datasets.kitti import *
from Archs_3D.build_retina3d_model import build_model
from Solver.Solver import Solver
import operator
import logging
import os
import sys
import copy
import datetime
from datasets.distributed_sampler import *
from datasets.dataset_utils import AspectRatioGroupedDataset
import numpy as np
import argparse
from utils.check_point import DetectronCheckpointer
from utils.metric_logger import MetricLogger
import time
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')

# def eval(model, b_images, b_labels, device=torch.device('cpu')):
#     model.eval()
#     with torch.no_grad():
#         pass
#
def train(model, solver, b_images, gt_labels):
    model.train()
    losses = model(copy.deepcopy(b_images), gt_labels=gt_labels, is_training=True)
    total_loss = solver.step(losses)
    return total_loss, losses




def run(model_name):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    logger = logging.getLogger("od.trainer")
    logger.info('Cuda available: ' + 'true' if torch.cuda.is_available() else 'false')
    model, backbone, CONFIGS = build_model(model_name)
    train_path = CONFIGS.TRAINING.LOGDIR
    device = torch.device(CONFIGS.DEVICE if torch.cuda.is_available() else "cpu")
    dataset = KITTI(CONFIGS, transform=TrainAugmentation(CONFIGS.INTENSOR_SIZE,
                                                         CONFIGS.DATASET.MEAN),
                    is_training=True)

    sampler = TrainingSampler(len(dataset))

    params = {'sampler': sampler,
              'batch_sampler': None,
              'collate_fn': operator.itemgetter(0),  # don't batch, but yield individual elements
              'num_workers': 8}

    generator = data.DataLoader(dataset, **params)
    generator = AspectRatioGroupedDataset(generator, CONFIGS.BATCH_SIZE)

    model.train()
    solver = Solver(model, CONFIGS)
    model = model.to(device)
    first_step = 0
    # load any previous weights
    model_path = os.path.abspath(train_path)
    checkpointer = DetectronCheckpointer(
        CONFIGS, model, solver, model_path, True, logger=logger
    )
    arguments = {}
    arguments['iteration'] = first_step
    weights = checkpointer.load(CONFIGS.TRAINING.CHECKPOINT_FILE)
    arguments.update(weights)


    max_steps = CONFIGS.SOLVER.MAX_ITER
    generator = iter(generator)
    meters = MetricLogger(delimiter=" ")
    first_step = arguments['iteration']
    start_training_time = time.time()
    end = time.time()
    for i in range(first_step, max_steps):
        iteration = i + 1
        arguments['iteration'] = iteration
        batch_data = next(generator)
        data_time = time.time() - end
        b_images = [d[0] for d in batch_data]
        gt_labels = [d[1] for d in batch_data]
        b_images = torch.stack(b_images, 0)
        b_images = b_images.to(device)

        total_loss,loss_dict = train(model, solver, b_images, gt_labels)

        batch_time = time.time() - end
        end = time.time()

        meters.update(loss=total_loss, **loss_dict)
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_steps - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 10 == 0 or iteration == max_steps:
            logger.info(meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.8f}",
                        "max men: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=solver.learnrate,
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 if torch.cuda.is_available() else 0
                ))
        # fixme: do we need checkpoint_period here
        if iteration % 1000 == 0 and iteration > 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_steps:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_steps)
        )
    )

def arg_parser():
    parser = argparse.ArgumentParser(description="Deepbox3D Training")
    parser.add_argument("--model-name", default="MOBI-V2-RETINA3D-FPN", help="specific model name")
    return parser

if __name__ == '__main__':
    args = arg_parser().parse_args()
    run(args.model_name)