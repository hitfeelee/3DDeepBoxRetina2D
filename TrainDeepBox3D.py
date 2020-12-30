from datasets.KittiDataset3D import *
from Archs_3D.build_model import build_model
from Solver.Solver import Solver
import operator
import logging
import os
import sys
import copy
from datetime import datetime
from datasets.distributed_sampler import *
from datasets.dataset_utils import AspectRatioGroupedDataset
import numpy as np
import argparse
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# def eval(model, b_images, b_labels, device=torch.device('cpu')):
#     model.eval()
#     with torch.no_grad():
#         pass
#
def train(model, solver, b_images, gt_orient, gt_conf, gt_dim):
    model.train()
    losses = model(copy.deepcopy(b_images),
                   gt_orient=gt_orient, gt_conf=gt_conf,  gt_dim=gt_dim, is_training=True)
    total_loss = solver.step(losses)
    return total_loss




def run(model_name):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    logging.info('Cuda available: ', 'True' if torch.cuda.is_available() else 'False')
    model, backbone, CONFIGS = build_model(model_name)
    dataset_path = CONFIGS.DATASET.PATH
    train_path = CONFIGS.TRAINING.LOGDIR
    device = torch.device(CONFIGS.DEVICE if torch.cuda.is_available() else "cpu")
    dataset = KittiDataset3D(dataset_path,
                             transform=TrainAugmentation(CONFIGS.INTENSOR_SIZE))
    num = len(dataset)
    sampler = TrainingSampler(len(dataset))

    params = {'sampler': sampler,
              'batch_sampler': None,
              'collate_fn': operator.itemgetter(0),  # don't batch, but yield individual elements
              'num_workers': 8}

    generator = data.DataLoader(dataset, **params)
    generator = AspectRatioGroupedDataset(generator, CONFIGS.BATCH_SIZE)

    model.train()
    model = model.to(device)
    solver = Solver(model, CONFIGS)
    first_step = 0
    total_loss_debug_steps = 0.
    total_time_debug_steps = 0.
    # load any previous weights
    model_path = os.path.abspath(train_path)
    if CONFIGS.TRAINING.CHECKPOINT_MODE == 'PRETRAINED':
        model.init_from_base_net(CONFIGS.TRAINING.CHECKPOINT_FILE)
    else:
        latest_model = None
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
            model.init()
            logging.info('Starting training....')
        else:
            try:
                model_list = [[int((x.split('.')[0]).split('_')[-1]), x] for x in os.listdir(model_path)
                              if x.endswith('.pkl')]
                model_list = np.array(model_list)
                index = model_list[:, 0].astype(np.int)
                latest_model = model_list[np.argmax(index, axis=0), 1]
            except:
                logging.info('Not found previous checkpoint!')
                model.init()
                logging.info('Starting training....')
        if latest_model is not None:
            checkpoint = torch.load(os.path.join(model_path, latest_model), map_location=torch.device(device))
            model.load_state_dict(checkpoint['model_state_dict'])
            if CONFIGS.TRAINING.CHECKPOINT_MODE == 'RESUME':
                solver.load_state_dict(checkpoint['solver_state_dict'])
                first_step = checkpoint['step']
                total_loss_debug_steps = checkpoint['total_loss_debug_steps']
                total_time_debug_steps = checkpoint['total_time_debug_steps']
                logging.info('Found previous checkpoint: %s at step %s' % (latest_model, first_step))
                logging.info('Resuming training....')
            else:
                logging.info('Starting training....')
                pass

    debug_steps = 10.
    max_steps = CONFIGS.SOLVER.MAX_ITER
    generator = iter(generator)
    binCoder = model.binCoder
    for i in range(first_step, max_steps):
        batch_data = next(generator)
        b_images = [d[0] for d in batch_data]
        b_gt_dim = [torch.from_numpy(d[1]['Dimensions'].astype(np.float32)) for d in batch_data]
        b_gt_alpha = [torch.tensor(d[1]['Alpha'], dtype=torch.float32) for d in batch_data]
        b_gt_class = [torch.tensor(d[1]['Class'], dtype=torch.long) for d in batch_data]
        b_images = torch.stack(b_images, 0)
        b_gt_dim = torch.stack(b_gt_dim, 0)
        b_gt_alpha = torch.stack(b_gt_alpha, 0)
        b_gt_class = torch.stack(b_gt_class, 0)
        b_images = b_images.to(device)
        b_gt_dim = b_gt_dim.to(device)
        b_gt_alpha = b_gt_alpha.to(device)
        b_gt_class = b_gt_class.to(device)
        b_gt_orient, b_gt_conf, b_gt_dim = binCoder.encode(b_gt_alpha, b_gt_dim, b_gt_class)
        t1 = datetime.now()
        total_loss = train(model, solver, b_images, b_gt_orient, b_gt_conf, b_gt_dim)
        t2 = datetime.now()
        total_loss_debug_steps += total_loss.item()
        total_time_debug_steps += (float((t2 - t1).microseconds) / 1e6)
        step = int(i + 1)
        if step % int(debug_steps) == 0:
            average_loss = total_loss_debug_steps / debug_steps
            average_time = total_time_debug_steps / debug_steps
            logging.info(
                f"Step/Total_Steps: {step}/{max_steps}, " +
                f"Learning Rate: {solver.learnrate}, " +
                f"Average Loss: {average_loss:.4f}, " +
                f"Average time(s)/Batch: {average_time:.4f}"
            )
            total_loss_debug_steps = 0.
            total_time_debug_steps = 0.

        if (step and step % 1000 == 0) or (step == max_steps):
            name = os.path.join(model_path, 'deepbox3d_mobilev2_%s.pkl' % step)
            logging.info("====================")
            logging.info("Done with step {}!".format(step))
            logging.info("Saving weights as {} ...".format(name))
            torch.save({
                'step': step,
                'total_loss_debug_steps': total_loss_debug_steps,
                'total_time_debug_steps':total_time_debug_steps,
                'model_state_dict': model.state_dict(),
                'solver_state_dict': solver.state_dict()
            }, name)
            logging.info("====================")
            if step == max_steps:
                logging.info('Finished Trainning!')

def arg_parser():
    parser = argparse.ArgumentParser(description="Deepbox3D Training")
    parser.add_argument("--model-name", default="MOBI-V2-DEEPBOX3D", help="specific model name")
    return parser

if __name__ == '__main__':
    args = arg_parser().parse_args()
    run(args.model_name)