from datasets.KittiDatasets import *
from Archs_2D.build_model import build_model
from Solver.Solver import Solver
import operator
from datasets.dataset_utils import AspectRatioGroupedDataset
import logging
import os
import sys
import copy
from datetime import datetime
from datasets.distributed_sampler import *
import matplotlib.pyplot as plt
import numpy as np
from postprocess.postprocessing import detector_postprocess
from postprocess.evaluating import eval_tp_fp_fn
import argparse

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def eval(model, b_images, b_labels, device=torch.device('cpu')):
    model.eval()
    with torch.no_grad():
        predictions = model(copy.deepcopy(b_images))
        processed_predictions = []
        for predictions_per_image in predictions:
            r = detector_postprocess(predictions_per_image, 1, 1)
            processed_predictions.append(r)
        return eval_tp_fp_fn(processed_predictions, copy.deepcopy(b_labels), device=device)

def train(model, solver, b_images, gt_classes, gt_bboxes):
    model.train()
    losses = model(copy.deepcopy(b_images), gt_classes, gt_bboxes, True)
    total_loss = solver.step(losses)
    return total_loss


def run(model_name):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    logging.info('Cuda available: ', torch.cuda.is_available())
    model, backbone, CONFIGS = build_model(model_name)
    dataset_path = os.path.join(CONFIGS.DATASET.PATH, 'training')
    train_path = CONFIGS.TRAINING.LOGDIR
    device = torch.device(CONFIGS.DEVICE if torch.cuda.is_available() else "cpu")
    dataset = KittiDatasets(dataset_path,
                            transform=TrainAugmentation(CONFIGS.INTENSOR_SHAPE[0], CONFIGS.DATASET.MEAN))

    sampler_name = CONFIGS.DATALOADER.SAMPLER_TRAIN
    logging.info("Using training sampler {}".format(sampler_name))
    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    params = {'sampler': sampler,
              'batch_sampler': None,
              'collate_fn':operator.itemgetter(0),  # don't batch, but yield individual elements
              'num_workers': 6}

    generator = data.DataLoader(dataset, **params)
    generator = AspectRatioGroupedDataset(generator, CONFIGS.BATCH_SIZE)

    bbox_coder = model.bbox_coder
    model.train()
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
            model.load_state_dict(checkpoint['model'])
            if CONFIGS.TRAINING.CHECKPOINT_MODE == 'RESUME':
                solver.load_state_dict(checkpoint['solver'], device)
                first_step = checkpoint['step']
                total_loss_debug_steps = checkpoint['total_loss_debug_steps']
                total_time_debug_steps = checkpoint['total_time_debug_steps']
                logging.info('Found previous checkpoint: %s at step %s'%(latest_model, first_step))
                logging.info('Resuming training....')
            else:
                logging.info('Starting training....')
    model = model.to(device)
    debug_steps = 10.
    max_steps = CONFIGS.SOLVER.MAX_ITER
    generator = iter(generator)
    #========init plot figure==========#
    plt.ion()
    plt.figure(1)
    x = []
    y1 = []
    #========init evaluating===========#
    t_tp = 0
    t_fp = 0
    t_fn = 0
    contents = []
    for i in range(first_step, max_steps):
        batch_data = next(generator)
        b_images = [d['image'] for d in batch_data]
        b_labels = [d['labels'] for d in batch_data]
        b_images = torch.stack(b_images, 0)
        b_images = b_images.to(device)
        gt_classes, gt_bboxes = bbox_coder.encode(b_labels, device)
        t1 = datetime.now()
        total_loss = train(model, solver, b_images, gt_classes, gt_bboxes)
        t2 = datetime.now()
        total_loss_debug_steps += total_loss.item()
        total_time_debug_steps += (float((t2 - t1).microseconds)/1e6)
        step = int(i+1)
        if step % int(debug_steps) == 0:
            average_loss = total_loss_debug_steps/debug_steps
            average_time = total_time_debug_steps/debug_steps
            logging.info(
                f"Step/Total_Steps: {step}/{max_steps}, " +
                f"Learning Rate: {solver.learnrate}, " +
                f"Average Loss: {average_loss:.4f}, " +
                f"Average time(s)/Batch: {average_time:.4f}"
            )
            total_loss_debug_steps = 0.
            total_time_debug_steps = 0.
            tp, fp, fn = eval(model, b_images, b_labels, device)
            t_tp += tp
            t_fp += fp
            t_fn += fn
        #if step % int(50) == 0:
        #    x.append(step)
        #    y1.append((average_loss))
        #    plt.plot(x, y1, c='r', ls='-', marker='o', mec='b', mfc='w')
        # save after every 1000 steps
        if (step and step % 1000 == 0) or (step == max_steps):
            name = os.path.join(model_path, 'model_%s.pkl' % step)
            logging.info("====================")
            logging.info("Done with step {}!".format(step))
            logging.info("Saving weights as {} ...".format(name))
            torch.save({
                'step': step,
                'total_loss_debug_steps': total_loss_debug_steps,
                'total_time_debug_steps':total_time_debug_steps,
                'model': model.state_dict(),
                'solver': solver.state_dict()
            }, name)
            logging.info("====================")
            precision = float(t_tp) / max(1, (t_tp + t_fp))
            recall = float(t_tp) / max(1, (t_tp + t_fn))
            f1_score = 2 * precision * recall / (precision + recall)
            logging.info(
                f"Batch_F1score: {f1_score}, " +
                f"Batch_Precision: {precision}, " +
                f"Batch_Recall: {recall}, "
            )
            contents.append(str(step) + f" {f1_score}" + f" {precision}" + f" {recall}" + "\n")
            t_tp = 0
            t_fp = 0
            t_fn = 0
            if step == max_steps:
                record = os.path.join(model_path, 'records.txt')
                with open(record, 'w') as f:
                    f.writelines(contents)
                logging.info('Finished Trainning!')



def arg_parser():
    parser = argparse.ArgumentParser(description="Retina Training")
    parser.add_argument("--model-name", default="MOBI-V2-RETINA-FPN", help="specific model name")
    return parser

if __name__ == '__main__':
    args = arg_parser().parse_args()
    run(args.model_name)


    pass

