
import torch
import os
from postprocess.postprocessing import *
from utils.check_point import DetectronCheckpointer
root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
from preprocess.data_preprocess import PredictionTransform
from utils.ParamList import ParamList

class Retina3DPredictor:
    """
    Create a simple end-to-end predictor with the given config.
    The predictor takes an BGR image, resizes it to the specified resolution,
    runs the model and produces a dict of predictions.

    This predictor takes care of model loading and input preprocessing for you.
    If you'd like to do anything more fancy, please refer to its source code
    as examples to build and use the model manually.

    Examples:

    .. code-block:: python

        pred = Retina3DPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, model, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model

        self.model = model
        self.model.eval()
        self.device = torch.device(self.cfg.DEVICE if torch.cuda.is_available() else "cpu")
        checkpoint_file = os.path.join(root_path, self.cfg.DETECTOR.CHECKPOINT)
        checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.TRAINING.LOGDIR)
        checkpointer.load(checkpoint_file, use_latest=False)
        self.input_format = cfg.INPUT.FORMAT
        self.transform = PredictionTransform(size=cfg.INTENSOR_SIZE, mean=cfg.DATASET.MEAN)
        assert self.input_format in ["RGB", "BGR"], self.input_format

        self.model.to(self.device)

    def __call__(self, original_image, targets):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict): the output of the model
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width, _ = original_image.shape
            pramas = ParamList((width, height), is_train=False)
            pramas._copy_extra_fields(targets)
            inputs, pramas = self.transform(original_image, pramas)
            inputs = inputs.to(self.device)
            inputs = inputs.unsqueeze(0)
            result = self.model(inputs)
            K = pramas.get_field('K').cpu().numpy()
            K[:, 3] = 0.
            r = result[0]
            pred_bboxes = r.get_field("bbox").cpu().numpy()
            pred_dims = r.get_field("dimension").cpu().numpy()
            pred_alphas = r.get_field("orientation").cpu().numpy()
            num = pred_bboxes.shape[0]
            if num <= 0:
                r.add_field('location', [])
                r.add_field('K', K)
                return r
            centers_x = pred_bboxes[:, 0::2].mean(axis=1)
            rays = np.arctan((centers_x - K[0, 2]) / K[0, 0])
            Rys = pred_alphas + rays

            pred_locations = []
            for i in range(num):
                loc, _ = calc_regressed_bbox_3d(pred_alphas[i],
                                                theta_ray=rays[i],
                                                dimension=pred_dims[i],
                                                bboxes=pred_bboxes[i],
                                                proj_matrix=K)
                pred_locations.append(loc)
            pred_locations = np.array(pred_locations, dtype=np.float32)
            r.add_field('location', pred_locations)
            r.add_field('K', K)
            r.add_field('Ry', Rys)
            return r
