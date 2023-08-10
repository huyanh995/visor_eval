import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from visor.visor import VISOR
from visor.metrics import db_eval_boundary, db_eval_iou
from visor import utils
from visor.results import Results
from scipy.optimize import linear_sum_assignment


class VISOREvaluation:
    def __init__(self, root):
        self.root = root
        self.dataset = VISOR(root=root)

    @staticmethod
    def compute_metrics(all_gt_masks,   # (T, n_objects, H, W)
                        all_res_masks,  # (T, n_objects, H, W)
                        all_void_masks,
                        metrics,
                        padding=False):

        # Padding objects
        if all_gt_masks.shape[0] < all_res_masks.shape[0]:
            sys.stdout.write("\nIn your PNG files there is an index higher than the number of objects in the sequence!")
            sys.exit()

        elif all_gt_masks.shape[0] > all_res_masks.shape[1]:
            zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
            all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)

        # Padding frames
        if all_gt_masks.shape[1] > all_res_masks.shape[1]:
            if padding:
                # Pad zero mask to the beginning of all_res_mask
                # Opposite to DAVIS where padding to the end
                zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
                all_res_masks = np.concatenate([zero_padding, all_res_masks], axis=0)
            else:
                # Truncate the first few empty gt masks
                all_gt_masks = all_gt_masks[:, all_gt_masks.shape[1] - all_res_masks.shape[1] : , ...]


        j_metrics_res, f_metrics_res = np.zeros(all_gt_masks.shape[:2]), np.zeros(all_gt_masks.shape[:2])
        for i in range(all_gt_masks.shape[0]):
            if 'J' in metrics:
                j_metrics_res[i, :] = db_eval_iou(all_gt_masks[i, ...], all_res_masks[i, ...], all_void_masks)
            if 'F' in metrics:
                f_metrics_res[i, :] = db_eval_boundary(all_gt_masks[i, ...], all_res_masks[i, ...], all_void_masks)

        return j_metrics_res, f_metrics_res


    def evaluate(self, res_path, metrics=('J', 'F'), padding=False):
        metrics = metrics if isinstance(metrics, tuple) or isinstance(metrics, list) else [metrics]
        if 'T' in metrics:
            raise ValueError('Temporal metric not supported!')
        if 'J' not in metrics and 'F' not in metrics:
            raise ValueError('Metric possible values are J for IoU or F for Boundary')

        metrics_res = {}
        if 'J' in metrics:
            metrics_res['J'] = {'M': [], 'R': [], 'D': [], 'M_per_object': {}} # Mean, Recall, Decay
        if 'F' in metrics:
            metrics_res['F'] = {'M': [], 'R': [], 'D': [], 'M_per_object': {}}

        results = Results(root_dir=res_path)
        for seq in tqdm(self.dataset.get_sequences(), total = len(self.dataset)):
            all_gt_masks, _, all_masks_id = self.dataset.get_all_masks(seq)
            all_res_masks = results.read_masks(seq, all_masks_id)
            j_metrics_res, f_metrics_res = self.compute_metrics(all_gt_masks, all_res_masks, None, metrics)

            for ii in range(all_gt_masks.shape[0]):
                seq_name = f'{seq}_{ii+1}'
                if 'J' in metrics:
                    [JM, JR, JD] = utils.db_statistics(j_metrics_res[ii])
                    metrics_res['J']["M"].append(JM)
                    metrics_res['J']["R"].append(JR)
                    metrics_res['J']["D"].append(JD)
                    metrics_res['J']["M_per_object"][seq_name] = JM
                if 'F' in metrics:
                    [FM, FR, FD] = utils.db_statistics(f_metrics_res[ii])
                    metrics_res['F']["M"].append(FM)
                    metrics_res['F']["R"].append(FR)
                    metrics_res['F']["D"].append(FD)
                    metrics_res['F']["M_per_object"][seq_name] = FM

        return metrics_res, results.uneven_length
