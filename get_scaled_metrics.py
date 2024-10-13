import copy

import scipy
import numpy as np


def obj(scale, gt_metrics, pred_metrics):
    gt_copy, pred_copy = copy.deepcopy(gt_metrics), copy.deepcopy(pred_metrics)
    pred_copy[:-1] *= scale
    pred_copy[-1:] = pred_copy[-1:] * np.power(scale, 3) / 10000
    gt_copy[-1:] = gt_copy[-1:] / 10.0
    res = np.linalg.norm(gt_copy - pred_copy)
    return res


def get_scale(gt_metrics, pred_metrics):
    res = scipy.optimize.minimize(obj, x0=np.asarray(100), method='Nelder-Mead', tol=1e-2,
                                  args=(gt_metrics, pred_metrics))
    print('best scale : ', res.x)
    return res.x


def prepare_input(gt_metrics, pred_metrics):
    gt = np.asarray(gt_metrics)
    gt[1:3] = sorted(gt[1:3])
    pred = np.asarray(pred_metrics)
    pred[1:3] = sorted(pred[1:3])
    return gt, pred


def scale_pred(gt_metrics, pred_metrics):
    gt, pred = prepare_input(gt_metrics, pred_metrics)
    scale = get_scale(gt, pred)
    scaled_pred = pred * scale
    scaled_pred[-1] *= scale ** 2 / 1000
    print('gt = ', gt)
    print('pred = ', '\t'.join(list(map(str, pred))))
    print('scaled pred = ', '\t'.join(list(map(str, scaled_pred))))
    return scale, scaled_pred


if __name__ == "__main__":
    gt_metrics = [76.7, 88.1, 92.28, 376]
    pred_metrics = [0.677906481, 0.682311424, 0.703075581, 0.166760526]
    scale_pred(gt_metrics, pred_metrics)
