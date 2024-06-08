import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

def discretize_map(gt: np.ndarray) -> np.ndarray:
    return (gt > 0).astype(int)


def generate_thresholds(step_size: float) -> np.ndarray:
    return np.arange(0, 1 + step_size, step_size)


def auc_shuffled(saliency_map: np.ndarray, gt: np.ndarray, other_map: np.ndarray, n_splits: int = 100,
                 step_size: float = 0.1, to_plot: bool = True):
    """
    :param saliency_map: Saliency map obtained by BMS method
    :param gt: binary matrix of ground truth
    :param other_map: binary fixation map (like gt) by taking the union of fixations from M other random images
    :param n_splits: number of random splits
    :param step_size: step size for sweeping through saliency map
    :param to_plot: displays ROC curve
    :return: shuffled auc score
    """

    gt = discretize_map(gt)
    other_map = discretize_map(other_map)

    num_fixations = np.sum(gt)

    x, y = np.where(other_map == 1)
    other_map_fixations = [j[0] * other_map.shape[0] + j[1] for j in zip(x, y)]
    ind = len(other_map_fixations)
    assert ind == np.sum(other_map), 'Incorrect data provided'

    num_pixels = gt.shape[0] * gt.shape[1]
    random_numbers = []
    for i in range(n_splits):
        temp = []
        for k in range(num_fixations):
            temp.append(np.random.randint(num_pixels))
        random_numbers.append(temp)

    aucs = []
    all_tp = []
    all_fp = []

    for i in random_numbers:
        r_sal_map = [saliency_map[k % saliency_map.shape[0] - 1, k // saliency_map.shape[0]] for k in i]
        r_sal_map = np.array(r_sal_map)
        thresholds = generate_thresholds(step_size)

        area = [(0.0, 0.0)]
        for thresh in thresholds:
            temp = np.zeros(saliency_map.shape)
            temp[saliency_map >= thresh] = 1.0
            num_overlap = np.where(np.add(temp, gt) == 2)[0].shape[0]
            tp = num_overlap / (num_fixations * 1.0)
            fp = len(np.where(r_sal_map > thresh)[0]) / num_fixations
            area.append((round(tp, 4), round(fp, 4)))

        area.append((1.0, 1.0))
        area.sort(key=lambda x: x[0])
        tp_list = [x[0] for x in area]
        fp_list = [x[1] for x in area]

        aucs.append(np.trapz(np.array(tp_list), np.array(fp_list)))
        all_tp.append(tp_list)
        all_fp.append(fp_list)

    if to_plot:
        plt.figure()
        for tp_list, fp_list in zip(all_tp, all_fp):
            plt.plot(fp_list, tp_list, color='grey', alpha=0.2)
        mean_fp = np.mean(all_fp, axis=0)
        mean_tp = np.mean(all_tp, axis=0)
        plt.plot(mean_fp, mean_tp, color='blue', lw=2, label='Mean ROC curve')
        plt.xlabel('FP Rate')
        plt.ylabel('TP Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

    return np.mean(aucs)

# Example usage
# saliencyMap = imread(r"img/MIT_test_set/Test/Output/i2087721279_fixMap.jpg")
# fixationMap = imread(r"img/MIT_test_set/Test/Output/i2087721279_fixMap.jpg")
# otherMap = imread(r"img/MIT_test_set/Test/Output/i2125418545_fixMap.jpg")

# average_auc = auc_shuffled(saliencyMap, fixationMap, otherMap, n_splits=100, step_size=0.1, to_plot=True)
# print("Average AUC after shuffling:", average_auc)
