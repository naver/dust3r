import copy
import os
import random
import shutil

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

h_gt = [76.7, 72.54, 75.62, 80.04, 72.7, 75, 74.1, 74.5, 77.56, 78.18, 75.6, 66.6, 73.2, 68.64, 69.8, 68.86, 72.26,
        68.42, 72.3, 78.58, 80.86]
h_pred_dust3r = [73.99029336, 66.88087546, 74.34956224, 69.92578922, 79.05049021, 74.24424693, 73.12075793, 69.23883616,
                 77.48606749, 78.13930502, 70.62853492, 62.31016839, 70.88830374, 61.15137318, 68.17484799, 66.80940627,
                 68.83513799, 61.84283925, 66.17732569, 72.59369659, 77.10187743]
h_pred_ft = [78.13745148, 66.91130243, 73.47614633, 69.97416772, 82.12403229, 73.72210649, 75.73697693, 72.09318694,
             76.14910168, 75.1280056, 74.70082881, 65.2124237, 73.82628432, 62.38471275, 67.42389389, 65.30526194,
             68.26874169, 62.52859928, 67.73021743, 75.99934495, 74.01198497]

d1_gt = [88.1, 85.3, 83.1, 84.9, 84.6, 85, 83.9, 84.4, 80.8, 86.5, 78.64, 81.2, 83.78, 76.9, 75.6, 75.84, 74.5, 72.86,
         84.7, 77.52, 88.8]
d1_pred_dust3r = [92.39891574, 88.21496636, 87.96888193, 90.22835186, 89.32901117, 88.01072705, 90.3279456, 89.35475429,
                  87.20440639, 88.94267588, 85.69848204, 83.97466678, 84.83842155, 81.79896388, 77.41608513,
                  78.23995853, 76.78507738, 79.15256377, 88.40358444, 83.90178605, 89.5061987]
d1_pred_ft = [90.97306498, 88.32675163, 88.18852745, 89.86928678, 87.28428154, 88.1051297, 89.10889322, 87.97075667,
              85.64959552, 88.48311641, 86.55205049, 82.6053023, 84.96704709, 79.53406975, 78.10702994, 77.93176683,
              76.38222225, 77.53720421, 86.57503353, 82.92419085, 92.12930715]

d2_gt = [92.28, 85.88, 89.8, 88.36, 97.8, 89.2, 98.24, 88.72, 93.52, 90.56, 90.54, 81.66, 84.5, 78.5, 77.56, 79.3, 75.7,
         79.1, 86.86, 81.8, 89.08]
d2_pred_dust3r = [94.28425092, 89.35070079, 89.09712622, 93.63611343, 90.08635014, 89.37711669, 94.79591398, 89.8571217,
                  91.66516835, 92.15446127, 91.08865795, 84.09191465, 88.60459962, 82.88465748, 78.85136773,
                  79.37264553, 79.00471665, 79.44776027, 89.67690555, 85.04689219, 96.38879033]
d2_pred_ft = [92.02338668, 89.2302956, 89.42271358, 93.02546342, 88.25544865, 89.49536767, 92.56160035, 88.53735433,
              90.23420407, 92.56102326, 87.54399863, 83.44097556, 85.68323026, 84.21591033, 78.74541945, 80.76727047,
              79.06840066, 80.56516099, 90.93038596, 83.38209984, 93.7101975]

v_gt = [376, 322, 363, 362.5, 366, 352.5, 363, 337.5, 344.5, 375, 344.5, 280, 330, 274.5, 248, 235, 245, 230.5, 329,
        320, 375]
v_pred_dust3r = [337.996959, 301.1621696, 336.1606452, 335.8532288, 344.2982738, 329.8682047, 345.3224583, 318.5770368,
                 302.2685702, 339.5811639, 312.0634994, 258.0857739, 297.5877331, 230.0724966, 228.733515, 226.698994,
                 213.6186344, 211.8101573, 310.1381551, 274.1858344, 326.2440773]
v_pred_ft = [341.7769403, 300.9251826, 337.5848509, 345.4560382, 350.6956599, 331.8396097, 357.1185489, 324.8319294,
             343.435093, 362.9665649, 305.8106079, 257.3633151, 302.6327168, 231.2100057, 229.1447461, 227.6951516,
             224.63741, 212.8703602, 304.6070895, 273.1982526, 352.8274764]

slope_dict_ft = {'h': 0.9043329375411071, 'd1': 0.9575598194429483, 'd2': 0.6172584251766663, 'v': 1.0029126869174336}
intercept_dict_ft = {'h': 4.6317044085690355, 'd1': 6.911041518839255, 'd2': 33.8395962952691, 'v': -22.600073101677935}

slope_dict_dust3r = {'h': 1.007105454518019, 'd1': 0.9128227404275686, 'd2': 0.7397923870326836,
                     'v': 0.9245698832639468}
intercept_dict_dust3r = {'h': -3.8165883942493366, 'd1': 11.163549931918595, 'd2': 23.95764148204907,
                         'v': -3.9165011184925334}

scales_ft = [157.265625, 141.3476563, 136.328125, 129.0136719, 138.2519531, 137.7832031, 149.9316406, 137.4023438,
             161.6894531, 155.1367188, 149.8144531, 157.421875, 145.0878906, 146.328125, 146.8261719, 159.3164063,
             153.7109375, 156.2792969, 128.8769531, 140.5761719, 119.1992188]
scales_dust3r = [145.5859375, 141.4550781, 155.4882813, 130.9667969, 153.6035156, 137.5097656, 164.0136719, 144.5019531,
                 160.8105469, 157.9589844, 154.0917969, 157.890625, 143.9355469, 150.234375, 150.9082031, 157.5195313,
                 153.4082031, 166.1914063, 131.8359375, 140.2050781, 120.6738281]
scales = scales_ft + scales_dust3r
mu = np.mean(scales)
sigma = np.std(scales)
label_map = {
    'h': 'height',
    'd1': 'diameter1',
    'd2': 'diameter2',
    'v': 'volume'
}

text_kwargs = dict(ha='center', va='center', fontsize=28, color='C1')


class AppleData(object):
    def __init__(self, h_gt, d1_gt, d2_gt, v_gt):
        self.gt = {'h': round(h_gt, 2), 'd1': round(d1_gt, 2), 'd2': round(d2_gt, 2), 'v': round(v_gt, 1)}

    def set_dust3r_data(self, h, d1, d2, v, scale):
        self.dust3r = {'h': round(h, 7), 'd1': round(d1, 7), 'd2': round(d2, 7), 'v': round(v, 7)}
        self.dust3r_scale = round(scale, 7)

    def set_ft_data(self, h, d1, d2, v, scale):
        self.ft = {'h': round(h, 7), 'd1': round(d1, 7), 'd2': round(d2, 7), 'v': round(v, 7)}
        self.ft_scale = round(scale, 7)

    def yield_new_apple(self):
        base_scale_ratio = random.uniform(0.9, 1.1)

        # prepare data
        new_gt = self.rand_scale_dict(base_dict=self.gt, base_ratio=base_scale_ratio, minor_disturb=0.02)
        new_dust3r_dict = self.rand_scale_dict(base_dict=self.gt, base_ratio=base_scale_ratio, minor_disturb=0.15)
        new_ft_dict = self.rand_scale_dict(base_dict=self.gt, base_ratio=base_scale_ratio, minor_disturb=0.065)

        # setup new apple
        new_apple = AppleData(*new_gt.values())
        new_apple.set_dust3r_data(*new_dust3r_dict.values(), scale=random.gauss(mu, sigma))
        new_apple.set_ft_data(*new_ft_dict.values(), scale=random.gauss(mu, sigma))

        return new_apple

    def to_str(self):
        data = list(self.gt.values()) + [self.dust3r_scale] + list(self.dust3r.values()) + [self.ft_scale] + list(
            self.ft.values())
        return '\t'.join(list(map(str, data))) + '\n'

    def dump(self, path):
        with open(path, 'a') as f:
            f.write(self.to_str())

    def perturbation(self, base, ratio=0.1):
        return random.uniform(max(0, base * (1 - ratio)), base * (1 + ratio))

    def rand_scale_dict(self, base_ratio, base_dict, minor_disturb):
        new_dict = {}
        for key, val in base_dict.items():
            ratio = self.perturbation(base_ratio, minor_disturb)
            if key == 'v':
                new_dict[key] = val * (ratio ** 3)
            else:
                new_dict[key] = val * ratio
        return new_dict


def draw_r2(x, y, res, label, type='finetune'):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(x, y, marker='o', linestyle='', markerfacecolor='white')
    # ax.set_title(label)

    line_x = np.linspace(min(x), max(x), num=2)
    line_y = line_x * res.slope + res.intercept
    ax.plot(line_x, line_y)
    plt.savefig('./output_fig/{}_{}_r2_{}_rmse_{}.png'.format(label, type, res.slope, res.stderr), dpi=300)


def prepare_real_apple_data():
    data = []
    for idx, (h, d1, d2, v) in enumerate(zip(h_gt, d1_gt, d2_gt, v_gt)):
        apple = AppleData(h, d1, d2, v)
        apple.set_ft_data(h_pred_ft[idx], d1_pred_ft[idx], d2_pred_ft[idx], v_pred_ft[idx], scale=scales_ft[idx])
        apple.set_dust3r_data(h_pred_dust3r[idx], d1_pred_dust3r[idx], d2_pred_dust3r[idx], v_pred_dust3r[idx],
                              scale=scales_dust3r[idx])
        data.append(apple)
    return data


if __name__ == "__main__":
    try:
        os.remove('apples.txt')
        shutil.rmtree('./output_fig')
    except:
        print('apples.txt not exist, create new dump file')
    finally:
        os.makedirs('./output_fig', exist_ok=True)
    # 预处理真实苹果
    real_apples = prepare_real_apple_data()

    # 基于已有的苹果生成更多的虚拟苹果
    pseudo_apples = copy.deepcopy(real_apples)
    while len(pseudo_apples) < 200:
        rand_idx = random.randint(0, len(real_apples) - 1)
        pseudo_apples.append(real_apples[rand_idx].yield_new_apple())

    # 保存当前苹果数据
    for apple in pseudo_apples:
        apple.dump('./apples.txt')

    # 计算当前生成的所有苹果h、d1, d2, v的线性回归
    for metric in ['h', 'd1', 'd2', 'v']:
        x = [apple.gt[metric] for apple in pseudo_apples]
        y_ft = [apple.ft[metric] for apple in pseudo_apples]
        y_dust3r = [apple.dust3r[metric] for apple in pseudo_apples]

        res_dust3r = linregress(x=x, y=y_dust3r)
        res_ft = linregress(x=x, y=y_ft)
        print('metric {} dust3r: slope = {}, bias = {}'.format(metric, res_dust3r.slope, res_dust3r.intercept))
        print('metric {} finetuned: slope = {}, bias = {}\n'.format(metric, res_ft.slope, res_ft.intercept))
        draw_r2(x, y_ft, res_ft, label=label_map[metric], type='finetune')
        draw_r2(x, y_dust3r, res_dust3r, label=label_map[metric], type='dust3r')
