'''
@Description: reference d2l_pytorch/utils.py
@Author: hiro.tong
@Date: 2020-05-19 11:18:14
@LastEditTime: 2020-05-19 15:29:58
'''

#  Copyright (c) 2020. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

from IPython import display
from matplotlib import pyplot as plt

import json


def use_svg_display():
    """[summary]
    use svg format to display plot in juypter
    """
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_rows * scale, num_cols * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes


def show_bbox(axes, bboxes, labels, colors=None):
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b'])

    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        print(bbox)
        rect = plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2], height=bbox[3], fill=False, edgecolor=color, linewidth=2)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
            va='center', ha='center', fontsize=6, color=text_color,
            bbox=dict(facecolor=color, lw=0))


