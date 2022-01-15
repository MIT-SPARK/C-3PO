import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from matplotlib.widgets import Slider
import animatplot as amp

'''
Run this script to use matplotlib to draw a plot with a horizontal slider that allows users to slide between two 
different cars. The animation can be slow depending on the machine performance.
'''
if __name__ == "__main__":
    frame_dir = "./output_frames"
    frame_paths = [os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".jpg")]
    frame_paths = sorted(frame_paths)
    frames = [image.imread(f) for f in frame_paths]

    fig = plt.figure(figsize=(12, 6), dpi=200)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(bottom=0, top=0.85, left=0, right=1)

    # create slider axes
    ax_suv_slider = fig.add_axes([0.4, 0.87, 0.4, 0.05])
    ax_suv_slider.spines['top'].set_visible(True)
    ax_suv_slider.spines['right'].set_visible(True)

    ax_sedan_slider = fig.add_axes([0.4, 0.8, 0.4, 0.05])
    ax_sedan_slider.spines['top'].set_visible(True)
    ax_sedan_slider.spines['right'].set_visible(True)

    # create sliders
    allowed_weights = np.linspace(0, 1, 100)
    val_setp = allowed_weights[1] - allowed_weights[0]
    suv_weight_slider = Slider(ax=ax_suv_slider, label="Audi Q7 Weight", valmin=0, valmax=1, valinit=0,
                               valstep=val_setp, facecolor='#cc7000', edgecolor=None)
    suv_weight_slider.label.set_size(20)
    suv_weight_slider.valtext.set_size(20)

    sedan_weight_slider = Slider(ax=ax_sedan_slider, label="Hyundai Sonata Weight", valmin=0, valmax=1, valinit=0,
                                 valstep=val_setp, facecolor='#cc7000', edgecolor=None)
    sedan_weight_slider.label.set_size(20)
    sedan_weight_slider.valtext.set_size(20)
    sedan_weight_slider.set_val(1)

    # frame data
    f_d = ax.imshow(frames[0])
    ax.axis("off")


    def suv_update(val):
        weight = suv_weight_slider.val
        sedan_weight_slider.set_val(1 - weight)
        i = np.where(allowed_weights == weight)
        i = int(i[0])
        f_d.set_data(frames[i])
        fig.canvas.draw_idle()


    suv_weight_slider.on_changed(suv_update)
