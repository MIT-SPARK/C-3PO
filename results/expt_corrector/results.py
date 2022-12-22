
import ipywidgets as widgets

import sys
sys.path.append("../../")
from c3po.datasets.shapenet import OBJECT_CATEGORIES as shapenet_objects

dd_object = widgets.Dropdown(
    options=shapenet_objects,
    value=shapenet_objects[0],
    description="Object"
)