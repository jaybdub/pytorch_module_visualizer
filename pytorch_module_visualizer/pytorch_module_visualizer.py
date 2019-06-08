import torchvision
import torch
from IPython.display import display
import ipywidgets
import numpy as np
import math
import io
import PIL.Image
import traitlets
from torchvision.transforms.functional import to_pil_image, to_tensor


def pil_to_jpeg(image):
    with io.BytesIO() as buffer:
        image.save(buffer, format='JPEG')
        return buffer.getvalue()
   

def make_grid(data, normalize=True):
    channels = data.shape[0]
    height = data.shape[1]
    width = data.shape[2]
    nrow = int(np.sqrt(channels))
    ncol = math.ceil(channels / float(nrow))

    grid_height = nrow * height
    grid_width = ncol * width
    grid = torch.zeros((grid_height, grid_width))
    for row in range(nrow):
        for col in range(ncol):
            channel = row * ncol + col
            if channel >= channels:
                break
            grid[row * height:(row + 1) * height, col * width:(col + 1) * width] = data[row * ncol + col]
    
    if normalize:
        grid -= grid.min()
        grid /= grid.max()
    return grid


class ModuleVisualizer2D(ipywidgets.VBox):
    
    module = traitlets.Instance(klass=torch.nn.Module)
    disabled = traitlets.Bool(default_value=False)
    
    def __init__(self, *args, **kwargs):
        self.hook = None
        super(ModuleVisualizer2D, self).__init__(*args, **kwargs)
        self.module_input_widget = ipywidgets.Image()
        self.module_output_widget = ipywidgets.Image()
        self.children = [
            ipywidgets.HBox([self.module_input_widget]),
            ipywidgets.HBox([self.module_output_widget])
        ]
    
    def __del__(self):
        if self.hook is not None:
            self.hook.remove()

    @traitlets.observe('module')
    def _on_module(self, change):
        # remove hook
        if self.hook is not None:
            self.hook.remove()
            
        if not self.disabled:
            # attach hook for new module
            self.hook = change['new'].register_forward_hook(self._visualize)
    
    @traitlets.observe('disabled')
    def _on_disabled(self, change):
        
        # remove hook
        if self.hook is not None:
            self.hook.remove()
        
        # re-attach hook if not disabled
        if not change['new']:
            self.hook = self.module.register_forward_hook(self._visualize)
            
    def _visualize(self, module, input, output):
        try:
            a = input[0][0]
            b = output[0]
            self.module_input_widget.value = pil_to_jpeg(to_pil_image(make_grid(a)))
            self.module_output_widget.value = pil_to_jpeg(to_pil_image(make_grid(b)))
        except:
            pass