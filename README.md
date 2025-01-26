# SCUNet
Practical Blind Denoising via Swin-Conv-UNet and Data Synthesis, based on https://github.com/cszn/SCUNet.


## Dependencies
- [PyTorch](https://pytorch.org/get-started/) 2.6.0.dev or later
- [VapourSynth](http://www.vapoursynth.com/) R66 or later

`trt` requires additional packages:
- [TensorRT](https://developer.nvidia.com/tensorrt) 10.4.0 or later
- [Torch-TensorRT](https://pytorch.org/TensorRT/) 2.6.0.dev or later

To install the latest nightly build of PyTorch and Torch-TensorRT, run:
```
pip install -U packaging setuptools wheel
pip install --pre -U torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu126
pip install --no-deps --pre -U torch_tensorrt --index-url https://download.pytorch.org/whl/nightly/cu126
pip install -U tensorrt --extra-index-url https://pypi.nvidia.com
```


## Installation
```
pip install -U vsscunet
```

If you want to download all models at once, run `python -m vsscunet`. If you prefer to only download the model you
specified at first run, set `auto_download=True` in `scunet()`.


## Usage
```python
from vsscunet import scunet

ret = scunet(clip)
```

See `__init__.py` for the description of the parameters.
