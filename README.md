# SCUNet
Practical Blind Denoising via Swin-Conv-UNet and Data Synthesis, based on https://github.com/cszn/SCUNet.


## Dependencies
- [NumPy](https://numpy.org/install)
- [PyTorch](https://pytorch.org/get-started) 1.13
- [VapourSynth](http://www.vapoursynth.com/) R55+


## Installation
```
pip install -U vsscunet
python -m vsscunet
```


## Usage
```python
from vsscunet import scunet

ret = scunet(clip)
```

See `__init__.py` for the description of the parameters.
