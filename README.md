# Das event detection flow
A high-dimensional template matching framework based on PyTorch.

## Installation
```bash
pip install dasflow
```

## Usage
```python

```

### TMF Example
```python
import numpy as np
from dasflow import tmf
# generate a random template
tmp = np.random.rand(10, 100, 20)
# generate a random image
data = np.random.rand(100, 1000)
# calculate the cross-correlation between the template and the image
corr = tmf.tma(data,tmp, step=1,device='cpu',moves = [],is_sum=False,batch_size=-1,half=False,save_memory=False)
```

#### Parameters
- `tmp` (numpy.ndarray or torch.Tensor): The template to be matched.
- `data` (numpy.ndarray or torch.Tensor): The image to search for the template.
- `step` (int, optional): The step size of the convolution. Defaults to 1.
- `device` (str, optional): The device to perform the computation on. Defaults to 'cpu'.
- `moves` (list, optional): A list of moves to apply to the template before matching. Defaults to [].
- `batch_size` (int, optional): The batch size to use for the computation. Defaults to -1.
- `save_memory` (bool, optional): Whether to use half-precision floating point numbers to save memory. Defaults to False.

#### Returns
- `numpy.ndarray`: The cross-correlation between the template and the image.

### Hough Example
```python
from dasflow import hough
import numpy as np
data = np.random.randn(512, 512)
hough(data,freq=100,bandpass=[2,8],sl=[.1,1],resample=1, sigma=1.3, low_threshold=3, high_threshold=6,theta=np.linspace(np.pi/2/90*10/100,np.pi/2/90*10,99), fil='bandpass', S_L=True,beta=0,kernel=(3,3))
```

#### Parameters
- `data` (numpy.ndarray or torch.Tensor): The image to search for the template.
- `freq` (int, optional): The frequency of the template. Defaults to 100.
- `bandpass` (list, optional): The bandpass filter to apply to the image. Defaults to [2,8].
- `sl` (list, optional): The size of the template. Defaults to [10,20].
- `resample` (int, optional): The resample rate of the image. Defaults to 1.
- `sigma` (float, optional): The sigma of the Gaussian filter. Defaults to 1.3.
- `low_threshold` (float, optional): The low threshold of the Canny edge detector. Defaults to 3.
- `high_threshold` (float, optional): The high threshold of the Canny edge detector. Defaults to 6.
- `theta` (numpy.ndarray, optional): The theta of the Hough transform. Defaults to np.linspace(np.pi/2/90*10/100,np.pi/2/90*10,99).
- `fil` (str, optional): The filter to apply to the image. Defaults to 'bandpass'.
- `S_L` (bool, optional): Whether to apply the Laplacian filter to the image. Defaults to True.
- `beta` (float, optional): The beta of the Laplacian filter. Defaults to 0.
- `kernel` (tuple, optional): The kernel size of the Laplacian filter. Defaults to (3,3).

### ADE-mini Example
```python
from dasflow.model import Mini
model = Mini()
data = torch.randn(1, 1, 512, 512)
model(data)
```
#### Parameters
- `data` (torch.Tensor): The input data to the model.

## License

MIT License

Copyright (c) [2023] []

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## References

