import numpy as np
from scipy import signal
def data_filter(data, fre_min, fre_max, freq=100):
    """
    Filters the input data using a bandpass filter.
 
    Parameters:
    - data: The input data to be filtered.
    - fre_min: The minimum frequency of the bandpass filter.
    - fre_max: The maximum frequency of the bandpass filter.
    - freq: The sampling frequency of the input data. Default is 100.
 
    Returns:
    - The filtered data.
 
    """
    fmin = fre_min / (freq / 2)
    fmax = fre_max / (freq / 2)
    s1, s2 = signal.butter(2, [fmin, fmax], 'bandpass')
    data_ = signal.filtfilt(s1, s2, data).copy()
    return data_
def data_SL(a, nsta, nlta, pad=False):
    """
    Calculate the short-term to long-term average ratio of seismic data.

    Parameters:
    a (ndarray): The seismic data array.
    nsta (int): The number of samples for the short-term average.
    nlta (int): The number of samples for the long-term average.
    pad (bool, optional): Whether to pad the output array. Defaults to False.

    Returns:
    ndarray: The short-term to long-term average ratio of the seismic data.

    """
    sta = np.cumsum(a**2, axis=1)
    lta = sta.copy()
    sta[:, nsta:] = sta[:, nsta:] - sta[:, :-nsta]
    sta = sta / nsta
    lta[:, nlta:] = lta[:, nlta:] - lta[:, :-nlta]
    lta = lta / nlta
    if pad:
        sta[:, :nlta-1] = 0
        sta[:, -nsta:] = 0
        lta[lta < 1e-15] = 1e-15
        return sta / lta
    else:
        lta[lta < 1e-15] = 1e-15
        return (sta / lta)[:, nlta:-nsta]
    
def Gauss_filter(data,kernel=(3,3),beta=0):
    """
    Applies Gaussian filter to the input image.

    Args:
    - data: input image
    - kernel: kernel size of the Gaussian filter (default is (3,3))
    - beta: standard deviation of the Gaussian filter (default is 0)

    Returns:
    - Guassian: filtered image
    """
    Guassian = cv2.GaussianBlur(data,kernel,beta)
    return Guassian

def preprocess(data, fil='bandpass', S_L=True, bandpass=[2,8],freq=100, sl=[0.5,1],beta=0,kernel=(3,3)):
    """
    Preprocesses the input data by applying a filter and/or a line detection algorithm.

    Args:
        data (numpy.ndarray): The input data to be preprocessed.
        fil (int, optional): The type of filter to be applied. Defaults to 2.
        S_L (int, optional): Whether to apply a line detection algorithm. Defaults to 1.

    Returns:
        numpy.ndarray: The preprocessed data.
    """
    if fil == 'bandpass':
        data_f = data_filter(data, bandpass[0], bandpass[1],freq=freq)
    else:
        data_f = Gauss_filter(data,kernel=kernel,beta=beta)
    if S_L:
        data_f = data_SL(data_f, int(sl[0]*freq), int(sl[1]*freq))[:,100:-100]
    return data_f

    
def auto_split(data,hwin=512,wwin=512,overlap=0.5): # 建议在实际操作中，使用等间隔抽样/叠加法以压缩道数
    """
    Splits a 2D array into smaller overlapping windows.

    Parameters:
    - data: numpy.ndarray
        The input 2D array to be split.
    - hwin: int, optional
        The height of the window. Default is 512.
    - wwin: int, optional
        The width of the window. Default is 512.
    - overlap: float, optional
        The overlap ratio between windows. Default is 0.5.

    Returns:
    - data_split: numpy.ndarray
        The array of split windows.

    """
    a,b = data.shape
    if a < hwin:
        data = np.pad(data,((0,hwin-a),(0,0)),'constant')
    if b < wwin:
        data = np.pad(data,((0,0),(0,wwin-b)),'constant')
    h_num = int((a-hwin)/(hwin*(1-overlap))+1)
    w_num = int((b-wwin)/(wwin*(1-overlap))+1)
    data_split = np.zeros((h_num, w_num,hwin,wwin))
    centers = []
    for i in range(h_num):
        for j in range(w_num):
            data_split[i, j] = data[int(i*hwin*(1-overlap)):int(i*hwin*(1-overlap))+hwin,int(j*wwin*(1-overlap)):int(j*wwin*(1-overlap))+wwin]
    return data_split

if __name__ == '__main__':
    data = np.random.randn(1000, 1000)
    data_split = auto_split(data)
    print(data_split.shape)
    data = np.random.randn(1000, 1000)
    data = data_filter(data, 1, 50)
    data = data_SL(data, 100, 1000)
    print(data.shape)
    print(data)