import numpy as np
from .preprocess import preprocess
from scipy.optimize import curve_fit
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks

# 判断直线在y = 0 和 y = image.shape[0]之间是否相交
def is_intersect(y0y1,y2y3,image_shape):
    """
    Determines if two lines, defined by their endpoints, intersect within the bounds of an image.

    Args:
    - y0y1: tuple of two floats representing the y-coordinates of the endpoints of the first line
    - y2y3: tuple of two floats representing the y-coordinates of the endpoints of the second line

    Returns:
    - True if the two lines intersect within the bounds of the image, False otherwise
    """
    image_shape0 = image_shape[0]
    image_shape1 = image_shape[1]
    # 如果存在inf的情况，直接返回False
    if y0y1[0]==np.inf or y0y1[1]==np.inf or y2y3[0]==np.inf or y2y3[1]==np.inf:
        return False
    y0,y1=y0y1
    y2,y3=y2y3
    x_0 = 0
    x_1 = image_shape1
    y0y1_k=(y1-y0)/(x_1-x_0)
    y0y1_b=y0-y0y1_k*x_0
    y2y3_k=(y3-y2)/(x_1-x_0)
    y2y3_b=y2-y2y3_k*x_0
    if y0y1_k==y2y3_k:
        return False
    else:
        x=(y2y3_b-y0y1_b)/(y0y1_k-y2y3_k)
        y=y0y1_k*x+y0y1_b
        if y<0 or y>image_shape0:
            return False
        elif x<0 or x>image_shape1:
            return False
        else:
            return True


def f(x, A, B):
    return A*x + B

def _f(y, A, B):
    return (y-B)/A

def dfs(intersect_matrix, visited, cur, all_path):
    """
    Depth-first search algorithm to traverse a graph represented by an adjacency matrix.

    Args:
    intersect_matrix (numpy.ndarray): Adjacency matrix representing the graph.
    visited (list): List to keep track of visited nodes.
    cur (int): Current node being visited.
    all_path (list): List to store the path traversed during the DFS.

    Returns:
    list: List of nodes visited during the DFS.
    """
    visited[cur] = 1
    for i in range(len(intersect_matrix)):
        if intersect_matrix[cur, i] == 1 and visited[i] == 0:
            all_path.append(i)
            dfs(intersect_matrix, visited, i, all_path)
    return all_path


def hough(data,freq=100,bandpass=[2,8],sl=[.1,1],resample=1, sigma=1.3, low_threshold=3, high_threshold=6,theta=np.linspace(np.pi/2/90*10/100,np.pi/2/90*10,99), fil='bandpass', S_L=True,beta=0,kernel=(3,3)):
    '''
    Detects lines in an image using the Hough transform algorithm.

    Parameters:
    data (numpy.ndarray): The input image data.
    resample (int): The resampling factor for the image. Default is 1.
    sigma (float): The standard deviation for the Gaussian filter applied to the image. Default is 1.3.
    low_threshold (float): The lower threshold value for the Canny edge detection algorithm. Default is 3.
    high_threshold (float): The higher threshold value for the Canny edge detection algorithm. Default is 6.
    theta (numpy.ndarray): The array of angles (in radians) at which to compute the Hough transform. Default is np.linspace(np.pi/2/90*10/100,np.pi/2/90*10,99).

    Returns:
    list: A list of arrays containing the x-coordinates of the detected lines.
    '''
    #sl = [int(sl[0]*freq),int(sl[1]*freq)]
    data_sl = preprocess(data,fil=fil,S_L=S_L,bandpass=bandpass,freq=freq,sl=sl,beta=beta,kernel=kernel)
    #data_raw = preprocess(data,fil=2,S_L=0)
    image = data_sl.astype(np.uint8)
    image = image[:,::resample] # 压缩
    image_can = canny(image**2, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
    h, theta, d = hough_line(image_can,theta=theta)
    all_y0y1=[]
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
        #ax[2].plot((0, image.shape[1]), (y0, y1), '-r', alpha=0.5)
        all_y0y1.append((y0,y1))
    # generate intersect matrix
    intersect_matrix=np.zeros((len(all_y0y1),len(all_y0y1)))
    for i in range(len(all_y0y1)):
        for j in range(len(all_y0y1)):
            if i==j:
                continue
            intersect_matrix[i,j]=is_intersect(all_y0y1[i],all_y0y1[j],image.shape)
    # search all path by dfs
    visited=np.zeros(len(all_y0y1))
    all_path=[]
    for i in range(len(all_y0y1)):
        if visited[i]==0:
            all_path.append([i]+dfs(intersect_matrix,visited,i,[]))
    x0,x1=0,image.shape[1]
    line_x = []
    for path_ in all_path:
        #x0,x1=all_y0y1[path_].mean()
        y0=0
        y1=0
        for i in path_:
            y0 += all_y0y1[i][0]
            y1 += all_y0y1[i][1]
        y0 /= len(path_)
        y1 /= len(path_)
        # 计算斜率
        k=2*(y1-y0)/(image.shape[1]/100)
        popt, pcov = curve_fit(f, [x0,x1], [y0,y1])
        y = np.arange(0, image.shape[0], 1)
        x = _f(y, *popt)
        line_x.append(x)
    return line_x

if __name__ == '__main__':
    data = np.random.randn(100, 1000)
    line_x = hough(data)
    print(line_x)