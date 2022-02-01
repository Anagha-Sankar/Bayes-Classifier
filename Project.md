# BAYES CLASSIFIER



### Modules Used

1. **Numpy** - NumPy is a Python library used for working with arrays.
2. **Opencv** - OpenCV-Python makes use of Numpy, which is a highly optimized library for numerical operations. All the OpenCV array structures are converted to and from Numpy arrays.
3. **math** - This module provides access to the mathematical functions defined by the C standard.
4. **PIL (Pillow)** - PIL adds image processing capabilities to your Python interpreter. 

### Mean Calculation

The function calculates the mean of each column. The input to this function is an mxn matrix, and outputs mean in array of size 1xn. The mxn input matrix contains the 4 pixel values at the training coordinates.

```python
def mean_0(X):
    # Calculates mean of each column
    m,n = X.shape
    sums = np.full([1,n],0)
    for row in X:
        sums += row
    return sums/m
```

### Covariance Calculation

Covariance matrix is calculated as its needed for calculating the mahalanobis distance.

```python
def calcCov(x, y):
    mean_x, mean_y = x.mean(), y.mean()
    n = len(x)
    return sum((x - mean_x) * (y - mean_y)) / n
def cov(data):
    rows, cols = data.shape
    cov_mat = np.zeros((cols, cols))
    for i in range(cols):
        for j in range(cols):
            cov_mat[i][j] = calcCov(data[:, i], data[:, j])
    return cov_mat
```

### Mahalanobis Distance

The Mahalanobis distance (MD) is the distance between two points in multivariate space.  It can be used to determine whether a sample is an outlier, whether a process is in control or whether a sample is a member of a group or not.

```python
def mahalanobis_distance(p1,p2,X): #p1 is model, p2 is the test point
    # X is inverse cov matrix
    distance = np.dot(np.dot(np.subtract(p2,p1),np.array(X)),np.subtract(p2,p1).T)
    return distance
```

### Dataset

There are 4 images of different bands (R,G,B,I) given. The image is a satellite image of a place which has a river crossing through that area. The aim of this project is to classify the pixels to River_Class and NonRiver_Class. All 4 images are of size (512,512).

##### Loading images

The name of 4 images are stored in a list; and images are appended to a new array one after the other using a loop.

```python
# Load 4 band images of 512x512 
imglist = 'band1.jpg', 'band2.jpg', 'band3.jpg','band4.jpg'
band_img = []
# Load in the images
for img in imglist:
    band_img.append(cv2.imread(img,0))
```

band_image is of the dimension (4,512,512)

##### Training Coordinates

For training, 50 points of River_Class and 100 points of NonRiver_Class are identified. Both stored in separate files are loaded.

```python
# Loads train 50 coordinates of River Class
df1 = np.genfromtxt('River', delimiter=',',dtype=int)
# Loads train 100 coordinates of Non-River Class
df2 = np.genfromtxt('NonRiver',  delimiter=',',dtype=int)
```

##### Training Data

The color value at the pixel locations from the previously loaded coordinate files are identified and stored in a list. Its done for both the classes.

```python
def get_dataset(img,XY):
    X_train = []
    for band in img:
        ls=[band[x[1]][x[0]] for x in XY]
        X_train.append(ls)
    return np.array(X_train).T
```

This function return (n,4) array, where n is the number of coordinates. (River-50, NonRiver-100).

Thus, river_train is of shape (50,4) and nonriver_train is of shape (100,4).

Next, the mean for both river_train and nonriver_train are calculated using the above explained function. Both river_mean and nonriver_mean is of the shape (1,4).

Next step is to calculate the covariance of river_train and nonriver_train. Its stored in river_cov and nonriver_cov, each of shape (4,4).

For ease of understanding the dimensions, the test_data is taken as the transpose of the band_img, i.e. the whole 4 images. So test_data will be of the shape (512,512,4)

### Bayes Classification

For each pixel, i.e.  512*512 pixels, distance of it from centroid of river_mean and nonriver_mean are calculated using the mahalanobis distance metric.

Using these distances river_class and nonriver_class, the density functions p1 and p2 are calculated. 

Then, using Bayes Decision Rule, the pixel is classified to River_Class or NonRiver_Class.

###### Baye's Decision Rule

$$
Class_1\hspace{2mm}if\hspace{2mm}(P1 * p1) >= (P2 * p2) \\else \hspace{2mm} Class_2
$$

###### Density function Class C

$$
\large P(x_i|y)=\frac{1}{\sqrt{|Cov_C|}} exp({-0.5*C})
$$

**np.linalg.det** - its a linear algebra function from numpy module. Its used to find the determinant of a matrix (here, determinant of covariance matrix).

```python
def bayes(P1, P2, mean_river, mean_nonriver, river_cov, nonriver_cov, test_data):
    X = np.linalg.inv(river_cov)
    Y = np.linalg.inv(nonriver_cov)
    m,n,p = test_data.shape
    Out_image = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            river_class =  mahalanobis_distance(mean_river,test_data[i][j],X)
            nonriver_class =  mahalanobis_distance(mean_nonriver,test_data[i][j],Y)
            p1 = (1/(np.linalg.det(river_cov)**0.5))*math.exp(-0.5*river_class)
            p2 = (1/(np.linalg.det(nonriver_cov)**0.5))*math.exp(-0.5*nonriver_class)
            
            if(P1*p1 >= P2*p2):
                Out_image[i][j] = 255
            else:
                Out_image[i][j] = 0
    return Out_image.T  
```

The Bayes classification step is done for the 3 cases mentioned below.

**Case 1 :**

River class (Prior Prob: ) = 0.3 , Nonriver class(Prior Prob) = 0.7

**Case 2 :**

River class (Prior Prob: ) = 0.7 , Nonriver class(Prior Prob) = 0.3

**Case 3 :**

River class (Prior Prob: ) = 0.5 , Nonriver class(Prior Prob) = 0.5

```python
P1 = 0.3
P2 = 0.7
Out_image = bayes(P1,P2, river_mean, nonriver_mean, river_cov, nonriver_cov, test_data)
im1 = Image.fromarray(np.uint8(Out_image))
im1.save("Output_Images/river03non07.jpeg")
display(im1)
```

The numpy array returned by the bayes() is converted to image using the PIL module.

The output images are stored to the folder 'Output_Images'.