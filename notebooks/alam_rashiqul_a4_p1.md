# Part 3: Single-View Geometry

## Usage
This code snippet provides an overall code structure and some interactive plot interfaces for the *Single-View Geometry* section of Assignment 3. In [main function](#Main-function), we outline the required functionalities step by step. Some of the functions which involves interactive plots are already provided, but [the rest](#Your-implementation) are left for you to implement.

## Package installation
- In this code, we use `tkinter` package. Installation instruction can be found [here](https://anaconda.org/anaconda/tk).

# Common imports


```python
%matplotlib tk
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
```

# Provided functions


```python
def get_input_lines(im, min_lines=3):
    """
    Allows user to input line segments; computes centers and directions.
    Inputs:
        im: np.ndarray of shape (height, width, 3)
        min_lines: minimum number of lines required
    Returns:
        n: number of lines from input
        lines: np.ndarray of shape (3, n)
            where each column denotes the parameters of the line equation
        centers: np.ndarray of shape (3, n)
            where each column denotes the homogeneous coordinates of the centers
    """
    n = 0
    lines = np.zeros((3, 0))
    centers = np.zeros((3, 0))

    plt.figure()
    plt.imshow(im)
    plt.show()
    print('Set at least %d lines to compute vanishing point' % min_lines)
    while True:
        print('Click the two endpoints, use the right key to undo, and use the middle key to stop input')
        clicked = plt.ginput(2, timeout=0, show_clicks=True)
        if not clicked or len(clicked) < 2:
            if n < min_lines:
                print('Need at least %d lines, you have %d now' % (min_lines, n))
                continue
            else:
                # Stop getting lines if number of lines is enough
                break

        # Unpack user inputs and save as homogeneous coordinates
        pt1 = np.array([clicked[0][0], clicked[0][1], 1])
        pt2 = np.array([clicked[1][0], clicked[1][1], 1])
        # Get line equation using cross product
        # Line equation: line[0] * x + line[1] * y + line[2] = 0
        line = np.cross(pt1, pt2)
        lines = np.append(lines, line.reshape((3, 1)), axis=1)
        # Get center coordinate of the line segment
        center = (pt1 + pt2) / 2
        centers = np.append(centers, center.reshape((3, 1)), axis=1)

        # Plot line segment
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='b')

        n += 1

    return n, lines, centers
```


```python
def plot_lines_and_vp(im, lines, vp):
    """
    Plots user-input lines and the calculated vanishing point.
    Inputs:
        im: np.ndarray of shape (height, width, 3)
        lines: np.ndarray of shape (3, n)
            where each column denotes the parameters of the line equation
        vp: np.ndarray of shape (3, )
    """
    bx1 = min(1, vp[0] / vp[2]) - 10
    bx2 = max(im.shape[1], vp[0] / vp[2]) + 10
    by1 = min(1, vp[1] / vp[2]) - 10
    by2 = max(im.shape[0], vp[1] / vp[2]) + 10

    plt.figure()
    plt.imshow(im)
    for i in range(lines.shape[1]):
        if lines[0, i] < lines[1, i]:
            pt1 = np.cross(np.array([1, 0, -bx1]), lines[:, i])
            pt2 = np.cross(np.array([1, 0, -bx2]), lines[:, i])
        else:
            pt1 = np.cross(np.array([0, 1, -by1]), lines[:, i])
            pt2 = np.cross(np.array([0, 1, -by2]), lines[:, i])
        pt1 = pt1 / pt1[2]
        pt2 = pt2 / pt2[2]
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'g')

    plt.plot(vp[0] / vp[2], vp[1] / vp[2], 'ro')
    plt.show()
```


```python
def get_top_and_bottom_coordinates(im, obj):
    """
    For a specific object, prompts user to record the top coordinate and the bottom coordinate in the image.
    Inputs:
        im: np.ndarray of shape (height, width, 3)
        obj: string, object name
    Returns:
        coord: np.ndarray of shape (3, 2)
            where coord[:, 0] is the homogeneous coordinate of the top of the object and coord[:, 1] is the homogeneous
            coordinate of the bottom
    """
    plt.figure()
    plt.imshow(im)

    print('Click on the top coordinate of %s' % obj)
    clicked = plt.ginput(1, timeout=0, show_clicks=True)
    x1, y1 = clicked[0]
    # Uncomment this line to enable a vertical line to help align the two coordinates
    # plt.plot([x1, x1], [0, im.shape[0]], 'b')
    print('Click on the bottom coordinate of %s' % obj)
    clicked = plt.ginput(1, timeout=0, show_clicks=True)
    x2, y2 = clicked[0]

    plt.plot([x1, x2], [y1, y2], 'b')

    return np.array([[x1, x2], [y1, y2], [1, 1]])
```

# Your implementation


```python
def get_vanishing_point(lines):
    """
    Solves for the vanishing point using the user-input lines.
    Args:
        lines: np.ndarray of shape (3, n), each column is [a, b, c] for line equation ax + by + c = 0
    Returns:
        v: np.ndarray of shape (3,), the vanishing point in homogeneous coordinates
    """
    # Transpose so that each row is a line
    # Shape of A is (n, 3)
    A = lines.T

    # Perform SVD: A = U S V^T
    U, S, VT = np.linalg.svd(A)

    # The vanishing point is the last column of V (or last row of VT)
    v = VT[-1]

    # Return the normalized vanishing point to make the last coordinate 1
    return (v / v[-1])  
```


```python
def get_horizon_line(vpts):
    """
    Calculates the ground horizon line.
    Args:
        vpts: np.ndarray of shape (3, 3), each column is a vanishing point
    Returns:
        horizon_line: np.ndarray of shape (3,), line parameters [a, b, c] for ax + by + c = 0
    """
    # The horizon line passes through two vanishing points on the ground plane
    # Typically vpts[:, 0] and vpts[:, 1] are the horizontal vanishing points
    # and vpts[:, 2] is the vertical vanishing point
    
    # Compute the line through the first two vanishing points using cross product
    horizon_line = np.cross(vpts[:, 0], vpts[:, 1])
    
    return horizon_line
```


```python
def plot_horizon_line(im, horizon_line):
    """
    Plots the horizon line.
    Args:
        im: np.ndarray of shape (height, width, 3)
        horizon_line: np.ndarray of shape (3,), line parameters [a, b, c] for ax + by + c = 0
    """
    # Get image boundaries
    height, width = im.shape[:2]
    
    # Find two points on the horizon line by intersecting with image boundaries
    # Left edge: x = 0
    left_pt = np.cross(horizon_line, np.array([1, 0, 0]))  # Vertical line at x=0
    left_pt = left_pt / left_pt[2]  # Normalize
    
    # Right edge: x = width
    right_pt = np.cross(horizon_line, np.array([1, 0, -width]))  # Vertical line at x=width
    right_pt = right_pt / right_pt[2]  # Normalize
    
    # Plot the image and horizon line
    plt.figure()
    plt.imshow(im)
    plt.plot([left_pt[0], right_pt[0]], [left_pt[1], right_pt[1]], 'r-', linewidth=2, label='Horizon Line')
    plt.legend()
    plt.show()
```


```python
def get_camera_parameters():
    """
    Computes the camera parameters. Hint: The SymPy package is suitable for this.
    """
    # <YOUR IMPLEMENTATION>
    pass
```


```python
def get_rotation_matrix():
    """
    Computes the rotation matrix using the camera parameters.
    """
    # <YOUR IMPLEMENTATION>
    return R
```


```python
def estimate_height():
    """
    Estimates height for a specific object using the recorded coordinates. You might need to plot additional images here for
    your report.
    """
    # <YOUR IMPLEMENTATION>
    pass
```

# Main function


```python
im = np.asarray(Image.open('CSL.jpg'))

# Part 1
# Get vanishing points for each of the directions
num_vpts = 3
vpts = np.zeros((3, num_vpts))
for i in range(num_vpts):
    print('Getting vanishing point %d' % i)
    # Get at least three lines from user input
    n, lines, centers = get_input_lines(im)
    # <YOUR IMPLEMENTATION> Solve for vanishing point
    vpts[:, i] = get_vanishing_point(lines)
    print('Vanishing point %d: %s' % (i, vpts[:, i]))
    # Plot the lines and the vanishing point
    plot_lines_and_vp(im, lines, vpts[:, i])

# <YOUR IMPLEMENTATION> Get the ground horizon line
horizon_line = get_horizon_line(vpts)
# <YOUR IMPLEMENTATION> Plot the ground horizon line
plot_horizon_line(im, horizon_line)

# Part 2
# <YOUR IMPLEMENTATION> Solve for the camera parameters (f, u, v)
f, u, v = get_camera_parameters()

# Part 3
# <YOUR IMPLEMENTATION> Solve for the rotation matrix
R = get_rotation_matrix()

# Part 4
# Record image coordinates for each object and store in map
objects = ('person', 'CSL building', 'the spike statue', 'the lamp posts')
coords = dict()
for obj in objects:
    coords[obj] = get_top_and_bottom_coordinates(im, obj)

# <YOUR IMPLEMENTATION> Estimate heights
for obj in objects[1:]:
    print('Estimating height of %s' % obj)
    height = estimate_height()

```

    Getting vanishing point 0
    Set at least 3 lines to compute vanishing point
    Click the two endpoints, use the right key to undo, and use the middle key to stop input


    can't invoke "event" command: application has been destroyed
        while executing
    "event generate $w <<ThemeChanged>>"
        (procedure "ttk::ThemeChanged" line 6)
        invoked from within
    "ttk::ThemeChanged"


    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Vanishing point 0: [-449.40014694  220.87113534    1.        ]
    Getting vanishing point 1
    Set at least 3 lines to compute vanishing point
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Vanishing point 1: [ 1.14946479e+03 -1.40231689e+02  1.00000000e+00]
    Getting vanishing point 2
    Set at least 3 lines to compute vanishing point
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Click the two endpoints, use the right key to undo, and use the middle key to stop input
    Vanishing point 2: [3.09229246e+02 1.38068469e+04 1.00000000e+00]


    /tmp/ipykernel_1822/2627096133.py:24: RuntimeWarning: divide by zero encountered in divide
      pt1 = pt1 / pt1[2]
    /tmp/ipykernel_1822/2627096133.py:24: RuntimeWarning: invalid value encountered in divide
      pt1 = pt1 / pt1[2]
    /tmp/ipykernel_1822/2627096133.py:25: RuntimeWarning: divide by zero encountered in divide
      pt2 = pt2 / pt2[2]
    /tmp/ipykernel_1822/2627096133.py:25: RuntimeWarning: invalid value encountered in divide
      pt2 = pt2 / pt2[2]



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[11], line 24
         20 plot_horizon_line(im, horizon_line)
         22 # Part 2
         23 # <YOUR IMPLEMENTATION> Solve for the camera parameters (f, u, v)
    ---> 24 f, u, v = get_camera_parameters()
         26 # Part 3
         27 # <YOUR IMPLEMENTATION> Solve for the rotation matrix
         28 R = get_rotation_matrix()


    TypeError: cannot unpack non-iterable NoneType object



```python

```
