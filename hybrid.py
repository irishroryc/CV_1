"""
Rory Connolly
rlc367
February 11, 2019

Code for the hybrid images assignment in CS 5670

(cs5670_python_env) Rorys-MacBook-Pro:Project1_Hybrid_Images roryc$ python test.py 
...................
----------------------------------------------------------------------
Ran 19 tests in 1.568s

OK

"""

import sys
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')

import cv2
import numpy as np

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''

    # Pull data about number of dimensions and kernel shape
    dimensions = len(img.shape)
    kernel_height, kernel_width = kernel.shape


    # Calculate relative locations and weights for kernel
    kernel_weights = []
    for row in range(len(kernel)):
        for col in range(len(kernel[row])):
            # Create an entry in the list of weights if it is non-zero
            if kernel[row][col] != 0:
                kernel_weights.append((col-(kernel_width/2),row-(kernel_height/2),kernel[row][col]))

    # Helper function to generate new images based on a list of kernel weights
    def cross_helper(image,width,height,k_weights):
        new_image = np.zeros_like(image)
        for i in range(height):
            for j in range(width):
                new_val = 0
                for weights in k_weights:
                    if ((j + weights[0]) >= 0 and (j + weights[0]) < width
                        and (i + weights[1]) >= 0 and (i + weights[1]) < height):
                        #print "Running values for img["+str(i+weights[1])+"]["+str(j+weights[0])+"]"
                        new_val += (weights[2] * image[i+weights[1]][j+weights[0]])
                new_image[i][j] = new_val
        return new_image


    # Run code specific to grayscale images if img has 2 dimensions
    if dimensions == 2:
        img_height, img_width = img.shape
        new_image = cross_helper(img,img_width,img_height,kernel_weights)
        return new_image
    # Otherwise run code for RGB channels
    else:
        img_height, img_width, image_channels = img.shape
        r,g,b = np.split(img,3,axis=2)
        new_image = cross_helper(r,img_width,img_height,kernel_weights)
        new_image = np.dstack((new_image, cross_helper(g,img_width,img_height,kernel_weights)))
        new_image = np.dstack((new_image, cross_helper(b,img_width,img_height,kernel_weights)))
        return new_image

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # Re-use code for cross correlation but with flipped kernel
    return cross_correlation_2d(img,np.flip(kernel))

def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # Helper function to generate gaussian value for specific cells
    def get_gauss_value(x,y):
        gauss_constant = (1/(2*sigma*np.pi)**2)
        gauss_base = (np.e)
        gauss_exponent = -(float(x**2 + y**2)/(2*(sigma**2)))
        gauss_value = gauss_constant * np.power(gauss_base, gauss_exponent)
        return gauss_value


    center_width = width/2
    center_height = height/2

    kernel = np.zeros((height,width))

    # Build out gaussian kernel based on calculated values from helper fcn
    for i in range(height):
       for j in range(width):
            kernel[i][j] = get_gauss_value(j-center_width,i-center_height)
    
    # Return normalized kernel values
    return kernel/np.sum(kernel)


def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # Reuse convolve code with gaussian kernel generator for low pass
    return convolve_2d(img,gaussian_blur_kernel_2d(sigma, size, size))


def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # Subtract the result of low-pass from original image for high pass image
    return img - low_pass(img,sigma,size)

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)
