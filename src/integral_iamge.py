
def get_integral_image(img):
    """
    Returns integral image given an image
    ---------
    Arguments
    ---------
    img [np.array]: Numpy array of shape (h,w) denoting grayscale image
    -------
    Returns
    -------
    integral_img [np.array]: Numpy array of shape (h,w) denoting integral image
    """
    h,w = img.shape
    integral_img = np.zeros((h,w))
    integral_img[0,:] = img[0,:].cumsum()
    integral_img[:,0] = img[:,0].cumsum()
    for i in range(1,h):
        for j in range(1,w):
            integral_img[i,j] = integral_img[i-1,j]+integral_img[i,j-1]+img[i,j]-integral_img[i-1,j-1]    
    return integral_img


def get_sum_pixels(img_integral, coord_tl, coord_br):
    """
    Returns sum of pixels in a box using integral image
    ---------
    Arguments
    ---------
    img_integral [np.array]: Numpy array of shape (h,w) denoting integral image
    coord_tl [tuple(int)]: Top left coordinate of box
    coord_br [tuple(int)]: Bottom right coordinate of box
    -------
    Returns
    -------
    float: Sum of pixel intensities in box
    """
    (a,b), (c,d) = coord_tl, coord_br
    s1 = img_integral[c,d]
    s2 = img_integral[a,d] if a>0 else 0
    s3 = img_integral[c,b] if b>0 else 0
    s4 = img_integral[a,b] if a>0 and b>0 else 0
    return s1-s2-s3+s4

