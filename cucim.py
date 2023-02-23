def apply_rgb2hsv(image, cuda: bool = False):

    if cuda == True:
        image = cp.asarray(image)
        image = rgb2hsv(image)
    else:
        image = skimage.color.rgb2hsv(image)

    return image


def apply_rgb2gray(image, cuda: bool = False):

    if cuda == True:
        image = cp.asarray(image)
        image = rgb2gray(image)
    else:
        image = skimage.color.rgb2gray(image)

    return image


def apply_gaussian(image, cuda: bool = False):

    if cuda == True:
        image = cp.asarray(image)
        image = gaussian(image, sigma=10)
    else:
        image = skimage.filters.gaussian(image, sigma=10)

    return image


def apply_equalize_hist(image, cuda: bool = False):

    if cuda == True:
        image = cp.asarray(image)
        image = equalize_hist(image)
    else:
        image = skimage.exposure.equalize_hist(image)

    return image


def apply_canny(image, cuda: bool = False):

    if cuda == True:
        image = rgb2gray(cp.asarray(image))
        image = canny(image)
    else:
        image = skimage.color.rgb2gray(image)
        image = skimage.feature.canny(image)

    return image


def apply_sobel(image, cuda: bool = False):

    if cuda == True:
        image = cp.asarray(image)
        image = sobel(image)
    else:
        image = skimage.filters.sobel(image)

    return image

# function list

benchmark_list = {'rgb2gray': apply_rgb2gray, 'rgb2hsv': apply_rgb2hsv, 'gaussian': apply_gaussian,
                  'equalize_hist': apply_equalize_hist, 'canny': apply_canny, 'sobel': apply_sobel}

# gpu n no gpu
gpu_available = [True, False]

