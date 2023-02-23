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

# benchmark function
# run benchmark for each function
for benchmark in benchmark_list.values():
    print(f"{benchmark.__name__} benchmark")
    print("="*10)
    for gpu_available in [True, False]:
        start = time.time()
        for image_name in image_list:
            image = skimage.io.imread(image_folder + image_name)
            image = benchmark(image, cuda=gpu_available)
        end = time.time()
        print(f"{benchmark.__name__} with {'gpu' if gpu_available else 'no_gpu'}: {end - start} seconds")
        print("="*10)

'''
apply_rgb2gray benchmark
==========
apply_rgb2gray with gpu: 38.2649507522583 seconds
==========
apply_rgb2gray with no_gpu: 48.634881019592285 seconds
==========

apply_rgb2hsv benchmark
==========
apply_rgb2hsv with gpu: 38.00744414329529 seconds
==========
apply_rgb2hsv with no_gpu: 231.91903972625732 seconds
==========

apply_gaussian benchmark
==========
apply_gaussian with gpu: 38.130988121032715 seconds
==========
apply_gaussian with no_gpu: 281.2401978969574 seconds

==========
apply_equalize_hist benchmark
==========
apply_equalize_hist with gpu: 40.849740743637085 seconds
==========
apply_equalize_hist with no_gpu: 134.53890371322632 seconds

==========
apply_canny benchmark
==========
apply_canny with gpu: 39.090564250946045 seconds
==========
apply_canny with no_gpu: 281.1434905529022 seconds

==========
apply_sobel benchmark
==========
apply_sobel with gpu: 35.74287939071655 seconds
==========
apply_sobel with no_gpu: 264.8947470188141 seconds
==========

'''