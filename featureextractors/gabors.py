import numpy as np
import cv2


def build_filters():
    filters = []
    ksize = 201
    for theta in np.arange(0, np.pi, np.pi / 4):
        kern = cv2.getGaborKernel((ksize, ksize), 5.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters


def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum


if __name__ == '__main__':
    import sys

    print(__doc__)
    try:
        img_fn = sys.argv[1]
    except:
        img_fn = cv2.imread('/media/leonardo/Images/full_mould_good/dataset_01/classes/good/1e8de054-7a32-4190-baa3-5663e6f0b268good.jpg')


    img = cv2.imread('/media/leonardo/Images/full_mould_good/dataset_01/classes/good/1e8de054-7a32-4190-baa3-5663e6f0b268good.jpg',0)
    if img is None:
        print('Failed to load image file:', img_fn)
        sys.exit(1)

    filters = build_filters()

    res1 = process(img, filters)
    cv2.imwrite("./watershed.png", res1)
    cv2.imshow('result', res1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

