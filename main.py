import cv2
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.preprocessing import normalize

import MatrixW

ITERATIONS_COUNT = 51
THREADS_NUMBER = 8


def splitImage(img, charNumVert, charNumHor):
    result = np.zeros((MatrixW.asciiCharWidth * MatrixW.asciiCharHeight, charNumHor * charNumVert))
    for y in range(charNumVert):
        for x in range(charNumHor):
            for j in range(MatrixW.asciiCharHeight):
                for i in range(MatrixW.asciiCharWidth):
                    color = img[x * MatrixW.asciiCharWidth + i][y * MatrixW.asciiCharHeight + j]
                    result[MatrixW.asciiCharWidth * j + i][charNumHor * y + x] = color

    return normalize(result, axis=0, norm='l2')


def updateH(v, w, h, threads_number, beta=2):
    epsilon = 1e-6
    vApprox = np.dot(w, h)

    def updateRow(j):
        for k in range(h.shape[1]):
            numerator = 0.0
            denominator = 0.0

            for i in range(w.shape[0]):
                if abs(vApprox[i][k]) > epsilon:
                    numerator += w[i][j] * v[i][k] / np.power(vApprox[i][k], 2.0 - beta)
                    denominator += w[i][j] * np.power(vApprox[i][k], beta - 1.0)
                else:
                    numerator += w[i][j] * v[i][k]
                    if beta - 1.0 > 0.0:
                        denominator += w[i][j] * np.power(vApprox[i][k], beta - 1.0)
                    else:
                        denominator += w[i][j]

            if abs(denominator) > epsilon:
                h[j][k] = h[j][k] * numerator / denominator
            else:
                h[j][k] = h[j][k] * numerator

    with ThreadPoolExecutor(threads_number) as p:
        p.map(updateRow, range(h.shape[0]))

    return h


if __name__ == '__main__':
    image = cv2.imread("putin.png", cv2.IMREAD_GRAYSCALE)
    ret, bwImage = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    imgHeight = bwImage[0].size
    imgWidth = int(bwImage.size / imgHeight)

    charNumVertical = imgHeight // MatrixW.asciiCharHeight
    charNumHorizontal = imgWidth // MatrixW.asciiCharWidth
    totalCharNum = charNumHorizontal * charNumVertical

    imgHeight = charNumVertical * MatrixW.asciiCharHeight
    imgWidth = charNumHorizontal * MatrixW.asciiCharWidth
    bwImage = cv2.resize(bwImage, (imgHeight, imgWidth))

    v = splitImage(bwImage, charNumVertical, charNumHorizontal)
    h = np.random.uniform(0, 1, (MatrixW.columnNum, totalCharNum))

    result = [['' for _ in range(charNumHorizontal)] for _ in range(charNumVertical)]

    for i in range(ITERATIONS_COUNT):
        updateH(v, MatrixW.wNorm, h, THREADS_NUMBER)

        if i == 10 or i == 25 or i == 50:
            for j in range(h.shape[1]):
                max = 0.0
                maxIndex = 0
                for n in range(h.shape[0]):
                    if max < h[n][j]:
                        max = h[n][j]
                        maxIndex = n
                result[j // charNumHorizontal][j % charNumHorizontal] = chr(MatrixW.firstAsciiCharCode + maxIndex) if (
                            max >= 0.2) else ' '

            for k in range(charNumVertical):
                print(*result[k], sep='')
            print()
            print()
