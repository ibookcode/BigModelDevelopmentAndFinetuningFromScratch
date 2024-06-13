import math
import numpy

def softmax(inMatrix):
    m, n = numpy.shape(inMatrix)
    outMatrix = numpy.mat(numpy.zeros((m, n)))
    soft_sum = 0
    for idx in range(0, n):
        outMatrix[0, idx] = math.exp(inMatrix[0, idx])
        soft_sum += outMatrix[0, idx]
    for idx in range(0, n):
        outMatrix[0, idx] = outMatrix[0, idx] / soft_sum
    return outMatrix

if __name__ == '__main__':
    a = numpy.array([[1, 2, 1, 2, 1, 1, 3]])
    print(softmax(a))