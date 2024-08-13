import sys
import numpy as np

MAXINPUTLINE = 1024
alpha = 0.8
threshold = 0.000001
nARG = 5
_ARGfileIn_ = 1
_ARGheader_ = 2
_ARGfileOut_ = 3
_sQuantile_ = 4
_rnQuantile_ = 5

def uccomma2space(inputstr):
    return inputstr.replace(',', ' ').upper()

def StrSep(inputstr, ch):
    if inputstr is None or inputstr == '':
        return None
    parts = inputstr.split(ch, 1)
    retstr = parts[0].strip()
    if len(parts) > 1:
        inputstr = parts[1].strip()
    else:
        inputstr = None
    return retstr, inputstr

def createMatrix(nRow, nCol):
    return np.zeros((nRow, nCol))

def createMatrixInt(nRow, nCol):
    return np.zeros((nRow, nCol), dtype=int)

def freeMatrix(mat):
    del mat

def freeMatrixInt(mat):
    del mat

def createVect(nElem):
    return np.zeros(nElem)

def partition(vec, low, high):
    mid = vec[high]
    i = low - 1
    for j in range(low, high):
        if vec[j] < mid:
            i += 1
            vec[i], vec[j] = vec[j], vec[i]
    vec[i + 1], vec[high] = vec[high], vec[i + 1]
    return i + 1

def qSort(vec, low, high):
    if low < high:
        mid = partition(vec, low, high)
        qSort(vec, low, mid - 1)
        qSort(vec, mid + 1, high)

def setFeature(features, col, x, nRow):
    min_val = np.min(x)
    max_val = np.max(x)
    cof = 1.0 / (max_val - min_val)
    features[:, col] = cof * (x - min_val)

def main(argv):
    if len(argv) != nARG + 1:
        print(f"Usage: {argv[0]} fileIn flagHeader fileOut sQuantile rnQuantile")
        sys.exit(1)

    fr = open(argv[_ARGfileIn_], "r")

    header = int(argv[_ARGheader_])
    sQuantile = float(argv[_sQuantile_])
    rnQuantile = 1.0 - float(argv[_rnQuantile_])

    if header:
        fr.readline()

    first_line = fr.readline().strip()
    nfeature = len(first_line.split(',')) - 2

    nnodi = sum(1 for line in fr) + 1
    fr.seek(0)

    if header:
        fr.readline()

    gene = []
    class_ = []
    feature = np.zeros((nnodi, nfeature))

    for i in range(nnodi):
        line = fr.readline().strip()
        parts = line.split(',')
        gene.append(parts[0].strip())
        class_.append(int(parts[1].strip()))
        feature[i, :] = list(map(float, parts[2:]))
    fr.close()

    nseed = sum(class_)

    normFeatures = np.zeros((nnodi, nfeature))
    for j in range(nfeature):
        setFeature(normFeatures, j, feature[:, j], nnodi)

    W = np.zeros((nnodi, nnodi))
    aveFeat = np.zeros(nfeature)
    min_val = 2.0
    max_val = 0.0
    for n1 in range(nnodi):
        if class_[n1]:
            aveFeat += normFeatures[n1, :] / nseed
        for n2 in range(n1 + 1, nnodi):
            dist = np.sum((normFeatures[n1, :] - normFeatures[n2, :]) ** 2)
            W[n1, n2] = np.sqrt(dist)
            W[n2, n1] = W[n1, n2]
            min_val = min(min_val, W[n1, n2])
            max_val = max(max_val, W[n1, n2])

    ie = nnodi * (nnodi - 1) // 2
    Wvec = np.zeros(ie)
    cof = 1.0 / (max_val - min_val)
    idx = 0
    for n1 in range(nnodi):
        W[n1, n1] = 1
        for n2 in range(n1 + 1, nnodi):
            W[n1, n2] = 1.0 - cof * (W[n1, n2] - min_val)
            W[n2, n1] = W[n1, n2]
            Wvec[idx] = W[n1, n2]
            idx += 1

    qSort(Wvec, 0, ie - 1)
    soglia = Wvec[int(ie * sQuantile)]
    nRejected = 0
    D = np.ones(nnodi)
    for n1 in range(nnodi):
        for n2 in range(n1 + 1, nnodi):
            if W[n1, n2] < soglia:
                W[n1, n2] = 0
                W[n2, n1] = 0
                nRejected += 2
            else:
                D[n1] += W[n1, n2]
                D[n2] += W[n2, n1]

    Wr = np.zeros((nnodi, nnodi))
    for n1 in range(nnodi):
        Wr[n1, :] = W[n1, :] / D[n1]
    freeMatrix(W)

    relNeg = np.zeros(nnodi)
    relNegSort = np.zeros(nnodi)
    for n1 in range(nnodi):
        if not class_[n1]:
            relNeg[n1] = np.sqrt(np.sum((aveFeat - normFeatures[n1, :]) ** 2))
            relNegSort[n1] = relNeg[n1]

    qSort(relNegSort, 0, nnodi - 1)
    soglia = relNegSort[int(nnodi * rnQuantile)]
    nRelNeg = 0
    G0 = np.zeros(nnodi)
    for n1 in range(nnodi):
        if class_[n1]:
            G0[n1] = 1
        elif relNeg[n1] > soglia:
            G0[n1] = -1
            nRelNeg += 1

    renorm = nseed / nRelNeg
    sumG0 = 0
    for n1 in range(nnodi):
        if G0[n1] == -1:
            G0[n1] *= renorm
        sumG0 += G0[n1]

    Gt = np.zeros(nnodi)
    Gt_1 = G0.copy()
    while True:
        sumGt = 0
        for n1 in range(nnodi):
            Gt[n1] = alpha * G0[n1]
            Gt[n1] += (1 - alpha) * np.dot(Wr[:, n1], Gt_1)
            sumGt += Gt[n1]

        dist = np.sum(np.abs(Gt - Gt_1))
        if dist <= threshold:
            break
        Gt_1 = Gt.copy()

    nLabel = nnodi - nseed - nRelNeg
    appLabel = np.zeros(nLabel)
    i = 0
    for n1 in range(nnodi):
        if G0[n1] == 0:
            appLabel[i] = Gt[n1]
            i += 1

    qSort(appLabel, 0, nLabel - 1)
    sogliaLNWN = appLabel[int(0.33333 * nLabel)]
    sogliaWNLP = appLabel[int(0.66666 * nLabel)]

    label = np.zeros(nnodi)
    for n1 in range(nnodi):
        if G0[n1] == 1:
            label[n1] = 1
        elif G0[n1] < 0:
            label[n1] = 5
        elif Gt[n1] < sogliaLNWN:
            label[n1] = 4
        elif Gt[n1] < sogliaWNLP:
            label[n1] = 3
        else:
            label[n1] = 2

    fw = open(argv[_ARGfileOut_], "w")
    for i in range(nnodi):
        fw.write(f"{gene[i]} {Gt[i]} {int(label[i])}\n")
    fw.close()

if __name__ == "__main__":
    main(sys.argv)
