import sys
import math
import time
import random
from collections import namedtuple, deque
from typing import List, Optional
import numpy as np

# Define structures using namedtuple for immutability or simple classes for mutability
class Node:
    def __init__(self, id: int, name: str, class_number: int, score: float):
        self.id = id
        self.name = name
        self.class_number = class_number
        self.score = score

class Link:
    def __init__(self, node1: int, node2: int, weight: float = 0.0):
        self.node1 = node1
        self.node2 = node2
        self.weight = weight
        self.next = None

# Initialize variables
nnodi = 0
nlink = 0
ngenes = 0
nseedgenes = 0
totscore = 0.0
hash_idgene = []
ring = []
count = []
rank = []
weightNS = []
diffus = []
wdt = 0.0
alpha = 0.5
minscore = 0.01
maxscore = 0.33

# Define functions

def geneRand():
    return random.random()

def PutInLinkList(testa: Optional[Link], coda: Optional[Link], elemento: Link) -> (Link, Link):
    elemento.next = None
    if testa is None:
        return elemento, elemento
    else:
        coda.next = elemento
        return testa, elemento

def NotExist(testa: Link, n1: int, n2: int) -> int:
    app = testa
    while app:
        if (app.node1 == n1 and app.node2 == n2) or (app.node1 == n2 and app.node2 == n1):
            return 0
        app = app.next
    return 1

def ReadRegularLink(filename: str) -> Link:
    global nnodi, nlink
    with open(filename, "r") as fp:
        newlinkhead, newlinktail = None, None
        for line in fp:
            l1, l2 = map(int, line.strip().split())
            nnodi = max(nnodi, l1, l2)
            elem = Link(l1, l2)
            newlinkhead, newlinktail = PutInLinkList(newlinkhead, newlinktail, elem)
            nlink += 1
    nnodi += 1
    return newlinkhead

hash_idgene = []
def createPositiveList(link, genes):
    newlinkhead = None
    newlinktail = None

    while link is not None:
        if genes[hash_idgene[link.node1]].class_number == 1 and genes[hash_idgene[link.node2]].class_number == 1:
            elem = Link(link.node1, link.node2)
            #elem.node1 = link.node1
            #elem.node2 = link.node2
            newlinkhead, newlinktail = PutInLinkList(newlinkhead, newlinktail, elem)
        link = link.next

    return newlinkhead


def ReadRegularGenes(filename: str) -> List[Node]:
    global ngenes, nseedgenes, totscore, hash_idgene
    nodes = []
    with open(filename, "r") as fp:
        for line in fp:
            id, gene, score = line.strip().split()
            id, score = int(id), float(score)
            nodes.append(Node(id, gene, 1 if score > 0 else 0, score))
            totscore += score if score > 0 else 0
            if score > 0:
                nseedgenes += 1
    ngenes = len(nodes)
    hash_idgene = [0] * ngenes
    for idx, node in enumerate(nodes):
        hash_idgene[node.id] = idx
    return nodes

def Connected(link: Link) -> int:
    field = [0] * nnodi
    field[0] = 1
    dimclust = 1
    change = 1
    while change:
        change = 0
        elem = link
        while elem:
            if field[elem.node1] ^ field[elem.node2]:
                field[elem.node1] = field[elem.node2] = 1
                dimclust += 1
                change = 1
            elem = elem.next
    return dimclust

def Clusters(link, nc):
    field = [0] * nnodi
    nclust = 1
    field[0] = nclust
    dimclust = 1
    sumclust = 0
    while sumclust < nnodi:
        change = 1
        while change:
            change = 0
            elem = link
            while elem:
                if ((field[elem.node1] == nclust and not field[elem.node2]) or 
                    (field[elem.node2] == nclust and not field[elem.node1])):
                    field[elem.node1] = field[elem.node2] = nclust
                    dimclust += 1
                    change = 1
                elem = elem.next
        sumclust += dimclust
        if sumclust < nnodi:
            nclust += 1
            dimclust = 1
            i = 0
            while field[i]:
                i += 1
            field[i] = nclust
    #nc[0] = nclust
    nc = nclust
    return field

def ClustersSeedGenes(link, genes, nc):
    field = [0] * nnodi  # Initialize the field array with zeros
    nclust = 1
    i = 0

    # Find the first gene with class == 1
    while genes[hash_idgene[i]].class_number != 1:
        i += 1

    field[i] = nclust
    dimclust = 1
    sumclust = 0

    while True:
        change = True
        while change:
            change = False
            elem = link
            while elem is not None:
                if ((field[elem.node1] == nclust and field[elem.node2] == 0) or
                    (field[elem.node2] == nclust and field[elem.node1] == 0)):
                    field[elem.node1] = nclust
                    field[elem.node2] = nclust
                    dimclust += 1
                    change = True
                elem = elem.next

        sumclust += dimclust
        nclust += 1
        dimclust = 1
        i = 0

        # Find the next unclustered gene with class == 1
        while i < nnodi and (field[i] != 0 or genes[hash_idgene[i]].class_number != 1):
            i += 1

        if i < nnodi:
            field[i] = nclust
        else:
            break

    #nc[0] = nclust
    nc = nclust
    return field

def computeDegree(link: Link) -> List[int]:
    grado = [0] * nnodi
    while link:
        grado[link.node1] += 1
        grado[link.node2] += 1
        link = link.next
    return grado

'''def netShort(link: Link, genes: List[Node]):
    global weightNS
    weightNS = [0.0] * nnodi
    for i in range(nnodi):
        if genes[hash_idgene[i]].class_number == 1:
            weightNS[i] = genes[hash_idgene[i]].score / maxscore
        elif genes[hash_idgene[i]].class_number == -99:
            weightNS[i] = 0.0
        else:
            weightNS[i] = alpha * minscore / maxscore

    d = [[float('inf')] * nnodi for _ in range(nnodi)]
    elem = link
    while elem:
        if weightNS[elem.node1] + weightNS[elem.node2] > 0:
            elem.weight = 2.0 / (weightNS[elem.node1] + weightNS[elem.node2])
        else:
            elem.weight = float('inf')
        d[elem.node1][elem.node2] = elem.weight
        d[elem.node2][elem.node1] = elem.weight
        elem = elem.next

    for k in range(nnodi):
        for i in range(nnodi):
            for j in range(nnodi):
                if d[i][j] > d[i][k] + d[k][j]:
                    d[i][j] = d[i][k] + d[k][j]
    for k in range(nnodi):
        d_k = d[:, k]  # Extract the k-th column
        d_kj = d_k + d[k, :]  # Calculate d[i][k] + d[k][j] for all i, j
        d = np.minimum(d, d_kj)
        
    for k in range(nnodi):
        for i in range(nnodi):
            d_ik = d[i][k]  # Store d[i][k] to avoid redundant access
            for j in range(nnodi):
                d_ij = d[i][j]
                d_kj = d[k][j]
                if d_ij > d_ik + d_kj:
                    d[i][j] = d_ik + d_kj

    weightNS = [sum(1.0 / d[i][j] for j in range(nnodi) if i != j) for i in range(nnodi)]'''


# Assuming nnodi, hash_idgene, alpha, minscore, maxscore are defined somewhere in the program
# Assuming Link and Node classes are defined elsewhere

#import numpy as np

def myNetShort(link, genes):
    weightNS = np.zeros(nnodi)
    
    for i in range(nnodi):
        gene = genes[hash_idgene[i]]
        if gene.class_number == 1:
            weightNS[i] = gene.score / maxscore
        elif gene.class_number == -99:
            weightNS[i] = 0.0
        else:
            weightNS[i] = alpha * minscore / maxscore

    elem = link
    while elem:
        elem.weight = 2.0 / (weightNS[elem.node1] + weightNS[elem.node2])
        elem = elem.next

    weightns = np.zeros(nnodi)

    for i in range(nnodi):
        weightns.fill(0)
        
        elem = link
        change = 0
        while elem:
            if elem.node1 == i:
                weightns[elem.node2] = elem.weight
                change += 1
            elif elem.node2 == i:
                weightns[elem.node1] = elem.weight
                change += 1
            elem = elem.next

        if change == 0:
            print(f"isolated node {i}")
        else:
            stat = 0
            while True:
                elem = link
                change = 0
                while elem:
                    if weightns[elem.node1 ] > 0.0:
                        appw = weightns[elem.node1] + elem.weight
                        if ((weightns[elem.node2] == 0 or weightns[elem.node2] > appw) and elem.node2 != i):
                            weightns[elem.node2] = appw
                            change += 1
                    elif weightns[elem.node2] > 0.0:
                        appw = weightns[elem.node2] + elem.weight
                        if ((weightns[elem.node1] == 0 or weightns[elem.node1] > appw) and elem.node1 != i):
                            weightns[elem.node1 ] = appw
                            change += 1
                    elem = elem.next

                stat += 1
                print(f"node {i} stat {stat} change {change}")
                if change == 0:
                    break

        for j in range(nnodi):
            if weightns[j] > 0:
                weightNS[i] += 1.0 / weightns[j]


def netShort(link: Link, genes: List[Node]):
    global weightNS
    weightNS = np.zeros(nnodi, dtype=float)
    
    print("Computing initial genes weight", file=sys.stderr)
    for i in range(nnodi):
        gene = genes[hash_idgene[i]]
        if gene.class_number == 1:
            weightNS[i] = gene.score / maxscore
        elif gene.class_number == -99:
            weightNS[i] = 0.0
        else:
            weightNS[i] = alpha * minscore / maxscore

    d = np.zeros((nnodi, nnodi), dtype=float)
    
    print("Computing initial distance", file=sys.stderr)
    elem = link
    while elem is not None:
        node1 = elem.node1
        node2 = elem.node2
        
        if weightNS[node1] + weightNS[node2] > 0:
            elem.weight = 2.0 / (weightNS[node1] + weightNS[node2])
        else:
            elem.weight = float('inf')
        
        ij = node1 * nnodi + node2
        ji = node2 * nnodi + node1
        
        if (ij >= nnodi * nnodi) or (ji >= nnodi * nnodi):
            print(f"Problem: {ij} or {ji} greater than {nnodi * nnodi}", file=sys.stderr)
            sys.exit(-1)
        
        d[node1][node2] = elem.weight
        d[node2][node1] = elem.weight
        
        elem = elem.next
    
    print("Resetting two-step distance", file=sys.stderr)
    for i in range(nnodi):
        for j in range(nnodi):
            if i != j and d[i][j] == 0:
                d[i][j] = float('inf')
    
    print("Computing shortest distance", file=sys.stderr)
    for k in range(nnodi):
        print(f"{k} of {nnodi}", file=sys.stderr)
        for i in range(nnodi):
            for j in range(nnodi):
                if d[i][j] > d[i][k] + d[k][j]:
                    d[i][j] = d[i][k] + d[k][j]
    
    # Reset weightNS array
    weightNS = np.zeros(nnodi, dtype=float)
    
    for i in range(nnodi):
        for j in range(nnodi):
            if i != j:
                weightNS[i] += 1.0 / d[i][j]


def score2rank(A):
    return (1.0 - (A) / maxscore)

import numpy as np

def netRank(link, genes, degree):
    ring = np.zeros(nnodi, dtype=int)
    count = np.zeros(nnodi, dtype=int)
    rank = np.zeros(nnodi, dtype=float)

    # assign a rank to the seed genes
    elem = link
    change = 0
    while elem:
        if genes[hash_idgene[elem.node1]].class_number == 1:
            ring[elem.node1] = 1
            count[elem.node1] += 1
            rank[elem.node1] += score2rank(genes[hash_idgene[elem.node2]].score)
        
        if genes[hash_idgene[elem.node2]].class_number == 1:
            ring[elem.node2] = 1
            count[elem.node2] += 1
            rank[elem.node2] += score2rank(genes[hash_idgene[elem.node1]].score)
        
        elem = elem.next

    for i in range(nnodi):
        if genes[hash_idgene[i]].class_number == 1:
            print(f"{genes[hash_idgene[i]].name} {genes[hash_idgene[i]].score} {rank[i]} {count[i]} =", end=" ")
            rank[i] = alpha * score2rank(genes[hash_idgene[i]].score) + (1 - alpha) * rank[i] / count[i]
            print(f"{rank[i]}")

    nring = 1
    while True:
        elem = link
        change = 0
        while elem:
            if ring[elem.node1] == nring:
                #if ring[elem.node2] == nring + 1 or ring[elem.node2] == 0:
                if ((ring[elem.node2] == nring + 1) or (ring[elem.node2] == 0)):
                    ring[elem.node2] = nring + 1
                    count[elem.node2] += 1
                    rank[elem.node2] += rank[elem.node1] - (nring - 1)
                    change = 1
            elif ring[elem.node2] == nring:
                if ((ring[elem.node1] == nring + 1) or (ring[elem.node1] == 0)):
                    ring[elem.node1] = nring + 1
                    count[elem.node1] += 1
                    rank[elem.node1] += rank[elem.node2] - (nring - 1)
                    change = 1
            elem = elem.next

        nring += 1
        sum_rank = 0
        stat = 0
        for i in range(nnodi):
            if ring[i] == nring:
                rank[i] = (nring - 1) + (rank[i] + degree[i] - count[i]) / degree[i]
                sum_rank += rank[i]
                stat += 1
        
        if stat > 0:
            print(f"{nring} {sum_rank/stat} {stat}")
        
        if not change:
            break

'''def netRank(link: Link, genes: List[Node], degree: List[int]):
    global rank, count, ring
    ring = [0] * nnodi
    count = [0] * nnodi
    rank = [0.0] * nnodi

    elem = link
    while elem:
        if genes[hash_idgene[elem.node1]].class_number == 1:
            ring[elem.node1] = 1
            count[elem.node1] += 1
            rank[elem.node1] += score2rank(genes[hash_idgene[elem.node2]].score)
        if genes[hash_idgene[elem.node2]].class_number == 1:
            ring[elem.node2] = 1
            count[elem.node2] += 1
            rank[elem.node2] += score2rank(genes[hash_idgene[elem.node1]].score)
        elem = elem.next

    for i in range(nnodi):
        if genes[hash_idgene[i]].class_number == 1:
            rank[i] = alpha * score2rank(genes[hash_idgene[i]].score) + (1 - alpha) * (rank[i] / count[i])

    nring = 1
    change = 1
    while change:
        change = 0
        elem = link
        while elem:
            if ring[elem.node1] == nring and (ring[elem.node2] == nring + 1 or not ring[elem.node2]):
                ring[elem.node2] = nring + 1
                count[elem.node2] += 1
                rank[elem.node2] += rank[elem.node1] - (nring - 1)
                change = 1
            elif ring[elem.node2] == nring and (ring[elem.node1] == nring + 1 or not ring[elem.node1]):
                ring[elem.node1] = nring + 1
                count[elem.node1] += 1
                rank[elem.node1] += rank[elem.node2] - (nring - 1)
                change = 1
            elem = elem.next
        nring += 1
        sum_rank = sum(rank[i] for i in range(nnodi) if ring[i] == nring)
        stat = sum(1 for i in range(nnodi) if ring[i] == nring)
        if stat > 0:
            print(f"{nring} {sum_rank / stat} {stat}")'''


def diffusionHeat(link: Link, degree: List[int], fieldnew: List[float], fieldold: List[float]):
    global diffus
    diffus = [0.0] * nnodi
    elem = link
    while elem:
        if degree[elem.node2] != 0:
            diffus[elem.node1] += fieldold[elem.node2] / degree[elem.node2]
        if degree[elem.node1] != 0:
            diffus[elem.node2] += fieldold[elem.node1] / degree[elem.node1]
        elem = elem.next

    for i in range(nnodi):
        fieldnew[i] = (1 - wdt) * fieldold[i] + diffus[i] * wdt


def diffusionInfo(link: Link, degree: List[int], fieldnew: List[float], fieldold: List[float]):
    global diffus
    diffus = [0.0] * nnodi
    elem = link
    while elem:
        diffus[elem.node1] += fieldold[elem.node2]
        diffus[elem.node2] += fieldold[elem.node1]
        elem = elem.next

    for i in range(nnodi):
        fieldnew[i] = (1 - wdt * degree[i]) * fieldold[i] + diffus[i] * wdt


def oneRing(present, ringnew, ringold, step):
    print(f"nuova versione step {step}")
    nchanged = 0
    while present is not None:
        if ringold[present.node1] == step:
            ringnew[present.node1] = step
            if ringold[present.node2] == 0:
                ringnew[present.node2] = step + 1
                nchanged += 1
        if ringold[present.node2] == step:
            ringnew[present.node2] = step
            if ringold[present.node1] == 0:
                ringnew[present.node1] = step + 1
                nchanged += 1
        present = present.next
    
    # Python equivalent of memcpy
    ringold[:] = ringnew[:]
    return nchanged

def covDegree(present, covdeg, ringGene):
    while present is not None:
        if ringGene[present.node1] == 1:
            covdeg[present.node2] += 1
        elif ringGene[present.node2] == 1:
            covdeg[present.node1] += 1
        present = present.next


import sys
import time
import random
import numpy as np
from math import sqrt


nARG = 3
ARGfileLink = 1
ARGfileGene = 2
ARGfileOut = 3


def main():
    if len(sys.argv) != nARG + 1:
        sys.stderr.write(f"[{sys._getframe().f_code.co_name}]: Uso: {sys.argv[0]} filelink filegene fileout\n")
        sys.exit(1)

    # Seed random generator
    '''if not FIXEDSEED:
        if UNIX:
            seed = int(time.time())
            random.seed(seed)
            startseed = seed
            print(f"UNIX-SEME = {startseed}")
        else:
            seed = int(time.time())
            random.seed(seed)
            startseed = seed
            print(f"WINDOWS-SEME = {startseed}")'''

    linklista = ReadRegularLink(sys.argv[ARGfileLink])
    genes = ReadRegularGenes(sys.argv[ARGfileGene])

    # Controllo connessione e stampa geni isolati
    print(f"major cluster {Connected(linklista)} elements over {nnodi}")
    nc = 0
    clusters = Clusters(linklista, nc)
    nc = len(clusters)
    elemClus = np.zeros(nc + 1, dtype=int)
    for i in range(1, nnodi):
        elemClus[clusters[i]] += 1

    print("clusters:")
    for i in range(1, nc + 1):
        print(f"{i} {elemClus[i]}")

    print("singoli:")
    for i in range(nnodi):
        if clusters[i] != 1:
            print(f"{genes[i].name} {genes[i].class_number}")
    
    # Blocco per il calcolo del grado e della clusterizzazione del sottografo dei seed genes
    print("compute seed subgraph:")
    linkseed = createPositiveList(linklista, genes)
    print("clusterization of seed subgraph:")
    clusters = ClustersSeedGenes(linkseed, genes, nc)
    nc = len(clusters)
    elemClus = np.zeros(nc + 1, dtype=int)
    for i in range(1, nnodi):
        elemClus[clusters[i]] += 1
    for i in range(1, nc + 1):
        print(f"{i} {elemClus[i]}")

    print("compute degree of seed subgraph:")
    stat = 0
    sumdeg = 0
    vardeg = 0
    degree = computeDegree(linkseed)
    for i in range(nnodi):
        if genes[hash_idgene[i]].class_number == 1:
            print(f"{i} {genes[hash_idgene[i]].name} {degree[i]} {clusters[i]}")
    
    print("Parametri del sistema:")
    print(f"          file link:   {sys.argv[ARGfileLink]}")
    print(f"             n link:   {nlink}")
    print(f"          file geni:   {sys.argv[ARGfileGene]}")
    print(f"             n geni:   {ngenes}")
    print(f"        n geni seed:   {nseedgenes}")
    
    # Controllo connessione
    print(f"major cluster {Connected(linklista)} elements over {nnodi}")
    nc = []
    clusters = Clusters(linklista, nc)
    nc = len(clusters)
    elemClus = np.zeros(nc + 1, dtype=int)
    for i in range(1, nnodi):
        elemClus[clusters[i]] += 1
    
    print("clusters:")
    for i in range(1, nc + 1):
        print(f"{i} {elemClus[i]}")
    
    print("compute degree")
    stat = 0
    sumdeg = 0
    vardeg = 0
    degree = computeDegree(linklista)
    for i in range(nnodi):
        if clusters[i] == 1:
            sumdeg += degree[i]
            vardeg += degree[i] * degree[i]
            stat += 1
        else:
            genes[hash_idgene[i]].class_number = -99
            print(f"{i} {genes[hash_idgene[i]].name} {genes[hash_idgene[i]].class_number}")
    
    sumdeg /= float(stat)
    vardeg = sqrt((vardeg - sumdeg * sumdeg * nnodi) / (nnodi - 1.0))
    print(f"nodes {stat}, average degree: {sumdeg}, standard deviation: {vardeg}")
    
    # allocate memory
    ring = np.zeros(nnodi, dtype=int)
    count = np.zeros(nnodi, dtype=int)
    rank = np.zeros(nnodi, dtype=float)
    weightNS = np.zeros(nnodi, dtype=float)
    fieldHeat = np.zeros(nnodi, dtype=float)
    fieldInfo = np.zeros(nnodi, dtype=float)
    diffus = np.zeros(nnodi, dtype=float)
    app = np.zeros(nnodi, dtype=float)
    
    print("net short:")
    netShort(linklista, genes)
    
    print("net rank:")
    netRank(linklista, genes, degree)
    
    coeff = float(nseedgenes) / float(totscore)
    wdt = 0.001
    step = 263
    print(f"heat diffusion {wdt} {step}")
    fieldHeat.fill(0)
    app.fill(0)
    for i in range(nnodi):
        if genes[hash_idgene[i]].class_number == 1:
            fieldHeat[i] = coeff * genes[hash_idgene[i]].score
    
    for i in range(step):
        diffusionHeat(linklista, degree, app, fieldHeat)
    
    wdt = 0.0001
    step = 50
    print(f"info diffusion {wdt} {step}")
    fieldInfo.fill(0)
    app.fill(0)
    for i in range(nnodi):
        if genes[hash_idgene[i]].class_number == 1:
            fieldInfo[i] = coeff * genes[hash_idgene[i]].score
    
    for i in range(step):
        diffusionInfo(linklista, degree, app, fieldInfo)
    
    with open(sys.argv[ARGfileOut], "w") as fw:
        fw.write("name,class,degree,ring,NetRank,NetShort,HeatDiff,InfoDiff\n")
        for i in range(nnodi):
            fw.write(f"{genes[hash_idgene[i]].name},{genes[hash_idgene[i]].class_number},{degree[i]},{ring[i]},{rank[i]},{weightNS[i]},{fieldHeat[i]},{fieldInfo[i]}\n")
    
    # No need for explicit free as in C, Python handles memory management
    
    return 0

if __name__ == "__main__":
    main()


'''def main(argv):
    global wdt

    if len(argv) != 4:
        print(f"Usage: {argv[0]} filelink filegene fileout")
        sys.exit(1)

    linklista = ReadRegularLink(argv[1])
    genes = ReadRegularGenes(argv[2])

    print(f"major cluster {Connected(linklista)} elements over {nnodi}")
    nc = [0]
    clusters = Clusters(linklista, nc)
    print("clusters:")
    elemClus = [0] * (nc[0] + 1)
    for i in range(1, nnodi):
        elemClus[clusters[i]] += 1

    for i in range(1, nc[0] + 1):
        print(f"{i} {elemClus[i]}")

    print("compute degree")
    degree = computeDegree(linklista)
    for i in range(nnodi):
        if clusters[i] == 1:
            print(f"{i} {genes[hash_idgene[i]].name} {degree[i]} {clusters[i]}")

    # Allocation and initialization
    ring = [0] * nnodi
    count = [0] * nnodi
    rank = [0.0] * nnodi
    weightNS = [0.0] * nnodi
    fieldHeat = [0.0] * nnodi
    fieldInfo = [0.0] * nnodi
    diffus = [0.0] * nnodi
    app = [0.0] * nnodi

    #print("net short:")
    #netShort(linklista, genes)
    #myNetShort(linklista, genes)

    print("net rank:")
    netRank(linklista, genes, degree)

    coeff = nseedgenes / totscore
    wdt = 0.001
    step = 263
    print(f"heat diffusion {wdt} {step}")
    for i in range(nnodi):
        if genes[hash_idgene[i]].class_number == 1:
            fieldHeat[i] = coeff * genes[hash_idgene[i]].score
    for i in range(step):
        diffusionHeat(linklista, degree, app, fieldHeat)

    wdt = 0.0001
    step = 50
    print(f"info diffusion {wdt} {step}")
    for i in range(nnodi):
        if genes[hash_idgene[i]].class_number == 1:
            fieldInfo[i] = coeff * genes[hash_idgene[i]].score
    for i in range(step):
        diffusionInfo(linklista, degree, app, fieldInfo)

    with open(argv[3], "w") as fw:
        fw.write("name,class,degree,ring,NetRank,NetShort,HeatDiff,InfoDiff\n")
        for i in range(nnodi):
            fw.write(f"{genes[hash_idgene[i]].name},{genes[hash_idgene[i]].class_number},{degree[i]},"
                     f"{ring[i]},{rank[i]},{weightNS[i]},{fieldHeat[i]},{fieldInfo[i]}\n")

if __name__ == "__main__":
    main(sys.argv)'''
