#!/usr/bin/env python
# encoding=utf-8

'''
FP-Growth FP means frequent pattern
the FP-Growth algorithm needs:
1. FP-tree (class treeNode)
2. header table (use dict)
'''

import logging

TRACE = logging.DEBUG - 1
logging.basicConfig(
    level=logging.DEBUG,
    # level=TRACE,
    format='[%(levelname)s %(module)s line:%(lineno)d] %(message)s',
)

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode      #needs to be updated
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):
        print '  '*ind, self.name, ' ', self.count
        for child in self.children.values():
            child.disp(ind+1)

def createTree(dataSet, minSupport=3):
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    for k in headerTable.keys():
        if headerTable[k] < minSupport:
            del(headerTable[k])

    # print(headerTable)
    # return headerTable
    freqItemSet = set(headerTable.keys())
    # print(freqItemSet)

    if len(freqItemSet) == 0: return None, None
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]

    retTree = treeNode('Null Set', 1, None)

    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:  #put transaction items in order
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)

    return retTree, headerTable

def updateTree(orderedItems, inTree, headerTable, count):
    if orderedItems[0] in inTree.children:
        inTree.children[orderedItems[0]].inc(count)
    else:
        inTree.children[orderedItems[0]] = treeNode(orderedItems[0], count, inTree)
        if headerTable[orderedItems[0]][1] == None:
            headerTable[orderedItems[0]][1] = inTree.children[orderedItems[0]]
        else:
            updateHeader(headerTable[orderedItems[0]][1], inTree.children[orderedItems[0]])
    if len(orderedItems) > 1:
        updateTree(orderedItems[1::], inTree.children[orderedItems[0]], headerTable, count)

def updateHeader(sourceNode, leafNode):
    while sourceNode.nodeLink != None:
        sourceNode = sourceNode.nodeLink
    sourceNode.nodeLink = leafNode

def ascendTree(leafNode, prefixPath): #ascends from leaf node to root
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

def findPrefixPath(basePat, treeNode): #treeNode comes from header table
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1])]#(sort header table)
    for basePat in bigL:  #start from bottom of header table
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        #print 'finalFrequent Item: ',newFreqSet    #append to set
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        #print 'condPattBases :',basePat, condPattBases
        #2. construct cond FP-tree from cond. pattern base
        myCondTree, myHead = createTree(condPattBases, minSup)
        #print 'head from conditional tree: ', myHead
        if myHead != None: #3. mine cond. FP-tree
            print 'conditional tree for: ',newFreqSet
            myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

def testKosarak(filename='./kosarak.dat'):
    parsedDat =  [line.split() for line in open(filename).readlines()]

    initSet = createInitSet(parsedDat)
    myFtree, myHeaderTab = createTree(initSet, 100000)
    myFrequentList = []
    mineTree(myFtree, myHeaderTab, 100000, set([]), myFrequentList)

# import fpGrowth
# reload(fpGrowth)

# simpDat = fpGrowth.loadSimpDat()
# retDict = fpGrowth.createInitSet(simpDat)
# retTree, headerTable = fpGrowth.createTree(retDict)
#
# fpGrowth.ascendTree(headerTable['x'][1], [])
# fpGrowth.findPrefixPath('x', headerTable['x'][1])
# fpGrowth.findPrefixPath('z', headerTable['z'][1])
# fpGrowth.findPrefixPath('r', headerTable['r'][1])

# frequentItems = []
# fpGrowth.mineTree(retTree, headerTable, 3, set([]), frequentItems)

# filename='./kosarak.dat'
# parsedDat =  [line.split() for line in open(filename).readlines()]
# initSet = fpGrowth.createInitSet(parsedDat)
# myFtree, myHeaderTab = fpGrowth.createTree(initSet, 100000)
# myFrequentList = []
# fpGrowth.mineTree(myFtree, myHeaderTab, 100000, set([]), myFrequentList)
# fpGrowth.mineTree(myFtree, myHeaderTab, 150000, set([]), myFrequentList)

