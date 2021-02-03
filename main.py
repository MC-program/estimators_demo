#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
from sklearn import preprocessing        # data prep


################## my packages
from plot import plot   
import xml_parser
import estimators
from debug_package import debug, info, warning, error, critical



OPTIMIZE_MODELS    = True
#OPTIMIZE_MODELS    = False



#COMPATIBLE_EXTENSIONS = ["xml", "pdf", "txt", "csv"]
COMPATIBLE_EXTENSIONS = ["xml"]

### for CSV export of ypred
ypred_names = [
"entry_2;", 
"entry_16;", 
"entry_28;", 
"entry_40;", 
"entry_44;", 
"entry_49;", 
"entry_61;", 
"entry_68;", 
"entry_74;", 
"entry_75;", 
"entry_90;", 
"entry_126;", 
"entry_132;", 
"entry_141;", 
"entry_157;", 
"entry_168;", 
"entry_169;", 
"entry_176;", 
"entry_191;", 
"entry_198;", 
"entry_202;", 
"entry_205;" ] 

#print("System arguments are:", sys.argv)

if len(sys.argv) < 2:
    print("1. argument: input name with relative or absolute path")
    exit()


if sys.argv[0] == "main.py": # a normal Python call?
    filelist = []
    #arglist = []

    if any([("."+ext in sys.argv[1]) for ext in COMPATIBLE_EXTENSIONS]):
        filelist.append(sys.argv[1])

    if len(sys.argv) >1:
        #arglist.extend(sys.argv[2:])
        inputfile2 = sys.argv[2]

    if len(filelist) > 1:
        #print("Input files are")
        for inputfile in filelist:
            print(inputfile)

    for inputfile in filelist:
        print("Read input file", inputfile)
        (Xtrain,ytrain) = xml_parser.read(inputfile)  # perform testing / x-validation
    print("Read input file", inputfile2)
    (Xtest,_) = xml_parser.read(inputfile2)

    ### Data prep
    #   Normalize on the input side
    allInputData = np.row_stack((Xtrain,Xtest))
    minMaxTransform = preprocessing.MinMaxScaler()
    minMaxTransform.fit(allInputData)
    XtrainMinMax = minMaxTransform.transform(Xtrain)
    XtestMinMax  = minMaxTransform.transform(Xtest)
    #print(XtrainMinMax)
    #print(XtestMinMax)

    model = estimators.getWinner(XtrainMinMax, ytrain, 20, OPTIMIZE_MODELS)  # best model and print plots
    model = estimators.updateModel(model, XtrainMinMax, ytrain)   # fit model again and print plot


    # final prediction. There are no target values
    # So, no need for a visualization :)
    ypred = model.predict(XtestMinMax)

    print("predicted values ypred=", ypred)

    fileExport = "entry_n;energy\n"

    assert( len(ypred) == len(ypred_names) )

    for pair in zip(ypred_names, ypred):
        fileExport += pair[0] +str(pair[1]) +"\n"

    #print(fileExport)
    xml_parser.writeFile("validation_prediction.csv", fileExport)


    ### VISUALIZATION FINAL END ###
    plot.plotSavePdf()

    exit() ######## necessary otherwise it loops again over?!

else:         # Iptyhon Notebook

    parser_package.init()

# EOF
