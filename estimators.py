print(__doc__)

import numpy as np

### grid search
from sklearn.model_selection import GridSearchCV
from sklearn import metrics         # MSE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

### single method with kernel/grid search
from sklearn.neighbors import KNeighborsRegressor   # knn, dont get confused with
                                                    # k-means clustering (unsupervised)
from sklearn.svm import SVR
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
### ensemble method / meta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor



####################     my modules
from plot import plot



# Generate sample data
#X = np.sort(5 * np.random.rand(40, 1), axis=0)
#y = np.sin(X).ravel()
#y[::5] += 3 * (0.5 - np.random.rand(8))        # noise






def calculateError(x,y):
    """ Get standard mean error

    MSE / distance based on L2 norm / metric
    """
    """
    diff = np.array(x) - np.array(y)
    dist = diff**2
    l2   = sum(dist)/len(diff)
    print("calculateError(): diff=", diff, "l2=", l2)
    return l2
    """
    return metrics.mean_squared_error(x,y)


def getWinner(X, y, test_size, optimizeModels):

    # append for each model:
    models          = []  # obj from model.fit 
    modelsShortName = []  # for heading of plot/page
    modelsYpred     = []  # ypred vector of model
    modelsLoss      = []  # loss value


    if test_size>0:   # make a validation set for us
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size, random_state=40)
    else:
        Xtrain, Xtest, ytrain, ytest = X, X, y, y

    print("training/test set is " +str(len(ytrain)) +"/"+ str(len(ytest)))

    xtest_uniform = range(len(ytest))     # index set to display elements in domain

    ### VISUALIZATION ###
    plot.plotBegin(1, 1, filename="plot.pdf")


    ###########################  NEXT MODEL  ##############################
    """
    plot.plotAddText("Method = Support Vector Machine (SVM)")
    ### kernel methods 
    svr_rbf   = SVR(kernel='rbf',   C=1, gamma='auto', epsilon=.1)
    svr_lin   = SVR(kernel='linear',C=1, gamma='auto')
    svr_poly  = SVR(kernel='poly',  C=1, gamma='auto', degree=3, epsilon=.1, coef0=1)
    #svr_sig   = SVR(kernel='sigmoid', C=10, gamma='auto', degree=3, epsilon=.1, coef0=1)  # not there
    svrs    = [svr_rbf, svr_lin, svr_poly]  # kernels ['RBF', 'Linear', 'Polynomial']

    for ix, svr in enumerate(svrs):

        model = svr.fit(Xtrain, ytrain)
        ypred = model.predict(Xtest)

        matrices.append(np.column_stack((xtest_uniform,ypred)))

    plot.addPlot(*matrices,
                 plottype="versus_2d",
                 labels=("index of points", "dependent variable"),
                 legend=["target"] + svrs,
                 filename="pdf")  ### do not leave inline comments with named parameters (!!?)
    #plot.plotShow()
    """
    if optimizeModels:   # grid search for hyper parameters
        parameters = {'kernel':  ('linear', 'rbf','poly'), 
                      'C':       (0.1, 0.5, 1, 5, 10),
                      'epsilon': (0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1),
                      'gamma':   (0.00001, 0.0001, 0.001, 0.005, 0.1, 1)
                     }
        ## cv: cross-validation 
        gridsearch = GridSearchCV(estimator=SVR(),
                                  param_grid=parameters,
                                  cv=5,
                                  scoring='neg_mean_squared_error',
                                  verbose=1,
                                  n_jobs=-1)
        model = gridsearch.fit(Xtrain,ytrain)
        params = model.best_params_
        print("SVR parameters found " + str(params))
    else:
        params = {'C': 1, 'epsilon': 0.00001, 'gamma': 0.001, 'kernel': 'rbf'}

    svr_final  = SVR(**params)          # winner for our use case
    model = svr_final.fit(Xtrain, ytrain)

    ypred = model.predict(Xtest)
    error = calculateError(ytest, ypred)
    models.append(model)
    modelsLoss.append(error)

    ### VISUALIZATION ###
    matrices = []
    matrices.append( np.column_stack((xtest_uniform,ytest)) )
    matrices.append(np.column_stack((xtest_uniform,ypred)))

    plot.plotBegin(1, 1, filename="plot.pdf")
    #plot.plotAddText("SVR " + str(model.best_params_))
    plot.plotAddText("SVR final. error=" +str(error))
    plot.addPlot(*matrices,
                 plottype="versus_2d",
                 labels=("index of points", "dependent variable"),
                 legend=["target"] + [str(model)],
                 filename="pdf")  ### do not leave inline comments with named parameters (!!?)
    plot.plotShow()



    ###########################  NEXT MODEL  ##############################
    model = GradientBoostingRegressor(loss="ls", random_state=0) # ‘ls’ refers to least squares regression
    model.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    error = calculateError(ytest, ypred)
    models.append(model)
    modelsLoss.append(error)

    ### VISUALIZATION ###
    matrices = []
    matrices.append( np.column_stack((xtest_uniform,ytest)) )
    matrices.append(np.column_stack((xtest_uniform,ypred)))

    plot.plotBegin(1, 1, filename="plot.pdf")
    plot.plotAddText("Gradiend Boosting. error=" +str(error))
    plot.addPlot(*matrices,
                 plottype="versus_2d",
                 labels=("index of points", "dependent variable"),
                 legend=["target"] + [str(model)],
                 filename="pdf")  ### do not leave inline comments with named parameters (!!?)
    plot.plotShow()




    ###########################  NEXT MODEL  ##############################
    model = RidgeCV(alphas=[1e-4, 1e-3, 1e-2, 1e-1, 1])       # cross-validated
    model.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    error = calculateError(ytest, ypred)
    #model.score(X, y)
    models.append(model)
    modelsLoss.append(error)

    ### VISUALIZATION ###
    matrices = []
    matrices.append( np.column_stack((xtest_uniform,ytest)) )
    matrices.append(np.column_stack((xtest_uniform,ypred)))

    plot.plotBegin(1, 1, filename="plot.pdf")
    plot.plotAddText("Ridge estimator with cross-validation. error=" +str(error))
    plot.addPlot(*matrices,
                 plottype="versus_2d",
                 labels=("index of points", "dependent variable"),
                 legend=["target"] + [str(model)],
                 filename="pdf")  ### do not leave inline comments with named parameters (!!?)
    plot.plotShow()



    ###########################  NEXT MODEL  ##############################
    model = DecisionTreeRegressor(criterion="mse", random_state=0)  # same metric as before
    #cross_val_score(model, X, y, cv=10)
    model.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    error = calculateError(ytest, ypred)
    #model.score(X, y)
    models.append(model)
    modelsLoss.append(error)

    ### VISUALIZATION ###
    matrices = []
    matrices.append( np.column_stack((xtest_uniform,ytest)) )
    matrices.append(np.column_stack((xtest_uniform,ypred)))

    plot.plotBegin(1, 1, filename="plot.pdf")
    plot.plotAddText("Decision tree. error=" +str(error))
    plot.addPlot(*matrices,
                 plottype="versus_2d",
                 labels=("index of points", "dependent variable"),
                 legend=["target"] + [str(model)],
                 filename="pdf")  ### do not leave inline comments with named parameters (!!?)
    plot.plotShow()



    ###########################  NEXT MODEL  ##############################
    if optimizeModels:
        parameters = { 'n_neighbors': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10) }
        ## cv: cross-validation 
        gridsearch = GridSearchCV(estimator=KNeighborsRegressor(),
                                  param_grid=parameters,
                                  cv=5,
                                  scoring='neg_mean_squared_error',
                                  verbose=1,
                                  n_jobs=-1)
        model = gridsearch.fit(Xtrain,ytrain)
        params = model.best_params_
        print("K-nn parameters found " + str(params))

    else:
        params = {"n_neighbors": 3}

    model = KNeighborsRegressor(**params).fit(Xtrain, ytrain)

    ypred = model.predict(Xtest)
    error = calculateError(ytest, ypred)
    #model.score(X, y)
    models.append(model)
    modelsLoss.append(error)

    ### VISUALIZATION ###
    matrices = []
    matrices.append( np.column_stack((xtest_uniform,ytest)) )
    matrices.append(np.column_stack((xtest_uniform,ypred)))

    plot.plotBegin(1, 1, filename="plot.pdf")
    plot.plotAddText("k-nn. error=" +str(error))
    plot.addPlot(*matrices,
                 plottype="versus_2d",
                 labels=("index of points", "dependent variable"),
                 legend=["target"] + [str(model)],
                 filename="pdf")  ### do not leave inline comments with named parameters (!!?)
    plot.plotShow()




    ###########################  NEXT MODEL  ##############################
    paramsFixed = {"random_state":1, "max_iter":2000, "activation": "tanh"}

    if optimizeModels:
        # for a small data set:
        parameters = { "batch_size":   (5, 10, 20, 40),
                       "hidden_layer_sizes":((32),(64),(128),(16,16),(32,32))    #, (64,64)]
                     }         # "epochs" :      (5,10,25),

        ## cv: cross-validation 
        gridsearch = GridSearchCV(estimator=MLPRegressor(**paramsFixed),
                                  param_grid=parameters,
                                  cv=5,
                                  scoring='neg_mean_squared_error',
                                  verbose=1,
                                  n_jobs=-1)
        model = gridsearch.fit(Xtrain,ytrain)
        params = model.best_params_
        print("K-nn parameters found " + str(params))
    else:
        params = {"hidden_layer_sizes":(128), "activation":"tanh"}
                         
    params = {**params, **paramsFixed}
        ### This merge is for Python >=3.5
        ### COMMENT: IN PYTHON 3.9:      {} = {} | {}
        ### keep in mind, the values are updated from left to right (right final if collision)

    model = MLPRegressor(**params)

    model.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    error = calculateError(ytest, ypred)
    #model.score(X, y)
    models.append(model)
    modelsLoss.append(error)

    ### VISUALIZATION ###
    matrices = []
    matrices.append( np.column_stack((xtest_uniform,ytest)) )
    matrices.append(np.column_stack((xtest_uniform,ypred)))

    plot.plotBegin(1, 1, filename="plot.pdf")
    plot.plotAddText("Multi-layer Perceptron. error=" +str(error))
    plot.addPlot(*matrices,
                 plottype="versus_2d",
                 labels=("index of points", "dependent variable"),
                 legend=["target"] + [str(model)],
                 filename="pdf")  ### do not leave inline comments with named parameters (!!?)
    plot.plotShow()


    iBest = np.argmin(modelsLoss)
    modelBest = models[iBest]
    print("getWinner() found model=", modelBest)
    return modelBest

# getWinner() end.


def updateModel(model, Xtrain, ytrain):

    model.fit(Xtrain,ytrain)

    Xtest = Xtrain         # no hold-out validation set
    ytest = ytrain

    ypred = model.predict(Xtest)
    error = calculateError(ytest, ypred)

    ### VISUALIZATION ###
    matrices = []
    xtest_uniform = range(len(ytest))     # index set to display elements in domain
    matrices.append( np.column_stack((xtest_uniform,ytest)) )
    matrices.append(np.column_stack((xtest_uniform,ypred)))


    plot.plotBegin(1, 1, filename="plot.pdf")
    plot.plotAddText("Final predictor. error=" +str(error))
    plot.addPlot(*matrices,
                 plottype="versus_2d",
                 labels=("index of points", "dependent variable"),
                 legend=["target"] + [str(model)],
                 filename="pdf")  ### do not leave inline comments with named parameters (!!?)
    plot.plotShow()


    print("updateModel() model=", model)

    return model

