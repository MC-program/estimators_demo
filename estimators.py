print(__doc__)

import numpy as np

### grid search
from sklearn.model_selection import GridSearchCV

### single method with kernel/grid search
from sklearn.neighbors import KNeighborsRegressor   # knn, dont get confused with
                                                    # k-means clustering (unsupervised)
from sklearn.svm import SVR
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
### ensemble method / meta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

from plot import plot

# #############################################################################
# Generate sample data
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - np.random.rand(8))        # noise

# #############################################################################
# Fit regression model
#svr_rbf   = SVR(kernel='rbf',   C=100, gamma=0.1, epsilon=.1)
svr_rbf   = SVR(kernel='rbf',   C=10, gamma='auto', epsilon=.1)
svr_lin   = SVR(kernel='linear',C=10, gamma='auto')
svr_poly  = SVR(kernel='poly',  C=10, gamma='auto', degree=3, epsilon=.1, coef0=1)
#svr_sig   = SVR(kernel='sigmoid', C=10, gamma='auto', degree=3, epsilon=.1, coef0=1)
svr_final  = SVR(kernel='rbf',  C=1, gamma=0.001, epsilon=.00001)      # winner

svrs         = [svr_rbf, svr_lin, svr_poly]
kernel_label = ['RBF', 'Linear', 'Polynomial']



def calculateError(x,y):
    """ Error for mean value

    L2 norm / metric
    """
    diff = np.array(x) - np.array(y)
    dist = diff**2
    l2   = sum(dist)/len(diff)
    #print("calculateError(): diff=", diff, "l2=", l2)
    return l2


def getWinner(X=X, y=y, test_size=20):

    models = []  
    losses = []  # to be ranked


    if test_size>0:   # make a validation set for us
                     # even the methods cross-validate?
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size, random_state=40)
    else:
        Xtrain, Xtest, ytrain, ytest = X, X, y, y

    print("training/test set is " +str(len(ytrain)) +"/"+ str(len(ytest)))

    xtest_uniform = range(len(ytest))     # index set to display elements in domain

    ### VISUALIZATION ###
    plot.plotBegin(1, 1, filename="plot.pdf")

    ### First method    .............................
    plot.plotAddText("Method = Support Vector Machine (SVM)")


    """
    matrices = []

    matrices.append( np.column_stack((xtest_uniform,ytest)) )

    for ix, svr in enumerate(svrs):

        model = svr.fit(Xtrain, ytrain)
        ypred = model.predict(Xtest)

        matrices.append(np.column_stack((xtest_uniform,ypred)))

    plot.addPlot(*matrices,
                 plottype="versus_2d",
                 labels=("index of points", "dependent variable"),
                 legend=["target"] + kernel_label,
                 filename="pdf")  ### do not leave inline comments with named parameters (!!?)
    #plot.plotShow()

    matrices = []

    matrices.append( np.column_stack((xtest_uniform,list([0]*len(ytest)))) )

    for ix, svr in enumerate(svrs):

        model = svr.fit(Xtrain, ytrain)
        ypred = abs(model.predict(Xtest) - ytest)        # distanc in R1

        matrices.append(np.column_stack((xtest_uniform,ypred)))

    #print(matrices)
    #print(*matrices)

    plot.addPlot(*matrices,
                 plottype="versus_2d",
                 labels=("index of points", "dependent variable [absolut distance]"),
                 legend=["target"] + kernel_label,
                 filename="pdf")  ### do not leave inline comments with named parameters (!!?)
    plot.plotShow()
    """


    #######################################################################
    ### grid search for hyper parameters

    """
    parameters = {'kernel': ('linear', 'rbf','poly'), 
                  'C': [0.1, 0.5, 1, 5, 10],
                  'epsilon': [0.00001, 0.0001, 0.001, 0.01, 0.1],
                  'gamma': [0.0001, 0.001, 0.005, 0.1, 1]
                 }
    ## cv: cross-validation 
    gridsearch = GridSearchCV(estimator=SVR(),
                              param_grid=parameters,
                              cv=5,
                              scoring='neg_mean_squared_error',
                              verbose=1,
                              n_jobs=-1)
    # {'C': 1, 'epsilon': 0.00001, 'gamma': 0.001, 'kernel': 'rbf'}
    model = gridsearch.fit(Xtrain,ytrain)
    #print("SVR Parameters found " + str(gridsearch.best_params_))
    print("SVR Parameters found " + str(model.best_params_))
    """
    model = svr_final.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    error = calculateError(ytest, ypred)
    models.append(model)
    losses.append(error)

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



    #######################################################################
    model = GradientBoostingRegressor(loss="ls", random_state=0) # ‘ls’ refers to least squares regression
    model.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    error = calculateError(ytest, ypred)
    models.append(model)
    losses.append(error)

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





    #######################################################################
    model = RidgeCV(alphas=[1e-4, 1e-3, 1e-2, 1e-1, 1])       # cross-validated
    model.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    error = calculateError(ytest, ypred)
    #model.score(X, y)
    models.append(model)
    losses.append(error)

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





    #######################################################################
    model = DecisionTreeRegressor(criterion="mse", random_state=0)  # same metric as before
    #cross_val_score(model, X, y, cv=10)
    model.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    error = calculateError(ytest, ypred)
    #model.score(X, y)
    models.append(model)
    losses.append(error)

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





    #######################################################################
    '''
    parameters = { 'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9] }
    ## cv: cross-validation 
    gridsearch = GridSearchCV(estimator=KNeighborsRegressor(),
                              param_grid=parameters,
                              cv=5,
                              scoring='neg_mean_squared_error',
                              verbose=1,
                              n_jobs=-1)
    model = gridsearch.fit(Xtrain,ytrain)
    print("k-nn parameter(s) found " + str(model.best_params_))  ### k=5
    '''
    #model = KNeighborsRegressor(n_neighbors=2)
    model = KNeighborsRegressor(n_neighbors=5).fit(Xtrain, ytrain)

    ypred = model.predict(Xtest)
    error = calculateError(ytest, ypred)
    #model.score(X, y)
    models.append(model)
    losses.append(error)

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




    #######################################################################
    model = MLPRegressor(hidden_layer_sizes=(256), activation="tanh",
                         random_state=1, max_iter=4000)
           # reul, sigmoid, tanh
    model.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    error = calculateError(ytest, ypred)
    #model.score(X, y)
    models.append(model)
    losses.append(error)

    matrices = []
    matrices.append( np.column_stack((xtest_uniform,ytest)) )
    matrices.append(np.column_stack((xtest_uniform,ypred)))


    ### VISUALIZATION ###
    plot.plotBegin(1, 1, filename="plot.pdf")
    plot.plotAddText("Multi-layer Perceptron. error=" +str(error))
    plot.addPlot(*matrices,
                 plottype="versus_2d",
                 labels=("index of points", "dependent variable"),
                 legend=["target"] + [str(model)],
                 filename="pdf")  ### do not leave inline comments with named parameters (!!?)
    plot.plotShow()


    iBest = np.argmin(losses)
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

