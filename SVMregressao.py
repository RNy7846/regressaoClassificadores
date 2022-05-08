import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, confusion_matrix, accuracy_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split,cross_val_score, KFold


dataset = pd.read_csv('C:/Users/roney/OneDrive/Área de Trabalho/Faculdade/python/MLP/ceara.csv')


labels = dataset.columns[:13]

x = dataset[labels]
#casos confirmados
y = dataset['new_confirmed']


#definindo o treino e teste a partir do dataset
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=0)

#Crie o objeto de regressão logística
regressor = SVR(kernel = 'rbf')
#Treina o modelo usando os dados de treino e confere o score
regressor.fit(xTrain, yTrain)
correlacao = regressor.score(xTrain, yTrain)

'''
kfold  = KFold(n_splits=10, shuffle=True)
result = cross_val_score(regressor, x, y, cv = kfold)
print("K-Fold: {0}\n".format(result))
print("Media K-Fold: {0}".format(result.mean()))
'''
previsoes = regressor.predict(xTest)
#erro absoluto
error = mean_absolute_error(yTest, previsoes)
print('erro absoluto = %.4f' % error)

regressor.score(xTest, yTest)
