#!/usr/bin/env python
# coding: utf-8

# Réseaux de neurones récurrents (RNN) : Les RNN peuvent être utilisés pour des tâches de régression où les données d'entrée sont des séquences temporelles ou des séquences en général.
# 
# Réseaux de neurones récurrents à mémoire à court terme (LSTM) : Ces réseaux sont particulièrement adaptés aux problèmes de régression impliquant des dépendances à long terme dans les séquences de données.
# 
# Réseaux de neurones récurrents à mémoire à court terme bidirectionnels (Bidirectional LSTM) : Ils sont utiles lorsque l'information contextuelle des deux directions temporelles est importante pour la régression.
# 
# Réseaux de neurones récurrents à mémoire à attention (Attention-based LSTM) : Ces réseaux sont bénéfiques lorsque l'attention sélective est requise pour traiter certaines parties de la séquence d'entrée.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use("seaborn")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Attention


# In[2]:


data_path = ''

data = pd.read_csv(f'{data_path}train.csv',delimiter=';',decimal=',',na_values=["#VALEUR!"],index_col="time")
data.index = pd.to_datetime(data.index, format='%d/%m/%Y %H:%M')

test = pd.read_csv(f'{data_path}test.csv',delimiter=';',decimal=',',na_values=["#VALEUR!"],index_col="time")
test.index = pd.to_datetime(test.index, format='%d/%m/%Y %H:%M')

print(data.info())


# In[3]:


chosen_strategy = 'median' # "mean" / "constant" / "most_frequent"
for col in data.columns:
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    data[col] = imp_mean.fit_transform(data[[col]]).squeeze()


# In[4]:


assert data.dtypes.equals(pd.Series(dict(zip(data.columns,["float64"]*len(data.columns)))))

assert test.dtypes.equals(pd.Series(dict(zip(test.columns,["float64"]*len(test.columns)))))

print(data.head())


# In[5]:


# Séparer la cible (Net Power) du reste des données
X = data.drop(columns=['Net Power (MW)'])
y = data['Net Power (MW)']

# Diviser les données en ensembles d'entraînement et de validation
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)


# In[11]:


def evaluate_model(model, X_train, X_eval, y_train, y_eval):
    model_train_mae = mean_absolute_error(y_train, model.predict(X_train))
    model_test_mae = mean_absolute_error(y_eval, model.predict(X_eval))
    return model_train_mae, model_test_mae
"""
print(X_train.info())
print(X_eval.info())
print(y_train.info())
print(y_eval.info())
"""
print(X_train.shape[0])
print(X_train.shape[1])


# In[14]:


# Modèle RNN
model_rnn = Sequential()
# Ajouter les couches RNN appropriées ici
# Par exemple :
model_rnn.add(LSTM(units=64, activation='relu', input_shape=(X_train.shape[0], X_train.shape[1])))
model_rnn.add(Dense(units=1, activation='linear'))  # Couche de sortie pour la régression

# Modèle LSTM
model_lstm = Sequential()
# Ajouter les couches LSTM appropriées ici
# Par exemple :
model_lstm.add(LSTM(units=128, activation='relu', input_shape=(X_train.shape[0], X_train.shape[1])))
model_lstm.add(Dense(units=1, activation='linear'))  # Couche de sortie pour la régression

# Modèle Bidirectional LSTM
model_bidirectional_lstm = Sequential()
# Ajouter les couches Bidirectional LSTM appropriées ici
# Par exemple :
model_bidirectional_lstm.add(Bidirectional(LSTM(units=128, activation='relu'), input_shape=(X_train.shape[0], X_train.shape[1])))
model_bidirectional_lstm.add(Dense(units=1, activation='linear'))  # Couche de sortie pour la régression
"""
# Modèle LSTM avec mémoire à attention
model_lstm_attention = Sequential()
# Ajouter les couches LSTM avec attention appropriées ici
# Par exemple :
model_lstm_attention.add(LSTM(units=64, activation='relu', return_sequences=True, input_shape=(X_train.shape[0], X_train.shape[1])))
model_lstm_attention.add(Attention())
model_lstm_attention.add(Dense(units=1, activation='linear'))  # Couche de sortie pour la régression
"""
# Compiler les modèles
model_rnn.compile(loss='mean_absolute_error', optimizer='adam')
model_lstm.compile(loss='mean_absolute_error', optimizer='adam')
model_bidirectional_lstm.compile(loss='mean_absolute_error', optimizer='adam')
#model_lstm_attention.compile(loss='mean_absolute_error', optimizer='adam')

# Entraîner les modèles
epochs = 100
batch_size = 128

model_rnn.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_eval, y_eval))
model_lstm.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_eval, y_eval))
model_bidirectional_lstm.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_eval, y_eval))
model_lstm_attention.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_eval, y_eval))


# In[ ]:


# Évaluer les modèles
rnn_train_mae, rnn_test_mae = evaluate_model(model_rnn, X_train, X_eval, y_train, y_eval)
lstm_train_mae, lstm_test_mae = evaluate_model(model_lstm, X_train, X_eval, y_train, y_eval)
bidirectional_lstm_train_mae, bidirectional_lstm_test_mae = evaluate_model(model_bidirectional_lstm, X_train, X_eval, y_train, y_eval)
lstm_attention_train_mae, lstm_attention_test_mae = evaluate_model(model_lstm_attention, X_train, X_eval, y_train, y_eval)

# Afficher les résultats
print("RNN - Train MAE: {:.2f}, Test MAE: {:.2f}".format(rnn_train_mae, rnn_test_mae))
print("LSTM - Train MAE: {:.2f}, Test MAE: {:.2f}".format(lstm_train_mae, lstm_test_mae))
print("Bidirectional LSTM - Train MAE: {:.2f}, Test MAE: {:.2f}".format(bidirectional_lstm_train_mae, bidirectional_lstm_test_mae))
print("LSTM with Attention - Train MAE: {:.2f}, Test MAE: {:.2f}".format(lstm_attention_train_mae, lstm_attention_test_mae))


# In[ ]:


from keras.models import save_model

# ... (Code d'entraînement des modèles)

# Enregistrer les modèles
model_rnn.save('model_rnn.h5')
model_lstm.save('model_lstm.h5')
model_bidirectional_lstm.save('model_bidirectional_lstm.h5')
model_lstm_attention.save('model_lstm_attention.h5')


# In[ ]:


"""from keras.models import load_model

# Charger les modèles depuis les fichiers
loaded_model_rnn = load_model('model_rnn.h5')
loaded_model_lstm = load_model('model_lstm.h5')
loaded_model_bidirectional_lstm = load_model('model_bidirectional_lstm.h5')
loaded_model_lstm_attention = load_model('model_lstm_attention.h5')
"""


# Pour choisir des hyperparamètres appropriés pour l'entraînement des modèles, il est important de prendre en compte la taille du jeu de données, la complexité du modèle, les caractéristiques des données et les ressources matérielles disponibles. Voici quelques suggestions pour des hyperparamètres appropriés, accompagnées d'explications sur les raisons de ces choix :
# 
# 1. **Nombre d'époques (epochs)** :
#    - Pour les RNN et les LSTM, commencer avec un nombre d'époques relativement faible (par exemple, entre 10 et 50) pour éviter le surapprentissage initial.
#    - Observer la courbe de perte (loss) sur l'ensemble de validation au fil des époques. Si la perte continue à diminuer de manière significative, continuez à augmenter le nombre d'époques jusqu'à ce que la perte de validation commence à augmenter, indiquant un possible surapprentissage.
# 
# 2. **Taille de lot (batch_size)** :
#    - Pour de petits ensembles de données, une taille de lot plus petite (par exemple, entre 32 et 128) est préférable pour une convergence plus rapide et une meilleure généralisation.
#    - Pour de grands ensembles de données, une taille de lot plus grande (par exemple, entre 128 et 512) peut être utilisée pour accélérer l'entraînement, mais cela nécessite également plus de mémoire.
# 
# 3. **Taux d'apprentissage (learning rate)** :
#    - Pour Adam, le taux d'apprentissage par défaut est généralement de 0.001, ce qui peut être un bon point de départ. Vous pouvez essayer des valeurs plus petites (par exemple, 0.0001) si vous avez un grand ensemble de données ou des valeurs plus grandes (par exemple, 0.01) si vous avez un petit ensemble de données.
#    - Vous pouvez également utiliser des techniques d'ajustement automatique du taux d'apprentissage, telles que le taux d'apprentissage adaptatif (Adaptive Learning Rate), qui ajuste automatiquement le taux d'apprentissage en fonction de la performance du modèle.
# 
# 4. **Fonction d'activation** :
#    - Pour les couches cachées des RNN et LSTM, la fonction d'activation ReLU (Rectified Linear Unit) est couramment utilisée car elle permet de surmonter le problème de disparition des gradients et d'accélérer l'entraînement.
#    - Pour la couche de sortie dans une tâche de régression, une fonction d'activation linéaire est appropriée car elle permet au modèle de prédire des valeurs continues sans contraintes.
# 
# 5. **Nombre de couches cachées** :
#    - Pour les petits ensembles de données, commencez avec une seule couche cachée pour éviter le surapprentissage.
#    - Pour les ensembles de données plus grands et plus complexes, vous pouvez expérimenter avec plusieurs couches cachées, en gardant à l'esprit que des modèles plus profonds nécessitent généralement plus de données pour éviter le surapprentissage.
# 
# 6. **Nombre de neurones par couche cachée** :
#    - Pour les RNN et LSTM, commencer avec un nombre de neurones relativement faible (par exemple, entre 64 et 128) pour éviter le surapprentissage initial.
#    - Augmenter progressivement le nombre de neurones si nécessaire pour une meilleure performance.
# 
# 7. **Dropout** :
#    - Utiliser des couches de dropout pour régulariser le modèle et éviter le surapprentissage. Commencer avec un taux de dropout de 0.2 à 0.5 pour les couches cachées.
# 
# 8. **Fonction de perte** :
#    - Pour les problèmes de régression, la perte "mean_absolute_error" (MAE) est appropriée car elle mesure l'écart absolu moyen entre les prédictions du modèle et les vraies valeurs.
# 
# 9. **Fonction d'attention** (pour les modèles LSTM avec mémoire à attention) :
#    - Utiliser la fonction d'attention "softmax" pour permettre au modèle de se concentrer sélectivement sur différentes parties de la séquence d'entrée.
# 
# Il est important de noter que l'exploration d'hyperparamètres peut être un processus itératif et que les valeurs optimales peuvent varier en fonction du problème spécifique et des données. Vous pouvez utiliser la validation croisée (cross-validation) pour évaluer différentes combinaisons d'hyperparamètres et choisir celle qui donne les meilleures performances sur l'ensemble de validation.

# Expliquation de chaque ligne de code :
# 
# ```python
# from sklearn.metrics import mean_absolute_error
# from keras.models import Sequential
# from keras.layers import LSTM, Dense, Bidirectional, Attention
# ```
# - Dans cette partie, nous importons les bibliothèques nécessaires : `mean_absolute_error` de `sklearn.metrics` pour calculer l'erreur absolue moyenne (MAE) pour évaluer la performance du modèle, et `Sequential`, `LSTM`, `Dense`, `Bidirectional` et `Attention` de `keras.models` et `keras.layers` pour construire les modèles de réseaux de neurones.
# 
# ```python
# # Modèle RNN
# model_rnn = Sequential()
# # Ajouter les couches RNN appropriées ici
# # Par exemple :
# model_rnn.add(LSTM(units=64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
# model_rnn.add(Dense(units=1, activation='linear'))  # Couche de sortie pour la régression
# ```
# - Nous commençons par créer un modèle séquentiel `model_rnn` pour le réseau de neurones récurrent (RNN). Nous ajoutons une couche LSTM avec 64 neurones et une fonction d'activation ReLU. La couche d'entrée a une forme `(X_train.shape[1], X_train.shape[2])`, qui correspond au nombre de caractéristiques (`X_train.shape[1]`) et à la taille des séquences (`X_train.shape[2]`).
# - Nous ajoutons ensuite une couche Dense avec 1 neurone et une fonction d'activation linéaire. Cette couche de sortie est utilisée pour la régression car nous essayons de prédire une valeur continue (la consommation nette d'énergie).
# 
# ```python
# # Modèle LSTM
# model_lstm = Sequential()
# # Ajouter les couches LSTM appropriées ici
# # Par exemple :
# model_lstm.add(LSTM(units=128, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
# model_lstm.add(Dense(units=1, activation='linear'))  # Couche de sortie pour la régression
# ```
# - Ici, nous créons un modèle séquentiel `model_lstm` pour le réseau de neurones LSTM. Nous ajoutons une couche LSTM avec 128 neurones et une fonction d'activation ReLU. La couche d'entrée a la même forme que précédemment.
# - Nous ajoutons également une couche Dense de sortie avec 1 neurone et une fonction d'activation linéaire.
# 
# ```python
# # Modèle Bidirectional LSTM
# model_bidirectional_lstm = Sequential()
# # Ajouter les couches Bidirectional LSTM appropriées ici
# # Par exemple :
# model_bidirectional_lstm.add(Bidirectional(LSTM(units=128, activation='relu'), input_shape=(X_train.shape[1], X_train.shape[2])))
# model_bidirectional_lstm.add(Dense(units=1, activation='linear'))  # Couche de sortie pour la régression
# ```
# - Ici, nous créons un modèle séquentiel `model_bidirectional_lstm` pour le réseau de neurones LSTM bidirectionnel (Bidirectional LSTM). Nous enveloppons la couche LSTM dans la couche Bidirectional pour obtenir un effet bidirectionnel.
# - La couche LSTM a 128 neurones et une fonction d'activation ReLU. La couche d'entrée a également la même forme que précédemment.
# - Nous ajoutons ensuite une couche Dense de sortie avec 1 neurone et une fonction d'activation linéaire.
# 
# ```python
# # Modèle LSTM avec mémoire à attention
# model_lstm_attention = Sequential()
# # Ajouter les couches LSTM avec attention appropriées ici
# # Par exemple :
# model_lstm_attention.add(LSTM(units=64, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
# model_lstm_attention.add(Attention())
# model_lstm_attention.add(Dense(units=1, activation='linear'))  # Couche de sortie pour la régression
# ```
# - Ici, nous créons un modèle séquentiel `model_lstm_attention` pour le réseau de neurones LSTM avec mémoire à attention (Attention-based LSTM). La couche LSTM est configurée avec `return_sequences=True` pour que nous puissions utiliser la couche d'attention après.
# - La couche LSTM a 64 neurones et une fonction d'activation ReLU. La couche d'entrée a la même forme que précédemment.
# - Nous ajoutons ensuite une couche d'attention qui permet au modèle de se concentrer sélectivement sur différentes parties de la séquence d'entrée.
# - Enfin, nous ajoutons une couche Dense de sortie avec 1 neurone et une fonction d'activation linéaire.
# 
# ```python
# # Compiler les modèles
# model_rnn.compile(loss='mean_absolute_error', optimizer='adam')
# model_lstm.compile(loss='mean_absolute_error', optimizer='adam')
# model_bidirectional_lstm.compile(loss='mean_absolute_error', optimizer='adam')
# model_lstm_attention.compile(loss='mean_absolute_error', optimizer='adam')
# ```
# - Pour chaque modèle, nous compilons le modèle avec une fonction de perte (loss) "mean_absolute_error" pour évaluer l'écart absolu moyen entre les prédictions et les vraies valeurs pour les problèmes de régression. L'optimiseur utilisé est "adam", qui est un optimiseur populaire et efficace.
# 
# ```python
# # Entraîner les modèles
# epochs = 100
# batch_size = 128
# 
# model_rnn.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_eval, y_eval))
# model_lstm.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_eval, y_eval))
# model_bidirectional_lstm.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_eval, y_eval))
# model_lstm_attention.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_eval, y_eval))
# ```
# - Nous entraînons chaque modèle avec `epochs=100` et `batch_size=128`. Cela signifie que chaque modèle effectuera 100 itérations complètes sur l'ensemble des données (époques) et mettra à jour les poids du modèle après chaque lot de 128 exemples d'entraînement.
# - Nous utilisons `X_train` et `y_train` comme données d'entraînement, et `X_eval` et `y_eval` comme données de validation. Cela nous permettra d'évaluer la performance des modèles sur l'ensemble de validation pendant l'entraînement.
# 
# Cependant, il est important de noter que ces hyperparamètres sont à titre indicatif et peuvent nécessiter des ajustements en fonction de la taille de votre jeu de données, de la complexité de la tâche et des performances réelles du modèle lors de l'entraînement. Vous pouvez essayer différentes valeurs d'epochs et de batch
# 
# _size, ainsi que d'autres hyperparamètres tels que le taux d'apprentissage, le nombre de neurones par couche, etc., pour trouver les valeurs qui donnent de bonnes performances sur votre problème spécifique. La recherche d'hyperparamètres est un processus itératif et expérimental qui nécessite souvent plusieurs essais et évaluations.
