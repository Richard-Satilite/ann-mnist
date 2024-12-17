
#NOVA MODIFICAÇÃO FEITA NO CLONE

# ANN MNIST CLASSIFICATION
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#
mnist = tf.keras.datasets.mnist


#Obtendo a tupla de arrays numpy
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train, x_test = x_train / 255.0, x_test / 255.0

print(x_train.shape)
print(x_test.shape)

"""# CARREGANDO MODELO"""

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(10, activation = 'softmax') #NÚMERO DE CLASSIFICAÇÕES
])

model.summary()

"""# COMPILANDO E TREINANDO MODELO"""

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) #Obtendo acuárcia

r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=25)

"""# PLOTANDO GRÁFICO DE PERDAS E ACURÁCIAS"""

plt.plot(r.history['loss'], label='perdas')
plt.plot(r.history['val_loss'], label='valor_perda')
plt.legend()

plt.plot(r.history['accuracy'], label='acurácia')
plt.plot(r.history['val_accuracy'], label='valor_acurácia')
plt.legend()

"""# AVALIAÇÃO DO MODELO"""

print(model.evaluate(x_test, y_test))

"""# AVALIANDO PERFORMANCE COM MATRIX DE ERRO"""

from sklearn.metrics import confusion_matrix
import itertools

def matriz_erro(cm, classes, nomalize = False, title = 'Matriz de Erro', cmap = plt.cm.Blues):
    if nomalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print('Matriz normalizada')
    else:
        print('Matriz não normalizada')

    print(cm)
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if nomalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt)),
        horizontalalignment = 'center',
        color = 'white' if cm[i, j] > thresh else 'black'

    plt.tight_layout()
    plt.ylabel('Verdadeiro')
    plt.xlabel('Previsto')
    plt.show()

p_teste = model.predict(x_test).argmax(axis = 1)
cm = confusion_matrix(y_test, p_teste)
matriz_erro(cm, range(10))

"""# VERIFICANDO IMAGENS DE ERRO"""

misclassified_idx = np.where(p_teste != y_test)[0]
i = np.random.choice(misclassified_idx)
plt.imshow(x_test[i], cmap = 'gray')
plt.title('Verdadeiro: %s Previsto: %s' % (y_test[i], p_teste[i]))