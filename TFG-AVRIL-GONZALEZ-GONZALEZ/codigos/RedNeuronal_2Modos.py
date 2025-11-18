import os 
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no info, 2=no warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# 1. Función para generar las etiquetas en el nuevo formato
def generate_new_labels(labels):
    new_labels = []
    for label in labels:
        l1, m1, w1, l2, m2, w2 = label
        l1, m1, l2, m2 = int(round(l1*5)), int(round(m1*5)), int(round(l2*5)), int(round(m2*5))
        
        # --- Ordenar los modos para evitar redundancias ---
        mode1 = (l1, m1)
        mode2 = (l2, m2)
        if mode1 > mode2:  # Ordenar por (l,m) ascendente
            mode1, mode2 = mode2, mode1
            w1, w2 = w2, w1  # Ajustar pesos correspondientes
        
        # Crear one-hot
        mode_vector = np.zeros(36)
        idx1 = mode1[0] * 6 + mode1[1]
        idx2 = mode2[0] * 6 + mode2[1]
        mode_vector[idx1] = 1.0
        mode_vector[idx2] = 1.0
        
        new_label = np.concatenate([mode_vector, [w1, w2]])
        new_labels.append(new_label)
    
    return np.array(new_labels)
#lo que hago aqui es crar vectores formato one hot para cada superposicion de tal forma que los vectores 
#tendran (0,0,0,1,0,...,1,0,...,w1,w2) donde los unicos elementos distintos de 0 serán los correspondientes
#a los modos presentes en la superposicion y sus correspondientes pesos.
#ademas tengo en cuenta que combinar (3,1) con (2,5) es lo mismo que combinar (2,5) con (3,1)

# 2. Función de pérdida personalizada
def custom_loss(y_true, y_pred):
    # 1. Separar modos y pesos
    y_true_modes = y_true[:, :36]
    y_true_weights = y_true[:, 36:]
    y_pred_modes = y_pred[:, :36]
    y_pred_weights = y_pred[:, 36:]
    
    # 2. Pérdida estándar (clasificación + regresión)
    mode_loss = tf.keras.losses.binary_crossentropy(y_true_modes, y_pred_modes)
    
    # --- NUEVO: solo calcular el error de pesos si los modos predichos son correctos ---
    top2_pred_idx = tf.argsort(y_pred_modes, axis=1)[:, -2:]
    top2_true_idx = tf.argsort(y_true_modes, axis=1)[:, -2:]

    # Comparar si los índices predichos y verdaderos coinciden (sin importar orden)
    def unordered_match(a, b):
        return tf.reduce_all(tf.sort(a, axis=-1) == tf.sort(b, axis=-1), axis=-1)

    correct_modes = tf.cast(unordered_match(top2_pred_idx, top2_true_idx), tf.float32)
    
    weight_loss = tf.keras.losses.MSE(y_true_weights, y_pred_weights)
    weight_loss = correct_modes * weight_loss  # solo cuenta si los modos son correctos

    # 3. Penalización por superposición de un modo consigo mismo
    top2_modes = tf.math.top_k(y_pred_modes, k=2).indices
    l1_pred, m1_pred = top2_modes[:, 0] // 6, top2_modes[:, 0] % 6
    l2_pred, m2_pred = top2_modes[:, 1] // 6, top2_modes[:, 1] % 6
    
    same_mode = tf.cast(
        tf.logical_and(
            tf.equal(l1_pred, l2_pred),
            tf.equal(m1_pred, m2_pred)
        ),
        tf.float32
    )
    
    penalty = same_mode * 20.0

    # 4. Combinar todas las pérdidas
    return mode_loss + 0.5 * weight_loss + penalty

#combina dos tipos de perdidas: binary crossentropy para la clasificacion de modos con one hot
#y mse para la regresion d epesos (los fallos en pesos tienen la mitad de importancia)
#ademas penalizamos *10 cuando predice la combinacion de un modo consigo mismo

def mae_pesos(y_true, y_pred):
    y_true_weights = y_true[:, 36:]  # Solo pesos
    y_pred_weights = y_pred[:, 36:]  # Solo pesos
    return tf.reduce_mean(tf.abs(y_true_weights - y_pred_weights))


def std_metric(y_true,y_pred):
    y_true = y_true[:, 36:]  # Solo pesos
    y_pred = y_pred[:, 36:]  # Solo pesos
    error=y_true-y_pred
    mean_error=tf.reduce_mean(error)
    squared_diff=tf.square(error-mean_error)
    variance=tf.reduce_mean(squared_diff)
    std_dev=tf.sqrt(variance)
    return std_dev

def acc_modos(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true = tf.round(y_true[:, :36])
    pred = tf.round(y_pred[:, :36])
    return tf.reduce_mean(tf.cast(tf.equal(true, pred), tf.float32))

# 3. Carga y preparación de datos
data_folder = "superposiciones"
images = []
old_labels = []

for filename in os.listdir(data_folder):
    if filename.endswith(".png"):
        img = Image.open(os.path.join(data_folder, filename)).convert('L')
        images.append(np.array(img) / 255.0)
        
        parts = filename.replace(".png", "").split("_")
        old_labels.append([
            float(parts[1])/5, float(parts[2])/5, float(parts[3]),
            float(parts[4])/5, float(parts[5])/5, float(parts[6])
        ])

images = np.array(images)
old_labels = np.array(old_labels)
labels = generate_new_labels(old_labels)

# Separación original
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Ahora saca validación desde el conjunto de entrenamiento
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42)
# → 0.125 = 0.125 * 0.8 = 0.1 total → Queda 70% train, 10% val, 20% test

X_train = X_train.reshape(-1, 64, 64, 1)
X_test = X_test.reshape(-1, 64, 64, 1)
X_val=X_val.reshape(-1,64,64,1)

# ================
# Visualización de imágenes aleatorias para ver como las procesa el programa
plt.figure(figsize=(15, 5))
num_samples = 5
random_indices = np.random.choice(len(X_train), num_samples, replace=False)

for i, idx in enumerate(random_indices):
    plt.subplot(1, num_samples, i+1)
    # Añade vmin y vmax para mantener la escala global
    plt.imshow(
        X_train[idx].reshape(64, 64), 
        cmap='inferno', 
        vmin=0, 
        vmax=1  # Esto es crítico
    )
    plt.axis('off')
    
    true_modes = y_train[idx][:36]
    true_weights = y_train[idx][36:]
    true_idx = np.where(true_modes > 0.5)[0]
    l1, m1 = divmod(true_idx[0], 6)
    l2, m2 = divmod(true_idx[1], 6)
    w1, w2 = true_weights[0], true_weights[1]
    
    plt.title(f"({l1},{m1},{w1:.2f})\n({l2},{m2},{w2:.2f})", fontsize=8)

plt.suptitle("Ejemplos aleatorios (Escala global: 0-1)", y=1.05)
plt.tight_layout()
plt.show()

#=====================================

from collections import Counter

# Ver combinaciones únicas de modos en las etiquetas originales e ignorar orden
combinations = []
for label in old_labels:
    l1, m1, _, l2, m2, _ = label
    mode1 = (int(round(l1*5)), int(round(m1*5)))
    mode2 = (int(round(l2*5)), int(round(m2*5)))
    combination = frozenset({mode1, mode2})  # Usar frozenset para ignorar orden
    combinations.append(combination)

# Crear el contador
contador = Counter(combinations)

# Imprimir solo las combinaciones que aparecen más de una vez
#for combo, count in contador.items():
#    if count > 1:
#        print(f"{dict(combo)}: aparece {count} veces")

# 4. Modelo
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    #Dense(128, activation='relu'),
    #Dropout(0.4),
    Dense(38, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate= 0.0001), loss=custom_loss, metrics=['mae',acc_modos,mae_pesos,std_metric])
#prueba cambiar a 0.00005
#==================
# Verifica que todas las imágenes usan la misma escala
print("Máximo en X_train:", np.max(X_train))  # Debe ser ~1.0
print("Mínimo en X_train:", np.min(X_train))  # Debe ser ~0.0

# Verifica que la red no altera la escala
#sample = X_train[:1]  # Toma una imagen de ejemplo
#pred = model.predict(sample)
#print("Máximo después de la primera capa:", np.max(model.layers[0](sample)))



#==================



# 5. Entrenamiento por 200 épocas
history = model.fit(
    X_train, y_train,
    epochs=200,  # Cambiar a 150 epocas o poner un call early stopping
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
)

# 6. Evaluación del error medio total
test_loss, test_mae, ACCmodos, MAEpesos, DESVpesos = model.evaluate(X_test, y_test, verbose=0)
print(f'\nPérdida total en el conjunto de prueba (pesos+modos): {test_loss:.4f}')
print(f'Error medio absoluto en el conjunto de prueba (pesos+modos): {test_mae:.4f}')
print(f'Acc modos en el conjunto de prueba: {ACCmodos:.4f}')
print(f'MAE pesos en el conjunto de prueba: {MAEpesos:.4f}')
print(f'Desviación estandar pesos en el conjunto de prueba: {DESVpesos:.4f}')
# 7. Gráficos
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Evolución de la Pérdida')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Entrenamiento')
plt.plot(history.history['val_mae'], label='Validación')
plt.title('Evolución del Error Medio Absoluto')
plt.xlabel('Épocas')
plt.ylabel('MAE')
plt.legend()
plt.tight_layout()
plt.show()

# 8. Función para visualizar ejemplos
def plot_examples(i1, i2):
    predictions = model.predict(X_test)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    for i, idx in enumerate([i1, i2]):
        ax[i].imshow(X_test[idx].reshape(64, 64), cmap='gray')
        ax[i].axis('off')
        
        # Procesar predicción
        pred_modes = predictions[idx][:36]
        pred_weights = predictions[idx][36:]
        top2_idx = np.argsort(pred_modes)[-2:]
        l1, m1 = divmod(top2_idx[1], 6)
        l2, m2 = divmod(top2_idx[0], 6)
        w1, w2 = pred_weights[0], pred_weights[1]
        w1, w2 = w1/(w1+w2), w2/(w1+w2)
        
        # Valores reales
        true_modes = y_test[idx][:36]
        true_weights = y_test[idx][36:]
        true_idx = np.where(true_modes > 0.5)[0]
        rl1, rm1 = divmod(true_idx[0], 6)
        rl2, rm2 = divmod(true_idx[1], 6)
        rw1, rw2 = true_weights[0], true_weights[1]
        
        ax[i].text(2, 60, f'Real: ({rl1},{rm1},{rw1:.2f}) ({rl2},{rm2},{rw2:.2f})',
                 color='yellow', fontsize=10, bbox=dict(facecolor='black', alpha=0.5))
        ax[i].text(2, 56, f'Pred: ({l1},{m1},{w1:.2f}) ({l2},{m2},{w2:.2f})',
                 color='red', fontsize=10, bbox=dict(facecolor='black', alpha=0.5))
        
        print(f"Imagen {idx} - Real: ({rl1},{rm1},{rw1:.2f}) ({rl2},{rm2},{rw2:.2f})")
        print(f"Predicción: ({l1},{m1},{w1:.2f}) ({l2},{m2},{w2:.2f})\n")
    
    plt.show()

# 9. Ejemplos de visualización
plot_examples(58, 199)
