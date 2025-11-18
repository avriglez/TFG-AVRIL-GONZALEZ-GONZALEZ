import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

optimizer = Adam(learning_rate=0.001)


# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 128 #128 el bueno
DATASET_PATH = "superposiciones"  
EPOCHS = 100
BATCH_SIZE = 32

# -----------------------------
# FUNCIONES DE UTILIDAD
# -----------------------------
def parse_filename(filename):
    name = os.path.splitext(filename)[0]
    parts = name.split('_')[1:]

    mode_mask = np.zeros(36)
    weights = []

    seen_modes = set()

    for i in range(0, 9, 3):
        l = int(parts[i])
        m = int(parts[i+1])
        w = float(parts[i+2])

        if not (0 <= l <= 5 and 0 <= m <= 5):
            raise ValueError(f"Modo fuera de rango permitido (0-5): ({l}, {m}) en {filename}")
        if (l, m) in seen_modes:
            raise ValueError(f"Modo repetido ({l}, {m}) en {filename}")
        seen_modes.add((l, m))

        idx = l * 6 + m  # Modo (l,m) se codifica como índice 6*l + m
        mode_mask[idx] = 1.0
        weights.append(w)

    return np.concatenate([mode_mask, weights])


def load_data(path):
    X, y = [], []
    for fname in os.listdir(path):
        if fname.endswith('.png'):
            img = load_img(os.path.join(path, fname), color_mode='grayscale', target_size=(IMG_SIZE, IMG_SIZE))
            img = img_to_array(img) / 255.0
            X.append(img)
            label = parse_filename(fname)
            y.append(np.array(label).flatten())
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
def mae_pesos(y_true, y_pred):
    y_true_weights = y_true[:, 36:]  # Solo pesos
    y_pred_weights = y_pred[:, 36:]  # Solo pesos
    return tf.reduce_mean(tf.abs(y_true_weights - y_pred_weights))

def acc_modos(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true = tf.round(y_true[:, :36])
    pred = tf.round(y_pred[:, :36])
    return tf.reduce_mean(tf.cast(tf.equal(true, pred), tf.float32))

def std_metric(y_true,y_pred):
    y_true = y_true[:, 36:]  # Solo pesos
    y_pred = y_pred[:, 36:]  # Solo pesos
    error=y_true-y_pred
    mean_error=tf.reduce_mean(error)
    squared_diff=tf.square(error-mean_error)
    variance=tf.reduce_mean(squared_diff)
    std_dev=tf.sqrt(variance)
    return std_dev

# 2. Función de pérdida personalizada
def custom_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # Separar modos y pesos
    y_true_modes = y_true[:, :36]
    y_true_weights = y_true[:, 36:]
    y_pred_modes = y_pred[:, :36]
    y_pred_weights = y_pred[:, 36:]

    # Pérdida de clasificación (binary crossentropy)
    mode_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true_modes, y_pred_modes))

    # Comparar top 3 modos predichos con los reales
    top3_pred_idx = tf.argsort(y_pred_modes, axis=1)[:, -3:]
    top3_true_idx = tf.argsort(y_true_modes, axis=1)[:, -3:]

    def unordered_match(a, b):
        return tf.reduce_all(tf.sort(a, axis=-1) == tf.sort(b, axis=-1), axis=-1)

    correct_modes = tf.cast(unordered_match(top3_pred_idx, top3_true_idx), tf.float32)

    # Pérdida de pesos (solo si modos son correctos)
    weight_loss = tf.reduce_mean(tf.keras.losses.MSE(y_true_weights, y_pred_weights))
    weight_loss = correct_modes * weight_loss

    # Penalización por repetir el mismo modo (misma combinación l,m en top 3)
    lms_top3 = tf.stack([top3_pred_idx // 6, top3_pred_idx % 6], axis=-1)  # shape: (batch, 3, 2)
    
    # Comparar pares de modos para detectar duplicados
    def is_same(a, b):
        return tf.reduce_all(tf.equal(a, b), axis=-1)
    
    same12 = is_same(lms_top3[:, 0], lms_top3[:, 1])
    same13 = is_same(lms_top3[:, 0], lms_top3[:, 2])
    same23 = is_same(lms_top3[:, 1], lms_top3[:, 2])
    
    same_mode = tf.cast(tf.logical_or(tf.logical_or(same12, same13), same23), tf.float32)
    penalty = same_mode * 10.0  # Puedes ajustar el peso de la penalización

    return mode_loss + 0.5 * weight_loss + penalty


#combina dos tipos de perdidas: binary crossentropy para la clasificacion de modos con one hot
#y mse para la regresion d epesos (los fallos en pesos tienen la mitad de importancia)
#ademas penalizamos *10 cuando predice la combinacion de un modo consigo mismo



def create_model():
    input_layer = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    # Modelo más liviano
    #x = layers.Conv2D(16, (3, 3), activation='relu')(input_layer)
    #x = layers.MaxPooling2D((2, 2))(x)
    #x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    #x = layers.MaxPooling2D((2, 2))(x)
    #x = layers.Flatten()(x)
    #x = layers.Dense(64, activation='relu')(x)
    #x = layers.Dropout(0.3)(x)

    #output_modes = layers.Dense(36, activation='sigmoid')(x)
    #output_weights = layers.Dense(3, activation='sigmoid')(x)


    x = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x= layers.Dropout(0.4)(x)
    output_modes = layers.Dense(36, activation='sigmoid')(x)
    output_weights = layers.Dense(3, activation='softmax')(x) #antes tenia puesto 'sigmoid' pero la suma d epesos NO era 1


    output = layers.Concatenate()([output_modes, output_weights])

    model = keras.Model(inputs=input_layer, outputs=output)
    
    model.compile(optimizer=optimizer,
              loss=custom_loss,
              metrics=['mae', mae_pesos, acc_modos, std_metric])

    return model

def plot_metrics(history):
    plt.figure(figsize=(12,5))
    
    # Pérdida
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Pérdida entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida validación')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida (MAE)')
    plt.legend()
    plt.title("Evolución de la Pérdida")
    
    # MAE
    plt.subplot(1,2,2)
    plt.plot(history.history['mae'], label='MAE entrenamiento')
    plt.plot(history.history['val_mae'], label='MAE validación')
    plt.xlabel('Épocas')
    plt.ylabel('MAE')
    plt.legend()
    plt.title("Evolución del MAE")
    
    plt.tight_layout()
    plt.show()

def show_predictions(model, X_test, y_test, num_examples=2):
    preds = model.predict(X_test[:num_examples])
    for i in range(num_examples):
        plt.imshow(X_test[i].squeeze(), cmap='gray')
        plt.axis('off')
        plt.title("Imagen de prueba")
        plt.show()
        
        y_true = y_test[i]
        y_pred = preds[i]

        true_mask = y_true[:36]
        true_weights = y_true[36:]

        pred_mask = y_pred[:36]
        pred_weights = y_pred[36:]

        true_modes = np.argwhere(true_mask > 0.5)
        pred_modes = np.argsort(pred_mask)[-3:][::-1]  # Top 3 activaciones predichas

        print(f"--- Ejemplo {i+1} ---")
        print("Modos reales:")
        for j, idx in enumerate(true_modes.flatten()):
            l, m = divmod(idx, 6)
            print(f"  l={l}, m={m}, peso={true_weights[j]:.2f}")

        print("Modos predichos:")
        for j, idx in enumerate(pred_modes):
            l, m = divmod(idx, 6)
            peso = pred_weights[j] if j < 3 else 0.0
            print(f"  l={l}, m={m}, peso={peso:.2f}")
        print()

# -----------------------------
# EJECUCIÓN
# -----------------------------
X, y = load_data(DATASET_PATH)
X, y = shuffle(X, y, random_state=42)
model = create_model()
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# División manual (80% train, 10% val, 10% test)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.111, random_state=42)  # 0.111 * 0.9 ≈ 0.1

# Crear datasets eficientes
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
    .shuffle(buffer_size=1000) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)) \
    .batch(BATCH_SIZE)

history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, callbacks=[early_stop])

# Evaluación
results = model.evaluate(test_dataset)
loss = results[0]
mae = results[1]
maepesos=results[2]
accmodos=results[3]
DESVest=results[4]
# Evaluación

#results = model.evaluate(X_test, y_test)
#loss = results[0]
#mae = results[1]

print(f"\nPérdida en test: {loss:.4f}")
print(f"MAE en test: {mae:.4f}")
#print(f"Exactitud aproximada (MAE < 0.05): {'✅' if mae < 0.05 else '❌'}")
#print(f"Pérdida deseada alcanzada (< 0.1): {'✅' if loss < 0.1 else '❌'}")
print(f"Accmodos en test: {accmodos:.4f}")
print(f"MAE pesos en test: {maepesos:.4f}")

print(f"DESVestandar pesos en test: {DESVest:.4f}")
# Mostrar predicciones
show_predictions(model, X_test, y_test)

# Gráficas de entrenamiento
plot_metrics(history)
