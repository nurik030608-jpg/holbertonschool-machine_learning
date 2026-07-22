#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras


def preprocess_data(X, Y):
    """
    Предобработка данных для CIFAR-10 под выбранную модель (DenseNet121).
    
    X: numpy.ndarray формы (m, 32, 32, 3)
    Y: numpy.ndarray формы (m,)
    
    Возвращает: X_p, Y_p
    """
    # 1. Нормализация/предобработка изображений для DenseNet
    X_p = keras.applications.densenet.preprocess_input(X)
    
    # 2. Перевод меток в one-hot векторы (10 классов)
    Y_p = keras.utils.to_categorical(Y, 10)
    
    return X_p, Y_p


def train_cifar10():
    # 1. Загрузка CIFAR-10
    (X_train, Y_train), (X_val, Y_val) = keras.datasets.cifar10.load_data()
    
    # Сплющиваем Y из (m, 1) в (m,) для корректной работы preprocess_data
    Y_train = Y_train.flatten()
    Y_val = Y_val.flatten()
    
    # Предобработка
    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_val_p, Y_val_p = preprocess_data(X_val, Y_val)
    
    # 2. Создаем базовую модель и слой масштабирования
    # Используем Lambda-слой для ресайза 32x32 -> 224x224
    input_tensor = keras.Input(shape=(32, 32, 3))
    resized_input = keras.layers.Lambda(
        lambda image: tf.image.resize(image, (224, 224))
    )(input_tensor)
    
    # Загружаем предобученный DenseNet121 без верхнего слоя
    base_model = keras.applications.DenseNet121(
        weights='imagenet',
        include_top=False,
        input_tensor=resized_input,
        pooling='avg'  # Автоматически добавляет GlobalAveragePooling2D
    )
    
    # Замораживаем базовую модель
    base_model.trainable = False
    
    # 3. Извлечение признаков (Hint 3: считаем 1 раз, чтобы сэкономить время)
    print("Извлечение признаков из замороженной модели (Feature Extraction)...")
    feature_extractor = keras.Model(inputs=input_tensor, outputs=base_model.output)
    
    features_train = feature_extractor.predict(X_train_p, batch_size=64, verbose=1)
    features_val = feature_extractor.predict(X_val_p, batch_size=64, verbose=1)import tensorflow as tf
from tensorflow import keras


def preprocess_data(X, Y):
    """
    Предобработка данных для CIFAR-10 под выбранную модель (DenseNet121).
    
    X: numpy.ndarray формы (m, 32, 32, 3)
    Y: numpy.ndarray формы (m,)
    
    Возвращает: X_p, Y_p
    """
    # 1. Нормализация/предобработка изображений для DenseNet
    X_p = keras.applications.densenet.preprocess_input(X)
    
    # 2. Перевод меток в one-hot векторы (10 классов)
    Y_p = keras.utils.to_categorical(Y, 10)
    
    return X_p, Y_p


def train_cifar10():
    # 1. Загрузка CIFAR-10
    (X_train, Y_train), (X_val, Y_val) = keras.datasets.cifar10.load_data()
    
    # Сплющиваем Y из (m, 1) в (m,) для корректной работы preprocess_data
    Y_train = Y_train.flatten()
    Y_val = Y_val.flatten()
    
    # Предобработка
    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_val_p, Y_val_p = preprocess_data(X_val, Y_val)
    
    # 2. Создаем базовую модель и слой масштабирования
    # Используем Lambda-слой для ресайза 32x32 -> 224x224
    input_tensor = keras.Input(shape=(32, 32, 3))
    resized_input = keras.layers.Lambda(
        lambda image: tf.image.resize(image, (224, 224))
    )(input_tensor)
    
    # Загружаем предобученный DenseNet121 без верхнего слоя
    base_model = keras.applications.DenseNet121(
        weights='imagenet',
        include_top=False,
        input_tensor=resized_input,
        pooling='avg'  # Автоматически добавляет GlobalAveragePooling2D
    )
    
    # Замораживаем базовую модель
    base_model.trainable = False
    
    # 3. Извлечение признаков (Hint 3: считаем 1 раз, чтобы сэкономить время)
    print("Извлечение признаков из замороженной модели (Feature Extraction)...")
    feature_extractor = keras.Model(inputs=input_tensor, outputs=base_model.output)
    
    features_train = feature_extractor.predict(X_train_p, batch_size=64, verbose=1)
    features_val = feature_extractor.predict(X_val_p, batch_size=64, verbose=1)
    
    # 4. Строим верхний классификатор (Top Model)
    top_input = keras.Input(shape=(base_model.output.shape[1],))
    x = keras.layers.BatchNormalization()(top_input)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(10, activation='softmax')(x)
    
    top_model = keras.Model(inputs=top_input, outputs=outputs)
    
    top_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callback для остановки и сохранения лучших весов
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)
    ]
    
    # Обучаем только классификатор на готовых фичах
    print("Обучение классификатора...")
    top_model.fit(
        features_train, Y_train_p,
        validation_data=(features_val, Y_val_p),
        epochs=20,
        batch_size=128,
        callbacks=callbacks
    )
    
    # 5. Собираем итоговую полную модель (для сохранения в cifar10.h5)
    full_outputs = top_model(base_model.output)
    full_model = keras.Model(inputs=input_tensor, outputs=full_outputs)
    
    full_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Проверяем точность итоговой модели на валидации
    val_loss, val_acc = full_model.evaluate(X_val_p, Y_val_p, batch_size=64)
    print(f"\nИтоговая точность на валидации: {val_acc * 100:.2f}%")
    
    # Сохраняем модель
    full_model.save('cifar10.h5')
    print("Модель успешно сохранена в cifar10.h5")


if __name__ == '__main__':
    train_cifar10()
    
    # 4. Строим верхний классификатор (Top Model)
    top_input = keras.Input(shape=(base_model.output.shape[1],))
    x = keras.layers.BatchNormalization()(top_input)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(10, activation='softmax')(x)
    
    top_model = keras.Model(inputs=top_input, outputs=outputs)
    
    top_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callback для остановки и сохранения лучших весов
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)
    ]
    
    # Обучаем только классификатор на готовых фичах
    print("Обучение классификатора...")
    top_model.fit(
        features_train, Y_train_p,
        validation_data=(features_val, Y_val_p),
        epochs=20,
        batch_size=128,
        callbacks=callbacks
    )
    
    # 5. Собираем итоговую полную модель (для сохранения в cifar10.h5)
    full_outputs = top_model(base_model.output)
    full_model = keras.Model(inputs=input_tensor, outputs=full_outputs)
    
    full_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Проверяем точность итоговой модели на валидации
    val_loss, val_acc = full_model.evaluate(X_val_p, Y_val_p, batch_size=64)
    print(f"\nИтоговая точность на валидации: {val_acc * 100:.2f}%")
    
    # Сохраняем модель
    full_model.save('cifar10.h5')
    print("Модель успешно сохранена в cifar10.h5")


if __name__ == '__main__':
    train_cifar10()
