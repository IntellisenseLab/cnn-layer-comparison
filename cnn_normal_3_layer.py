import tensorflow
from keras.callbacks import EarlyStopping
import pathlib
import matplotlib.pyplot as plt

data_dir = pathlib.Path("AnnotatedImages")

batch_size = 32
img_height = 245
img_width  = 262

train_ds = tensorflow.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tensorflow.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tensorflow.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tensorflow.data.AUTOTUNE)

hiddenLayer = [300]
firstLayerConv2D = [4, 8]
firstLayerMaxPool = [3, 4]
secondLayerConv2D = [16, 32]
secondLayerMaxPool = [3, 4]
thirdLayerConv2D = [16, 32, 64]
thirdLayerMaxPool = [2, 3, 4]
dropOut = [0.1, 0.2, 0.3]

iteration = 1

resultsFile = open("rgb_results_three_layer.txt", "w+")

for firstLayerConv in firstLayerConv2D:
    for firstLayerPool in firstLayerMaxPool:
        for secondLayerConv in secondLayerConv2D:
            for secondLayerPool in secondLayerMaxPool:
                for thirdLayerConv in thirdLayerConv2D:
                    for thirdLayerPool in thirdLayerMaxPool:
                        for neuralCount in hiddenLayer:
                            for percentage in dropOut:

                                model = tensorflow.keras.Sequential(
                                  [
                                    tensorflow.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

                                    tensorflow.keras.layers.Conv2D(firstLayerConv, (3,3), padding='same', activation="relu"),
                                    tensorflow.keras.layers.MaxPooling2D((firstLayerPool, firstLayerPool), strides=firstLayerPool),

                                    tensorflow.keras.layers.Conv2D(secondLayerConv, (3,3), padding='same', activation="relu"),
                                    tensorflow.keras.layers.MaxPooling2D((secondLayerPool, secondLayerPool), strides=secondLayerPool),

                                    tensorflow.keras.layers.Conv2D(thirdLayerConv, (3,3), padding='same', activation="relu"),
                                    tensorflow.keras.layers.MaxPooling2D((thirdLayerPool, thirdLayerPool), strides=thirdLayerPool),

                                    tensorflow.keras.layers.Flatten(),
                                    tensorflow.keras.layers.Dense(neuralCount, activation="relu"),
                                    tensorflow.keras.layers.Dropout(percentage),
                                    tensorflow.keras.layers.Dense(len(class_names), activation="softmax")
                                  ]
                                )

                                tensorflow.keras.utils.plot_model(model, "model.png", show_shapes=True)
                                model.compile(optimizer='adam', loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
                                model.summary()
                                
                                model_checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(
                                              filepath='model.h5',
                                              save_weights_only=False,
                                              monitor='val_loss',
                                              mode='min',
                                              save_best_only=True)
                                
                                callbacksE = [
                                              EarlyStopping(patience=4, restore_best_weights=True),
                                              model_checkpoint_callback,
                                        ]
                                
                                history = model.fit(
                                                    train_ds,
                                                    validation_data=val_ds,
                                                    epochs=100,
                                                    callbacks=callbacksE,
                                                  )
                                
                                acc = history.history['accuracy']
                                val_acc = history.history['val_accuracy']
                                loss = history.history['loss']
                                val_loss = history.history['val_loss']

                                epochs_range = range(100)
                                
                                plt.figure(figsize=(8, 8))
                                plt.subplot(1, 2, 1)
                                plt.plot(acc, label='Training Accuracy')
                                plt.plot(val_acc, label='Validation Accuracy')
                                plt.legend(loc='lower right')
                                plt.title('Training and Validation Accuracy')
                                
                                plt.subplot(1, 2, 2)
                                plt.plot(loss, label='Training Loss')
                                plt.plot(val_loss, label='Validation Loss')
                                plt.legend(loc='upper right')
                                plt.title('Training and Validation Loss')

                                imageName = "rgb_3_"+str(iteration)+".png"
                                plt.savefig(imageName)

                                results = str(firstLayerConv) + ", " + str(firstLayerPool) + ", " + str(secondLayerConv) + ", " + str(secondLayerPool) +  ", " \
                                        + str(thirdLayerConv) + ", " + str(thirdLayerPool) + ", " + str(neuralCount) + ", " + str(percentage) + ", " \
                                        + "{:.4f}".format(acc[-5]) + ", " + "{:.4f}".format(loss[-5]) + ", " \
                                        + "{:.4f}".format(val_acc[-5]) + ", " + "{:.4f}".format(val_loss[-5]) + ", " \
                                        + str(len(acc)) + ", " + imageName + "\n"

                                resultsFile.write(results)
                                resultsFile.flush()
                                model = None
                                iteration += 1

