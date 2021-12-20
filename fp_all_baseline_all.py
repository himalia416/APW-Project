# Get helper_functions.py script from course GitHub
# !wget https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/helper_functions.py

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''

import tensorflow as tf
# Import helper functions we're going to use
from helper_functions import create_tensorboard_callback, plot_loss_curves, unzip_data, walk_through_dir
import datetime

# Create data inputs for test and train directory
train_dir = r"\train_test\/train"
test_dir = r"\train_test\/test"


# train_dir = "./livdet-2019/train/trainOrcathus-reorganized/"
# train_dir = "/home/kiran/Desktop/fp-book-chapter-2021/livdet-2019/train-single-set/"


def create_tensorboard_callback(dir_name, experiment_name):
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir
    )
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback


# Function to convert number into string
# Switcher is dictionary data type for selecting model
def get_dl_model(argument):
    switcher = {
        0: tf.keras.applications.DenseNet121(include_top=False),
        1: tf.keras.applications.DenseNet169(include_top=False),
        2: tf.keras.applications.DenseNet201(include_top=False),
        3: tf.keras.applications.EfficientNetB0(include_top=False),
        4: tf.keras.applications.EfficientNetB1(include_top=False),
        5: tf.keras.applications.EfficientNetB2(include_top=False),
        6: tf.keras.applications.EfficientNetB3(include_top=False),
        7: tf.keras.applications.EfficientNetB4(include_top=False),
        8: tf.keras.applications.EfficientNetB5(include_top=False),
        9: tf.keras.applications.EfficientNetB6(include_top=False),
        10: tf.keras.applications.EfficientNetB7(include_top=False),
        11: tf.keras.applications.InceptionResNetV2(include_top=False),
        12: tf.keras.applications.InceptionV3(include_top=False),
        13: tf.keras.applications.MobileNet(include_top=False),
        14: tf.keras.applications.MobileNetV2(include_top=False),
        15: tf.keras.applications.MobileNetV3Large(include_top=False),
        16: tf.keras.applications.MobileNetV3Small(include_top=False),
        17: tf.keras.applications.ResNet101(include_top=False),
        18: tf.keras.applications.ResNet101V2(include_top=False),
        19: tf.keras.applications.ResNet152(include_top=False),
        20: tf.keras.applications.ResNet152V2(include_top=False),
        21: tf.keras.applications.ResNet50(include_top=False),
        22: tf.keras.applications.ResNet50V2(include_top=False),
        23: tf.keras.applications.VGG16(include_top=False),
        24: tf.keras.applications.VGG19(include_top=False),
        25: tf.keras.applications.Xception(include_top=False),
        26: tf.keras.applications.NASNetLarge(include_top=False),
        27: tf.keras.applications.NASNetMobile(include_top=False),
    }

    # get() method of dictionary data type returns
    # value of passed argument if it is present
    # in dictionary otherwise second argument will
    # be assigned as default value of passed argument
    return switcher.get(argument, tf.keras.applications.EfficientNetB0(include_top=False))


# Function to convert number into string
# Switcher is dictionary data type for selecting model
def get_model_specific_preprocessing(argument, input_data):
    switcher = {
        0: tf.keras.applications.densenet.preprocess_input(input_data),
        1: tf.keras.applications.densenet.preprocess_input(input_data),
        2: tf.keras.applications.densenet.preprocess_input(input_data),
        3: input_data,
        4: input_data,
        5: input_data,
        6: input_data,
        7: input_data,
        8: input_data,
        9: input_data,
        10: input_data,
        11: tf.keras.applications.inception_resnet_v2.preprocess_input(input_data),
        12: tf.keras.applications.inception_v3.preprocess_input(input_data),
        13: tf.keras.applications.mobilenet.preprocess_input(input_data),
        14: tf.keras.applications.resnet.preprocess_input(input_data),
        15: tf.keras.applications.resnet_v2.preprocess_input(input_data),
        16: tf.keras.applications.resnet.preprocess_input(input_data),
        17: tf.keras.applications.resnet_v2.preprocess_input(input_data),
        18: tf.keras.applications.resnet.preprocess_input(input_data),
        19: tf.keras.applications.resnet_v2.preprocess_input(input_data),
        20: tf.keras.applications.vgg16.preprocess_input(input_data),
        21: tf.keras.applications.vgg19.preprocess_input(input_data),
        22: tf.keras.applications.xception.preprocess_input(input_data),
        23: tf.keras.applications.nasnet.preprocess_input(input_data),
        24: tf.keras.applications.nasnet.preprocess_input(input_data),
        25: tf.keras.applications.mobilenet_v2.preprocess_input(input_data),
        26: tf.keras.applications.mobilenet_v3.preprocess_input(input_data),
        27: tf.keras.applications.mobilenet_v3.preprocess_input(input_data),
    }

    # get() method of dictionary data type returns
    # value of passed argument if it is present
    # in dictionary otherwise second argument will
    # be assigned as default value of passed argument
    return switcher.get(argument, tf.keras.applications.densenet.preprocess_input(input_data))


NUM_CLASSES = 2
NUM_EPOCHS = 50
IMG_SIZE = (224, 224)  # define image size
train_data = tf.keras.preprocessing.image_dataset_from_directory(directory=train_dir,
                                                                 labels="inferred",
                                                                 image_size=IMG_SIZE,
                                                                 validation_split=0.2,
                                                                 subset="training",
                                                                 seed=123,
                                                                 label_mode="categorical",  # what type are the labels?
                                                                 batch_size=32)  # batch_size is 32 by default, this
# is generally a good number
validation_data = tf.keras.preprocessing.image_dataset_from_directory(directory=train_dir,
                                                                      validation_split=0.2,
                                                                      subset="validation",
                                                                      seed=123,
                                                                      labels="inferred",
                                                                      image_size=IMG_SIZE,
                                                                      label_mode="categorical")
class_names = train_data.class_names
print(class_names)

network_names = {
    0: "DenseNet121",
    1: "DenseNet169",
    2: "DenseNet201",
    3: "EfficientNetB0",
    4: "EfficientNetB1",
    5: "EfficientNetB2",
    6: "EfficientNetB3",
    7: "EfficientNetB4",
    8: "EfficientNetB5",
    9: "EfficientNetB6",
    10: "EfficientNetB7",
    11: "InceptionResNetV2",
    12: "InceptionV3",
    13: "MobileNet",
    14: "ResNet101",
    15: "ResNet101V2",
    16: "ResNet152",
    17: "ResNet152V2",
    18: "ResNet50",
    19: "ResNet50V2",
    20: "VGG16",
    21: "VGG19",
    22: "Xception",
    23: "NASNetLarge",
    24: "NASNetMobile",
    25: "MobileNetV2",
    26: "MobileNetV3Large",
    27: "MobileNetV3Small"
}
print(type(network_names))
# for network_number in network_names:
for network_number in range(0, 27):
    print(network_names[network_number])
    base_model = get_dl_model(network_number)

    # 1. Create base model with tf.keras.applications
    # base_model = tf.keras.applications.EfficientNetB0(include_top=False)

    # 2. Freeze the base model (so the pre-learned patterns remain)
    base_model.trainable = True

    # 3. Create inputs into the base model
    inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")

    # 4. If using ResNet50V2, add this to speed up convergence, remove for EfficientNet
    x = get_model_specific_preprocessing(network_number, inputs)

    # 5. Pass the inputs to the base_model (note: using tf.keras.applications, EfficientNet inputs don't have to be
    # normalized)
    x = base_model(inputs)
    # Check data shape after passing it to base_model
    print(f"Shape after base_model: {x.shape}")

    # 6. Average pool the outputs of the base model (aggregate all the most important information, reduce number of
    # computations)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
    print(f"After GlobalAveragePooling2D(): {x.shape}")

    # 7. Create the output activation layer
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="output_layer")(x)
    # outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="sigmoid", name="output_layer")(x)

    # 8. Combine the inputs with the outputs into a model
    model_0 = tf.keras.Model(inputs, outputs)

    # 9. Compile the model
    model_0.compile(loss='categorical_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["accuracy"])

    checkpoint_filepath = 'livdet2019-combined-saved-model-100-epochs-retrained/' + network_names[
        network_number] + 'checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # 10. Fit the model (we use less steps for validation so it's faster)
    history = model_0.fit(train_data,
                          epochs=NUM_EPOCHS,
                          steps_per_epoch=len(train_data),
                          validation_data=validation_data,
                          # Go through less of the validation data so epochs are faster (we want faster experiments!)
                          validation_steps=int(0.25 * len(validation_data)),
                          # Track our model's training logs for visualization later
                          callbacks=[create_tensorboard_callback("transfer_learning-retrained",
                                                                 "100_epochs_" + network_names[
                                                                     network_number]), model_checkpoint_callback])

    model_0.save('livdet2019-combined-saved-model-100-epochs-retrained/' + network_names[network_number])
