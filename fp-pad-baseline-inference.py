# Get helper_functions.py script from course GitHub
# !wget https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/helper_functions.py
import os

from numpy import savetxt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
import numpy as np
import tensorflow as tf
# Import helper functions we're going to use

import datetime

test_dir = r"\train_test\/test"
saved_model_path = "./livdet2019-combined-saved-model-100-epochs-retrained/"
save_path = "./livdet2019-combined-saved-model-100-epochs-retrained-results/"


# testDigitalPersona-reorganized
# testGreenBit-reorganized
# testOrcathus-reorganized


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

NUM_CLASSES = 2
IMG_SIZE = (224, 224)  # define image size
saved_paths = [""]
test_paths = ["combined"]
model_paths = ["100-epochs"]
for model_path_suffix in model_paths:
    for saved_path in saved_paths:
        train_set = saved_path.split("-")
        train_set = train_set[0]

        for test_path in test_paths:
            saved_model_path = saved_model_path + saved_path
            test_dir = test_dir
            save_path = "./livdet2019-train-" + train_set + "-test-" + test_path + "-saved-model-" + model_path_suffix + "-results/"

            if not os.path.exists(save_path):
                os.mkdir(save_path)

            validation_data = tf.keras.preprocessing.image_dataset_from_directory(directory=test_dir,
                                                                                  labels="inferred",
                                                                                  image_size=IMG_SIZE,
                                                                                  label_mode="categorical",
                                                                                  batch_size=128,
                                                                                  shuffle=False)

            class_names = validation_data.class_names
            print(class_names)
            '''
            file_paths = validation_data.file_paths
            print(file_paths)
            '''

            # for network_number in network_names:
            for network_number in range(0, 17):
                print(network_names[network_number])
                # checkpoint_path = saved_model_path + network_names[network_number] + 'checkpoint'
                checkpoint_path = saved_model_path + network_names[network_number]
                with tf.device('/gpu:1'):
                    model_0 = tf.keras.models.load_model(checkpoint_path)
                    '''
                    val_steps_per_epoch = 128
                    final_loss, final_accuracy = model_0.evaluate(validation_data, steps=val_steps_per_epoch)
                    print("Final loss: {:.2f}".format(final_loss))
                    print("Final accuracy: {:.2f}%".format(final_accuracy * 100))
                    '''

                    # get the labels
                    predictions = []
                    true_labels = []
                    predicted_scores = np.empty((0, 2))
                    for data_to_predict, true_label in validation_data:
                        predicted_labels = model_0.predict(data_to_predict)
                        predicted_scores = np.append(predicted_scores, predicted_labels, axis=0)
                        true_label = np.argmax(true_label, axis=-1)
                        # print(true_label)
                        predicted_label = np.argmax(predicted_labels, axis=-1);
                        predictions = np.concatenate([predictions, predicted_label])
                        true_labels = np.concatenate([true_labels, true_label])

                    predictions_scores_true_labels = np.transpose(
                        np.vstack((predicted_scores[:, 1], predictions, true_labels)))
                    result_file = save_path + network_names[network_number] + '-scores-prediction.txt'
                    savetxt(result_file, predictions_scores_true_labels, fmt=[b'%f', '%d', '%d'], delimiter=',',
                            newline='\n')

                    # Save filenames corresponding to batch indices
                    path_files = save_path + network_names[network_number] + '-file-paths.txt'
                    path_file = open(path_files, "w")
                    for element in validation_data.file_paths:
                        path_file.write(element + "\n")
                    path_file.close()

