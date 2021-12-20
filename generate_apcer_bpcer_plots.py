import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

# Import helper functions we're going to use

plt.style.use("ggplot")

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

error_array = pd.DataFrame(columns=["Network", "EER", "BPCER_5", "BPCER_10"])
# for network_number in network_names:
# for network_number in range(0, 23):
for network_number in range(0, 1):
    print(network_names[network_number])
    result_file = 'livdet2019-train--test-combined-saved-model-100-epochs-results/' + network_names[
        network_number] + '-scores-prediction.txt'
    data = pd.read_csv(result_file)
    predictions = data.iloc[:, 0].to_numpy()
    true_labels = data.iloc[:, 2].to_numpy()
    fpr, tpr, threshold = roc_curve(true_labels, predictions)
    plt.plot(fpr * 100, (1 - tpr) * 100, label="{}".format(network_names[network_number]))

    attack_scores = 1 - predictions[true_labels == 0]
    bonafide_scores = 1 - predictions[true_labels == 1]
    sorted_attack_scores = np.sort(attack_scores)

    index_apcer_5 = round(0.05 * len(sorted_attack_scores))
    apcer_5 = sorted_attack_scores[index_apcer_5]
    index_apcer_10 = round(0.1 * len(sorted_attack_scores))
    apcer_10 = sorted_attack_scores[index_apcer_10]

    bpcer_5 = 100 - 100 * (bonafide_scores <= apcer_5).sum() / len(bonafide_scores)
    # print(bpcer_5)
    bpcer_10 = 100 - 100 * (bonafide_scores <= apcer_10).sum() / len(bonafide_scores)
    print("BPCER 5% - {} :: 10% {}".format(bpcer_5, bpcer_10))

    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    error_array.at[network_number, 'Network'] = network_names[network_number]
    error_array.at[network_number, 'BPCER_5'] = bpcer_5
    error_array.at[network_number, 'BPCER_10'] = bpcer_10
    error_array.at[network_number, 'EER'] = EER * 100

print(error_array)
print(error_array.to_latex(index=False))
# x coordinates for the lines
xcoords = [5, 10]
# colors for the lines
colors = ['r', 'k']

for xc, c in zip(xcoords, colors):
    plt.axvline(x=xc, linestyle='--', label='APCER = {}%'.format(xc), c=c)
plt.legend()
plt.title('DET Curves')
plt.xlabel('APCER (%)')
plt.ylabel('BPCER (%)')
plt.grid(True)
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.savefig("apcer-bpcer-plot2.png")
plt.show()
