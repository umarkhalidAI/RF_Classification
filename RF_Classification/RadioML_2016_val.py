from rfml.data import build_dataset
from rfml.nn.eval import (
    compute_accuracy,
    compute_accuracy_on_cross_sections,
    compute_confusion,
)
from rfml.nn.model import build_model
from rfml.nn.train import build_trainer, PrintingTrainingListener

train, val, test, le = build_dataset(dataset_name="RML2016.10a", path='./data/RML2016.10a_dict.pkl')
model = build_model(model_name="resnet", input_samples=128, n_classes=len(le))
trainer = build_trainer(
    strategy="standard", max_epochs=3, gpu=True
)  # Note: Disable the GPU here if you do not have one
#trainer.register_listener(PrintingTrainingListener())
#trainer(model=model, training=train, validation=val, le=le)
# acc = compute_accuracy(model=model, data=test, le=le)
# acc_vs_snr, snr = compute_accuracy_on_cross_sections(
#     model=model, data=test, le=le, column="SNR"
# )
import seaborn
import matplotlib.pyplot as plt


def plot_confusion_matrix(data, labels, output_filename):
    """Plot confusion matrix using heatmap.

    Args:
        data (list of list): List of lists with confusion matrix data.
        labels (list): Labels which will be plotted across x and y axis.
        output_filename (str): Path to output file.

    """
    seaborn.set(color_codes=True)
    seaborn.set_style('whitegrid', {'font.family': 'serif', 'font.serif': 'Times New Roman'})
    plt.figure(1, figsize=(14, 14))

    plt.title("Confusion Matrix")

    seaborn.set(font_scale=1.4)
    ax = seaborn.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'})

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set(ylabel="True Label", xlabel="Predicted Label")

    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()



model.load('con5_re_fc.pt')
model.eval()
acc = compute_accuracy(model=model, data=test, le=le)
acc_vs_snr, snr = compute_accuracy_on_cross_sections(
    model=model, data=test, le=le, column="SNR"
)
cmn = compute_confusion(model=model, data=test, le=le)

# Calls to a plotting function could be inserted here
# For simplicity, this script only prints the contents as an example
print("===============================")
print("Overall Testing Accuracy: {:.4f}".format(acc))
print("SNR (dB)\tAccuracy (%)")
# print("===============================")
for acc, snr in zip(acc_vs_snr, snr):
    print("{snr:d}\t{acc:0.1f}".format(snr=snr, acc=acc * 100))
print("===============================")
print("Confusion Matrix:")
print(cmn)
labels = ['WBFM', 'AM-DSB', 'AM-SSB', 'CPFSK', 'GFSK', 'PAM4', 'BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64']

# create confusion matrix
plot_confusion_matrix(cmn, labels, "confusion_matrix_conv5_fc.svg")
#model.save("cnn.pt")
