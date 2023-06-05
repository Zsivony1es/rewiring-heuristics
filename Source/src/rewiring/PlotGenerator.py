import matplotlib.pyplot as plt
from datetime import datetime
import logging

log = logging.getLogger(__name__)
class PlotGenerator:
    def __init__(self) -> None:
        self.out_path = './Source/out/plots/'

    def generate_loss_plot(self, 
                           train_loss: list[float], 
                           val_loss: list[float], 
                           dset_name: str, 
                           model_type: str) -> None:
        print("Generating loss plot...")
        plt.plot(train_loss, label='Training')
        plt.plot(val_loss, label='Validation')
        plt.title(f'Training and validation loss for {dset_name} dataset using a {model_type} model')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d_%H:%M:%S")
        filename = 'lossPlot_' + dset_name + '_' + model_type + '_' + timestamp_str + '.png'
        plt.savefig(self.out_path + filename)
        log.info(f"Saved figure to {self.out_path} as {filename}")

    def generate_accuracy_plot(self, 
                               accuracy: list[float],
                               dset_name: str, 
                               model_type: str) -> None:
        print("Generating accuracy plot...")
        plt.plot(accuracy)
        plt.title(f'Accuracy measurments for {dset_name} dataset using a {model_type} model')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d_%H:%M:%S")
        filename = 'accPlot_' + dset_name + '_' + model_type + '_' + timestamp_str + '.png'
        plt.savefig(self.out_path + filename)
        log.info(f"Saved figure to {self.out_path} as {filename}")
