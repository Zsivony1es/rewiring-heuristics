from datetime import datetime
import logging
import os
import glob
import heapq

import matplotlib.pyplot as plt


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
        plt.clf()
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
        plt.clf()
        plt.plot(accuracy)
        plt.title(f'Accuracy measurments for {dset_name} dataset using a {model_type} model')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d_%H:%M:%S")
        filename = 'accPlot_' + dset_name + '_' + model_type + '_' + timestamp_str + '.png'
        plt.savefig(self.out_path + filename)
        log.info(f"Saved figure to {self.out_path} as {filename}")

    def generate_accuracy_plot_multi(self, 
                                     accuracy: dict,
                                     dset_name: str, 
                                     model_type: str) -> None:
        """
        Creates one accuracy plot for the accuracy data of multiple models. The :param:`accuracy` dictionary should contain
        the name of the dataset as a key and the list of accuracies corresponding to it as the value. 
        """
        print("Generating accuracy plot...")
        plt.clf()
        for key, val in accuracy.items():
            plt.plot(val, label=key)
        plt.legend()
        plt.title(f'Accuracy measurments using a {model_type} model')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d_%H:%M:%S")
        filename = 'multiAccPlot_' + model_type + '_' + timestamp_str + '.png'
        plt.savefig(self.out_path + filename)
        log.info(f"Saved figure to {self.out_path} as {filename}")

    def leave_last_n_plots(self, n: int = 0):
        """
        Deletes all generated plot pictures from the specified folder, except for the last :param:`n` files (based on modification time).
        """
        files = glob.glob(os.path.join(self.out_path, '*'))
        files = sorted(files, key=os.path.getmtime)

        files_to_keep = heapq.nlargest(n, files, key=os.path.getmtime)

        for file in files:
            if file not in files_to_keep:
                os.remove(file)
