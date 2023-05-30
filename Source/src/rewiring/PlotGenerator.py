import matplotlib.pyplot as plt
from datetime import datetime

class PlotGenerator:
    def __init__(self) -> None:
        self.out_path = '../out'

    def generate_loss_plot(self, train_loss: list[float], val_loss: list[float], dset_name: str, model_type: str):
        plt.plot(train_loss, label='train_loss')
        plt.plot(val_loss, label='val_loss')
        plt.title(f'Training and validation loss for {dset_name} dataset using a {model_type} model')
        plt.legend()

        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d_%H:%M:%S")
        plt.savefig(self.out_path + '/plot_' + dset_name + '_' + model_type + '_' + timestamp_str + '.png')
