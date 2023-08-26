from keras.callbacks import Callback
import GPUtil

class VRAMMonitor(Callback):
    def __init__(self):
        super(VRAMMonitor, self).__init__()
        self.memory_usages = []
        self.monitor_every_n_batches = 10  # Change this to your preference

    def on_batch_end(self, batch, logs=None):
        if batch % self.monitor_every_n_batches == 0:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                self.memory_usages.append(gpu.memoryUsed)


    def on_epoch_end(self, epoch, logs=None):
        average_memory_usage = sum(self.memory_usages) / len(self.memory_usages) if self.memory_usages else 0
        logs['average_memory_usage'] = average_memory_usage
        logs['min_memory_usage'] = min(self.memory_usages)
        logs['max_memory_usage'] = max(self.memory_usages)
        self.memory_usages = []
