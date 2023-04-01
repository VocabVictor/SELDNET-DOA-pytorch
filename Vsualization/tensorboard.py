import importlib
from datetime import datetime

from Vsualization import Logger


class TensorBoard:

    def __init__(self, log_dir="../Log", logger=None):
        self.writer = None
        self.selected_module = ""
        self.__step = 0

        if logger is None:
            logger = Logger(log_dir)

        for module in ["torch.utils.tensorboard", "tensorboardX"]:
            try:
                self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                succeeded = True
                self.selected_module = module
                break
            except ImportError:
                succeeded = False

        if not succeeded:
            message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                      "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch " \
                      "to " \
                      "version >= 1.1 to use 'torch.Utils.tensorboard' or turn off the option in the " \
                      "'config.json' file. "
            logger.warning(message)

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.timer = datetime.now()

    def inc_step(self):
        duration = datetime.now() - self.timer
        self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
        self.timer = datetime.now()
        self.__step += 1

    def add_scalar(self, key, value):
        self.writer.add_scalar(key, value, self.__step)

    def add_histogram(self, name, p, bins='auto'):
        self.writer.add_histogram(name, p, bins=bins)

    def close(self):
        self.writer.close()
