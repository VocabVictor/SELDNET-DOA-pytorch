from abc import abstractmethod
from glob import glob
from os.path import join

import torch
from numpy import inf
from torch import optim

from Config import Config
from Model import NetWork, loss, metric
from Vsualization import TensorBoard, Logger


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, device):
        config = Config()
        self.config = config
        self.logger = Logger(config.log_dir)
        cfg_trainer = config.trainer
        self.model = NetWork().to(device)
        self.criterion = getattr(loss, config.loss)
        if cfg_trainer.metric:
            self.metric_ftns = [getattr(metric, function) for function in config.metrics]
        optimizer = config.optimizer
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = getattr(torch.optim, optimizer.type)(
            params=trainable_params,
            **optimizer.args.dict()
        )
        lr_scheduler = config.lr_scheduler
        self.lr_scheduler = getattr(optim.lr_scheduler, lr_scheduler.type)(optimizer=self.optimizer,
                                                                           **lr_scheduler.args.dict())
        self.epochs = cfg_trainer.epochs
        self.save_period = cfg_trainer.save_period
        self.monitor = cfg_trainer.monitor
        self.batch_size = cfg_trainer.batch_size

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.early_stop
            if self.early_stop <= 0:
                self.early_stop = inf

        self.checkpoint_dir = cfg_trainer.save_dir

        # setup visualization writer instance
        self.writer = TensorBoard(config.log_dir, self.logger, cfg_trainer.tensorboard)

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _test(self):
        raise NotImplementedError

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.epochs):
            result = self._train_epoch(epoch)
            log = {'epoch': epoch, "epochs": self.epochs, "batch_size": self.batch_size}
            log.update(result)
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))
            best = False
            if self.mnt_mode != 'off':
                try:
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _save_checkpoint(self, epoch, save_best=False):
        arch = self.config.arch.type
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        if save_best:
            best_path = str(join(self.checkpoint_dir, 'model_best.pth'))
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        path = join(resume_path, self.config.suffix)
        if not glob(path):
            return
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        if checkpoint['arch'] != self.config.arch.type:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        else:
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.logger.info("Checkpoint loaded. Resume training ")

    def warning(self, args, sep=""):
        if isinstance(args, str):
            self.logger.warning(args)
        elif isinstance(args, list):
            self.logger.warning(sep.join(args))
