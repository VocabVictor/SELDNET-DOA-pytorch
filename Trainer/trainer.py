import torch
from numpy import sqrt
from Config import Config
from Base import BaseTrainer
from DataLoader import DataLoader
from Utils import MetricTracker
from Model import metric as Metric


class Trainer(BaseTrainer):
    def __init__(self):
        config = Config()
        self.device, _ = self.prepare_device(config.n_gpu)
        super().__init__(self.device)
        self.data_loader = DataLoader(test=False)
        self.len_epoch = self.data_loader.batch_size
        if config.test_loader:
            self.valid_data_loader = DataLoader(test=True)
            self.do_validation = True
        else:
            self.do_validation = False
        self.log_step = int(sqrt(self.data_loader.batch_size))
        self.metric_ftns = [getattr(Metric, metric) for metric in config.metrics]
        keys = [m.__name__ for m in self.metric_ftns]
        keys.append("loss")
        self.train_metrics = MetricTracker("train", *keys, writer=self.writer)
        self.valid_metrics = MetricTracker("val", *keys, writer=self.writer)

    def prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.warning("Warning: There\'s no GPU available on this machine,",
                         "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.warning(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are ",
                         "available on this machine.")
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            self.writer.inc_step()
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()
        if self.do_validation:
            val_log = self._valid_epoch()
            log.update(**{k: v for k, v in val_log.items()})
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _test(self):
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

    def _valid_epoch(self):

        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p)
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
