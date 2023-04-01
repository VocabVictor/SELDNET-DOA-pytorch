from torch import tensor
from torchdata.dataloader2 import DataLoader2
from Config import Config


class BaseDataLoader(DataLoader2):
    """
    Base class for all Data loaders
    """

    def __init__(
            self,
            datapipe,
            label=None,
            test=False,
            unzip=True,
            datapipe_adapter_fn=None,
            reading_service=None
    ):
        self.__length = 0
        datapipe = datapipe.shuffle()
        if test:
            config = Config().test_loader.args
        else:
            config = Config().train_loader.args
        if unzip:
            labelpipe, datapipe = datapipe.unzip(sequence_length=2)
            labelpipe = labelpipe.map(lambda x: label[x])
            self.batch_size = config.batch_size
            datapipe = datapipe.batch(batch_size=config.batch_size, drop_last=config.drop_last)
            labelpipe = labelpipe.batch(batch_size=config.batch_size, drop_last=config.drop_last)
            datapipe = datapipe.collate(self.collate_fn)
            labelpipe = labelpipe.collate(self.collate_fn)
            datapipe = datapipe.zip(labelpipe)
        super().__init__(
            datapipe,
            datapipe_adapter_fn,
            reading_service
        )

    def collate_fn(self, batch):
        new_batch = tensor(batch)
        self.__length += len(new_batch)
        return new_batch

    def __len__(self):
        return self.__length // 2
