import torch
from torchdata.datapipes.iter import FileLister
from Base import BaseDataLoader
from Config import Config
from DataLoader.DataPipe.textreader import CsvReader
from re import sub


class DataLoader(BaseDataLoader):

    def __init__(self, test=False):
        if test:
            config = Config().test_loader.args
            datapipe = FileLister(config.data_dir, "*.csv")
        else:
            config = Config().train_loader.args
            datapipe = FileLister(config.data_dir, "*.csv")

        self.label = {}
        self.get_label(datapipe)
        datapipe = datapipe.open_files(mode="r")
        datapipe = CsvReader(datapipe, return_path=True)
        super().__init__(datapipe, self.label, test)

    def get_label(self, datapipe):
        for name in datapipe:
            index = int(sub(r"\D*(\d+)\D*", r"\1", name))
            index = index // 45
            self.label[name] = index


if __name__ == "__main__":
    for data, target in DataLoader():
        print(data, target)
