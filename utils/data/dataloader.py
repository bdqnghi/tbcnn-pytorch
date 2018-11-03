from torch.utils.data import DataLoader

class ProgramDataloader(DataLoader):

    def __init__(self, *args, **kwargs):
        super(ProgramDataloader, self).__init__(*args, **kwargs)
