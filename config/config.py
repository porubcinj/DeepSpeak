class Config:
    def __init__(
            self, *,
            datasets_dir="datasets", raw_dir="raw",
            groups_csv="groups.csv", messages_csv="messages.csv",
            val_split=0.2, test_split=0.2, recreate_datasets=False,
            device="cpu",
            num_epochs=32, batch_size=4,
            max_context_length=128, max_group_size=6,
            output_dir="output", log="log.log",
        ):
        # Datasets
        self.datasets_dir = datasets_dir
        self.raw_dir = raw_dir
        self.groups_csv = groups_csv
        self.messages_csv = messages_csv
        self.val_split = val_split
        self.test_split = test_split
        self.recreate_datasets = recreate_datasets
        # PyTorch
        self.device = device
        # Hyperparameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_context_length = max_context_length
        self.max_group_size = max_group_size
        # Logging
        self.output_dir = output_dir
        self.log = log

    @property
    def datasets_dir(self):
        return self._datasets_dir
    @datasets_dir.setter
    def datasets_dir(self, datasets_dir):
        self._datasets_dir = str(datasets_dir)

    @property
    def raw_dir(self):
        return self._raw_dir
    @raw_dir.setter
    def raw_dir(self, raw_dir):
        self._raw_dir = str(raw_dir)

    @property
    def groups_csv(self):
        return self._groups_csv
    @groups_csv.setter
    def groups_csv(self, groups_csv):
        self._groups_csv = str(groups_csv)

    @property
    def messages_csv(self):
        return self._messages_csv
    @messages_csv.setter
    def messages_csv(self, messages_csv):
        self._messages_csv = str(messages_csv)

    @property
    def val_split(self):
        return self._val_split
    @val_split.setter
    def val_split(self, val_split):
        val_split = float(val_split)
        assert 0 <= val_split < 1, val_split
        self._val_split = val_split

    @property
    def test_split(self):
        return self._test_split
    @test_split.setter
    def test_split(self, test_split):
        test_split = float(test_split)
        assert 0 <= test_split < 1, test_split
        self._test_split = test_split

    @property
    def recreate_datasets(self):
        return self._recreate_datasets
    @recreate_datasets.setter
    def recreate_datasets(self, recreate_datasets):
        self._recreate_datasets = bool(recreate_datasets)

    @property
    def device(self):
        return self._device
    @device.setter
    def device(self, device):
        self._device = str(device)

    @property
    def num_epochs(self):
        return self._num_epochs
    @num_epochs.setter
    def num_epochs(self, num_epochs):
        num_epochs = int(num_epochs)
        assert num_epochs > 0, num_epochs
        self._num_epochs = num_epochs

    @property
    def batch_size(self):
        return self._batch_size
    @batch_size.setter
    def batch_size(self, batch_size):
        batch_size = int(batch_size)
        assert batch_size > 0, batch_size
        self._batch_size = batch_size

    @property
    def max_context_length(self):
        return self._max_context_length
    @max_context_length.setter
    def max_context_length(self, max_context_length):
        max_context_length = int(max_context_length)
        assert max_context_length > 0, max_context_length
        self._max_context_length = max_context_length

    @property
    def max_group_size(self):
        return self._max_group_size
    @max_group_size.setter
    def max_group_size(self, max_group_size):
        max_group_size = int(max_group_size)
        assert max_group_size > 2, max_group_size
        self._max_group_size = max_group_size

    @property
    def output_dir(self):
        return self._output_dir
    @output_dir.setter
    def output_dir(self, output_dir):
        self._output_dir = str(output_dir)

    @property
    def log(self):
        return self._log
    @log.setter
    def log(self, log):
        self._log = str(log)
