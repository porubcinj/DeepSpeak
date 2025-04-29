class Config:
    def __init__(
            self, *,
            datasets_dir="datasets", raw_dir="raw",
            split_dir="split", train_dir="train", val_dir="val", test_dir="test",
            groups_dir="groups", messages_dir="messages",
            samples_dir="samples", meta_json="meta.json",
            recreate_datasets=False, recreate_samples=False,
            val_split=0.2, test_split=0.2,
            device="cpu", best_model_path="best_model.pt", checkpoint="checkpoint.pt",
            resume=False, num_epochs=256, batch_size=8,
            max_context_length=512, max_group_size=4096, max_samples_per_group=65536, patience=8,
            output_dir="output", log="log.log",
        ):
        # Datasets
        self.datasets_dir = datasets_dir
        self.raw_dir = raw_dir
        self.split_dir = split_dir
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.groups_dir = groups_dir
        self.messages_dir = messages_dir
        self.samples_dir = samples_dir
        self.meta_json = meta_json
        self.recreate_datasets = recreate_datasets
        self.recreate_samples = recreate_samples
        self.val_split = val_split
        self.test_split = test_split
        # PyTorch
        self.device = device
        self.best_model_path = best_model_path
        self.checkpoint = checkpoint
        # Training
        self.resume = resume
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_context_length = max_context_length
        self.max_group_size = max_group_size
        self.max_samples_per_group = max_samples_per_group
        self.patience = patience
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
    def split_dir(self):
        return self._split_dir
    @split_dir.setter
    def split_dir(self, split_dir):
        self._split_dir = str(split_dir)

    @property
    def train_dir(self):
        return self._train_dir
    @train_dir.setter
    def train_dir(self, train_dir):
        self._train_dir = str(train_dir)

    @property
    def val_dir(self):
        return self._val_dir
    @val_dir.setter
    def val_dir(self, val_dir):
        self._val_dir = str(val_dir)

    @property
    def test_dir(self):
        return self._test_dir
    @test_dir.setter
    def test_dir(self, test_dir):
        self._test_dir = str(test_dir)

    @property
    def groups_dir(self):
        return self._groups_dir
    @groups_dir.setter
    def groups_dir(self, groups_dir):
        self._groups_dir = str(groups_dir)

    @property
    def messages_dir(self):
        return self._messages_dir
    @messages_dir.setter
    def messages_dir(self, messages_dir):
        self._messages_dir = str(messages_dir)

    @property
    def samples_dir(self):
        return self._samples_dir
    @samples_dir.setter
    def samples_dir(self, samples_dir):
        self._samples_dir = str(samples_dir)

    @property
    def meta_json(self):
        return self._meta_json
    @meta_json.setter
    def meta_json(self, meta_json):
        self._meta_json = str(meta_json)

    @property
    def recreate_datasets(self):
        return self._recreate_datasets
    @recreate_datasets.setter
    def recreate_datasets(self, recreate_datasets):
        self._recreate_datasets = bool(recreate_datasets)

    @property
    def recreate_samples(self):
        return self._recreate_samples
    @recreate_samples.setter
    def recreate_samples(self, recreate_samples):
        self._recreate_samples = bool(recreate_samples)

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
    def device(self):
        return self._device
    @device.setter
    def device(self, device):
        self._device = str(device)

    @property
    def best_model_path(self):
        return self._best_model_path
    @best_model_path.setter
    def best_model_path(self, best_model_path):
        self._best_model_path = str(best_model_path)

    @property
    def checkpoint(self):
        return self._checkpoint
    @checkpoint.setter
    def checkpoint(self, checkpoint):
        self._checkpoint = str(checkpoint)

    @property
    def resume(self):
        return self._resume
    @resume.setter
    def resume(self, resume):
        self._resume = bool(resume)

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
    def max_samples_per_group(self):
        return self._max_samples_per_group
    @max_samples_per_group.setter
    def max_samples_per_group(self, max_samples_per_group):
        max_samples_per_group = int(max_samples_per_group)
        assert max_samples_per_group > 1, max_samples_per_group
        self._max_samples_per_group = max_samples_per_group

    @property
    def patience(self):
        return self._patience
    @patience.setter
    def patience(self, patience):
        patience = int(patience)
        assert patience > 0, patience
        self._patience = patience

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
