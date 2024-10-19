# from .auto_task import AutoTask

from .constants import *
from .metrics import Scorer


def create_collate_fn(pad_token_id, max_length=None):
    def collate_fn(batch):
        fields = {field for example in batch for field in example.keys()}
        output_batch = {}
        for key in fields:
            if key in ["input_ids", "target_ids"]:
                output_batch[key] = torch.nn.utils.rnn.pad_sequence(
                    [example[key] for example in batch],
                    batch_first=True,
                    padding_value=pad_token_id,
                )
                if max_length is not None:
                    output_batch[key] = output_batch[key][..., :max_length]
                # cast to long
                output_batch[key] = output_batch[key].long()
            elif key == "answer_choices_ids":
                flat_answer_choice_ids = [
                    choice for example in batch for choice in example[key]
                ]
                num_choice = [len(example[key]) for example in batch]
                if max(num_choice) == 0:
                    continue
                # if max(num_choice) != min(num_choice) or max(num_choice) == 0:
                #     continue
                #     raise NotImplementedError(
                #         "The collate_fn is not implmented for variable number of choices"
                #     )
                if max(num_choice) != min(num_choice):
                    # print(f"Encountered variable number of choices: {num_choice}")
                    padded_choices = []
                    for example in batch:
                        example_choices = example[key]
                        while len(example_choices) < max(num_choice):
                            example_choices.append(torch.tensor([pad_token_id]))
                        padded_choices.extend(example_choices)
                    flat_answer_choice_ids = padded_choices
                flat_answer_choices_ids = torch.nn.utils.rnn.pad_sequence(
                    flat_answer_choice_ids, batch_first=True, padding_value=pad_token_id
                )
                output_batch[key] = flat_answer_choices_ids.view(
                    len(batch), max(num_choice), -1
                )
                if max_length is not None:
                    output_batch[key] = output_batch[key][..., :max_length]
            elif key == "label":
                output_batch[key] = torch.cat([example[key] for example in batch])
            else:
                output_batch[key] = [example[key] for example in batch]

        return output_batch

    return collate_fn


class P3CLDataModule:
    def __init__(
            self, config, tokenizer, loggers, stage="train,val,test", is_moe=False, **kwargs
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.logger = loggers["logger"]
        if len(config.dataset) == 1 and config.dataset[0] in TAG2TASK_LIST.keys():
            self.all_tasks = TAG2TASK_LIST[config.dataset[0]]
        else:
            self.all_tasks = config.dataset
        self.logger.info(f"Tasks\t{self.all_tasks}")

        self.task_modules = {}
        for data_tag in self.all_tasks:
            data_config = (
                FULL_DATASET_CONFIGS_MOE[data_tag] if is_moe else FULL_DATASET_CONFIGS[data_tag]
            )
            for split in data_config:
                data_config[split].update(kwargs)
            if data_tag in P3_TASKS:
                self.task_modules[data_tag] = P3DataModule(
                    data_tag,
                    tokenizer,
                    config,
                    data_config=data_config,
                    logger=self.logger,
                    stage=stage,
                )
            else:
                self.task_modules[data_tag] = FlatDataModule(
                    data_tag,
                    tokenizer,
                    config,
                    data_config=data_config,
                    logger=self.logger,
                    stage=stage,
                )

    def __call__(self, data_tag):
        return self.task_modules[data_tag]

    def __getitem__(self, data_tag):
        return self.task_modules[data_tag]


class BBCLDataModule:
    def __init__(
            self, config, tokenizer, loggers, stage="val", is_moe=False, **kwargs
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.logger = loggers["logger"]
        if len(config.dataset) == 1 and config.dataset[0] in TAG2TASK_LIST.keys():
            self.all_tasks = TAG2TASK_LIST[config.dataset[0]]
        else:
            self.all_tasks = config.dataset
        self.logger.info(f"Tasks\t{self.all_tasks}")

        self.task_modules = {}
        for data_tag in self.all_tasks:
            data_config = (
                BB_DATASET_CONFIGS_MOE[data_tag]
                if is_moe else BB_DATASET_CONFIGS[data_tag]
            )
            for split in data_config:
                data_config[split].update(kwargs)
            self.task_modules[data_tag] = BBDataModule(
                data_tag,
                tokenizer,
                config,
                data_config=data_config,
                logger=self.logger,
                stage=stage,
            )

    def __call__(self, data_tag):
        return self.task_modules[data_tag]

    def __getitem__(self, data_tag):
        return self.task_modules[data_tag]


class FlatCLDataModule:
    def __init__(
            self, config, tokenizer, loggers, stage="train,val,test", is_moe=False, **kwargs
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.logger = loggers["logger"]
        if len(config.dataset) == 1 and config.dataset[0] in TAG2TASK_LIST.keys():
            self.all_tasks = TAG2TASK_LIST[config.dataset[0]]
        else:
            self.all_tasks = config.dataset
        self.logger.info(f"Tasks\t{self.all_tasks}")

        self.task_modules = {}
        for data_tag in self.all_tasks:
            data_config = FULL_DATASET_CONFIGS_MOE[data_tag]
            for split in data_config:
                data_config[split].update(kwargs)
            self.task_modules[data_tag] = FlatDataModule(
                data_tag,
                tokenizer,
                config,
                data_config=data_config,
                logger=self.logger,
                stage=stage,
            )

    def __call__(self, data_tag):
        return self.task_modules[data_tag]

    def __getitem__(self, data_tag):
        return self.task_modules[data_tag]


class P3DataModule:
    def __init__(
            self, data_tag, tokenizer, config, data_config, logger, stage, **kwargs
    ):
        """
        :param config:
        """
        self.data_tag = data_tag
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logger
        self.dataset_cls = FullTASK2CLASS[self.data_tag]
        self.data_config = data_config
        for split in self.data_config:
            self.data_config[split].update(kwargs)
        self.dataset = {}
        self.template = {}
        self.setup(stage)
        self.get_metric()

    def setup(self, stage):

        if "train" in stage:
            train_ds_config = {
                "name": f"{self.data_tag}_train",
                "config": self.config,
                "tokenizer": self.tokenizer,
                **self.data_config["train"],
            }
            # py: modify here with molora batchsize if training molora
            self.dataset["train"] = self.dataset_cls(**train_ds_config)
            self.template["train"] = self.dataset["train"]._templates
            self.logger.info(
                f"Train\tDataset Path: {train_ds_config['dataset_path']}\tNum Templates: {len(self.template['train'])}\t Datasize {len(self.dataset['train'])}"
            )

        if "val" in stage:
            val_ds_config = {
                "name": f"{self.data_tag}_val",
                "config": self.config,
                "tokenizer": self.tokenizer,
                **self.data_config["val"],
            }
            # py: modify here with molora batchsize if training molora
            self.dataset["val"] = self.dataset_cls(**val_ds_config)
            self.template["val"] = self.dataset["val"]._templates
            self.logger.info(
                f"Val\tDataset Path: {val_ds_config['dataset_path']}\tNum Templates: {len(self.template['val'])}\t Datasize {len(self.dataset['val'])}"
            )

        if "test" in stage:
            test_ds_config = {
                "name": f"{self.data_tag}_test",
                "config": self.config,
                "tokenizer": self.tokenizer,
                **self.data_config["test"],
            }
            # py: modify here with molora batchsize if training molora
            self.dataset["test"] = self.dataset_cls(**test_ds_config)
            self.template["test"] = self.dataset["test"]._templates
            self.logger.info(
                f"Test\tDataset Path: {test_ds_config['dataset_path']}\tNum Templates: {len(self.template['test'])}\t Datasize {len(self.dataset['test'])}"
            )

    def get_dataloader(self, dataset, shuffle=True, drop_last=True):
        generator = torch.Generator()
        generator.manual_seed(dataset.seed)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=8,
            collate_fn=create_collate_fn(
                dataset.tokenizer.pad_token_id, dataset.max_length
            ),
            generator=generator,
        )
        return data_loader

    def get_train_dataloader(self):
        return self.get_dataloader(self.dataset["train"], shuffle=True, drop_last=True)

    def get_val_dataloader(self):
        return self.get_dataloader(self.dataset["val"], shuffle=False, drop_last=False)

    def get_test_dataloader(self):
        return self.get_dataloader(self.dataset["test"], shuffle=False, drop_last=False)

    def get_metric(self):
        self.scorer = Scorer(self.data_config["val"]["metrics"])


class BBDataModule:
    def __init__(
            self, data_tag, tokenizer, config, data_config, logger, stage, **kwargs
    ):
        """
        :param config:
        """
        self.data_tag = data_tag
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logger
        self.dataset_cls = BigBenchTASK2CLASS[self.data_tag]
        self.data_config = data_config
        for split in self.data_config:
            self.data_config[split].update(kwargs)
        self.dataset = {}
        self.template = {}
        self.setup(stage)
        self.get_metric()

    def setup(self, stage):

        if "train" in stage:
            raise ValueError("Bigbench does not have train split")

        if "val" in stage:
            raise ValueError("Bigbench does not have val split")
            # val_ds_config = {
            #     "name": f"{self.data_tag}_val",
            #     "config": self.config,
            #     "tokenizer": self.tokenizer,
            #     **self.data_config["val"],
            # }
            # # py: modify here with molora batchsize if training molora
            # self.dataset["val"] = self.dataset_cls(**val_ds_config)
            # # self.template["val"] = self.dataset["val"]._templates
            # # self.logger.info(
            # #     f"Val\tDataset Path: {val_ds_config['dataset_path']}\tNum Templates: {len(self.template['val'])}\t Datasize {len(self.dataset['val'])}"
            # # )
            # self.logger.info(
            #     f"Val\tDataset Path: {val_ds_config['dataset_path']}\t Datasize {len(self.dataset['val'])}"
            # )

        if "test" in stage:
            test_ds_config = {
                "name": f"{self.data_tag}_test",
                "config": self.config,
                "tokenizer": self.tokenizer,
                **self.data_config["test"],
            }
            # py: modify here with molora batchsize if training molora
            self.dataset["test"] = self.dataset_cls(**test_ds_config)
            # self.template["test"] = self.dataset["test"]._templates
            # self.logger.info(
            #     f"Test\tDataset Path: {test_ds_config['dataset_path']}\tNum Templates: {len(self.template['test'])}\t Datasize {len(self.dataset['test'])}"
            # )
            self.logger.info(
                f"Test\tDataset Path: {test_ds_config['dataset_path']}\t Datasize {len(self.dataset['test'])}"
            )

    def get_dataloader(self, dataset, shuffle=True, drop_last=True):
        generator = torch.Generator()
        generator.manual_seed(dataset.seed)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=8,
            collate_fn=create_collate_fn(dataset.tokenizer.pad_token_id, dataset.max_length),
            generator=generator,
        )
        return data_loader

    def get_train_dataloader(self):
        raise ValueError("Bigbench does not have train split")

    def get_val_dataloader(self):
        raise ValueError("Bigbench does not have val split")

    def get_test_dataloader(self):
        return self.get_dataloader(self.dataset["test"], shuffle=False, drop_last=False)

    def get_metric(self):
        if "val" in self.data_config:
            self.scorer = Scorer(self.data_config["val"]["metrics"])
        else:
            self.scorer = Scorer(self.data_config["test"]["metrics"])


class FlatDataModule:
    def __init__(
            self, data_tag, tokenizer, config, data_config, logger, stage, **kwargs
    ):
        """
        :param config:
        """
        self.data_tag = data_tag
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logger
        self.dataset_cls = FullTASK2CLASS[self.data_tag]
        self.data_config = data_config
        for split in self.data_config:
            self.data_config[split].update(kwargs)
        self.dataset = {}
        self.setup(stage)
        self.get_metric()

    def setup(self, stage):

        if "train" in stage:
            train_ds_config = {
                "name": f"{self.data_tag}_train",
                "config": self.config,
                "tokenizer": self.tokenizer,
                **self.data_config["train"],
            }
            # py: modify here with molora batchsize if training molora
            self.dataset["train"] = self.dataset_cls(**train_ds_config)
            self.logger.info(
                f"Train\tDataset Path: {train_ds_config['dataset_path']}\t Datasize {len(self.dataset['train'])}"
            )

        if "val" in stage:
            val_ds_config = {
                "name": f"{self.data_tag}_val",
                "config": self.config,
                "tokenizer": self.tokenizer,
                **self.data_config["val"],
            }
            # py: modify here with molora batchsize if training molora
            self.dataset["val"] = self.dataset_cls(**val_ds_config)
            self.logger.info(
                f"Val\tDataset Path: {val_ds_config['dataset_path']}\t Datasize {len(self.dataset['val'])}"
            )

        if "test" in stage:
            test_ds_config = {
                "name": f"{self.data_tag}_test",
                "config": self.config,
                "tokenizer": self.tokenizer,
                **self.data_config["test"],
            }
            # py: modify here with molora batchsize if training molora
            self.dataset["test"] = self.dataset_cls(**test_ds_config)
            self.logger.info(
                f"Test\tDataset Path: {test_ds_config['dataset_path']}\t Datasize {len(self.dataset['test'])}"
            )

    def get_dataloader(self, dataset, shuffle=True, drop_last=True):
        generator = torch.Generator()
        generator.manual_seed(dataset.seed)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=8,
            collate_fn=create_collate_fn(
                dataset.tokenizer.pad_token_id, dataset.max_length
            ),
            generator=generator,
        )
        return data_loader

    def get_train_dataloader(self):
        return self.get_dataloader(self.dataset["train"], shuffle=True, drop_last=True)

    def get_val_dataloader(self):
        return self.get_dataloader(self.dataset["val"], shuffle=False, drop_last=False)

    def get_test_dataloader(self):
        return self.get_dataloader(self.dataset["test"], shuffle=False, drop_last=False)

    def get_metric(self):
        self.scorer = Scorer(self.data_config["val"]["metrics"])
