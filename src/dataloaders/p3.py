import json

import numpy as np
import torch
from jinja2 import Template
from promptsource.templates import DatasetTemplates

from .dataset import Dataset, InterfaceInfo


# from .constants import *


def find_label(target, answer_choices):
    if len(answer_choices) == 0:
        return -1
    else:
        matched_len = [
            (
                (target[: len(choice)] == choice[: len(target)]).long().sum(),
                -len(choice),
                idx,
            )
            for idx, choice in enumerate(answer_choices)
        ]
        return max(matched_len)[2]


class P3Dataset(Dataset):
    def __init__(
            self,
            include_templates="all",
            ignore_templates=None,
            max_pretemplate_examples_per_dataset=None,
            round_robin_template=False,
            t0_instruction=None,
            chatgpt_instruction=None,
            chatgpt_instruction_v2=None,
            **kwargs,
    ):
        """
        include_templates: list, str
            list: list of the template names to use.
            str: "original" to use templates for original task, or "all" to use all templates.
            when using str, you can also specify ignore_templates.
        ignore_templates: list
            list of indices of the templates to ignore.
        max_pretemplate_examples_per_dataset: int
            Maximum number of examples to use from the dataset, before applying tempaltes. Useful for few-shot learning.
        """
        self.include_templates = include_templates
        self.ignore_templates = ignore_templates if ignore_templates is not None else []
        self.max_pretemplate_examples_per_dataset = max_pretemplate_examples_per_dataset
        self.round_robin_template = round_robin_template
        self.interface_info = InterfaceInfo(**kwargs)
        self.t0_instruction = t0_instruction
        self.chatgpt_instruction = chatgpt_instruction
        self.chatgpt_instruction_v2 = chatgpt_instruction_v2
        super().__init__(**kwargs)

    def _get_templates(self, templates, include_templates, ignore_templates):
        if isinstance(include_templates, list):
            templates = [
                templates[template_name]
                for template_name in templates.all_template_names
                if template_name in include_templates
            ]
        elif isinstance(include_templates, str) and include_templates == "all":
            templates = [
                templates[template_name]
                for template_name in templates.all_template_names
                if template_name not in ignore_templates
            ]
        elif isinstance(include_templates, str) and include_templates == "original":
            templates = [
                templates[template_name]
                for template_name in templates.all_template_names
                if templates[template_name].metadata.original_task
                   and template_name not in ignore_templates
            ]
        return templates

    def load_data(self):
        super().load_data()
        self._templates = self._get_templates(
            DatasetTemplates(*self.dataset_path[1:]),
            self.include_templates,
            self.ignore_templates,
        )

    def process_data(self):
        self._examples = [example for example in self._examples]

    def truncate_dataset(self):
        _rng = np.random.default_rng(self.seed)
        if (
                self.max_pretemplate_examples_per_dataset is not None
                and len(self._examples) > self.max_pretemplate_examples_per_dataset
        ):
            self._examples = _rng.choice(
                self._examples,
                self.max_pretemplate_examples_per_dataset,
                replace=False,
            ).tolist()
        if self.max_examples_per_dataset is not None:
            if (
                    len(self._examples) * len(self._templates)
                    > self.max_examples_per_dataset
            ):
                all_example_template_idx_tuples = list(
                    range(len(self._examples) * len(self._templates))
                )
                self._example_template_idx_tuples = _rng.choice(
                    all_example_template_idx_tuples,
                    self.max_examples_per_dataset,
                    replace=False,
                ).tolist()
            else:
                self._example_template_idx_tuples = list(
                    range(len(self._examples) * len(self._templates))
                )

    def __len__(self):
        if self.max_examples_per_dataset is not None:
            return len(self._example_template_idx_tuples)
        elif self.round_robin_template:
            return len(self._examples)
        else:
            return len(self._examples) * len(self._templates)

    def __getitem__(self, idx):
        if self.max_examples_per_dataset is not None:
            idx = self._example_template_idx_tuples[idx]
            example_idx = idx // len(self._templates)
        elif self.round_robin_template:
            example_idx = idx
        else:
            example_idx = idx // len(self._templates)
        template_idx = idx % len(self._templates)
        example = self._examples[example_idx]
        template = self._templates[template_idx]
        try:
            inputs_and_targets = template.apply(example)
        except:
            print(f"Error in applying template {template_idx} to example {example_idx}")
            inputs_and_targets = []
        if len(inputs_and_targets) == 2:
            input_str, target_str = template.apply(example)
            if target_str == "":
                target_str = "<NO LABEL>"
        else:
            input_str, target_str = "<NO INPUT>", "<NO INPUT>"
        answer_choices = template.get_answer_choices_list(example)
        if answer_choices is None:
            answer_choices = []
        input_ids = self.tokenize(input_str)
        target_ids = self.tokenize(target_str)
        answer_choices_ids = [
            self.tokenize(answer_choice) for answer_choice in answer_choices
        ]
        label = find_label(target_ids, answer_choices_ids)

        label = torch.LongTensor([label])
        tokenized_example = {
            "example_idx": example_idx,
            "template_idx": template_idx,
            "input_str": input_str,
            "target_str": target_str,
            "answer_choices": answer_choices,
            "input_ids": input_ids,
            "target_ids": target_ids,
            "answer_choices_ids": answer_choices_ids,
            "label": label,
            "references": example.get("references", []),
        }
        tokenized_example = {
            k: v for k, v in tokenized_example.items() if v is not None
        }
        # add additional keys to tokenized_example
        tokenized_example.update(super().__getitem__(idx))
        tokenized_example.update({f"_{key}": value for key, value in example.items()})
        return tokenized_example


class P3MnliDataset(P3Dataset):
    def load_data(self):
        import pyarrow  # noqa: F401 # to get PowerPC to work
        from datasets import load_dataset as load_huggingface_dataset

        if self.split == "train":
            examples = load_huggingface_dataset(*self.dataset_path[1:], split="train")
            self._examples = self.train_val_split(
                examples, self.split, self.config.num_val_samples, self.seed
            )

        elif self.split == "validation":
            examples = load_huggingface_dataset(*self.dataset_path[1:], split="train")
            self._examples = self.train_val_split(
                examples, self.split, self.config.num_val_samples, self.seed
            )

        elif self.split == "test":
            self._examples = load_huggingface_dataset(
                *self.dataset_path[1:], split="validation_matched"
            )

        else:
            self._examples = load_huggingface_dataset(
                *self.dataset_path[1:], split=self.split
            )

        self._templates = self._get_templates(
            DatasetTemplates(*self.dataset_path[1:]),
            self.include_templates,
            self.ignore_templates,
        )


class P3StoryClozeDataset(P3Dataset):
    def load_data(self):
        super().load_data()
        self._templates = self._get_templates(
            DatasetTemplates("story_cloze", "2016"),
            self.include_templates,
            self.ignore_templates,
        )


class P3WikiHopDataset(P3Dataset):
    def load_data(self):
        import pyarrow  # noqa: F401 # to get PowerPC to work
        from datasets import (
            load_dataset as load_huggingface_dataset)

        examples = load_huggingface_dataset(
            *self.dataset_path[1:], split=self.load_split, trust_remote_code=True
        ).rename_column("query", "question")
        if self.stage == "validation":
            self._examples = self.split_dataset(
                examples, "selected", self.seed, self.config.val_fraction
            )
        elif self.stage == "test":
            self._examples = self.split_dataset(
                examples, "rest", self.seed, self.config.val_fraction
            )
        else:
            self._examples = examples

        self._templates = self._get_templates(
            DatasetTemplates("wiki_hop", "original"),
            self.include_templates,
            self.ignore_templates,
        )


class P3AdversarialQADataset(P3Dataset):
    def process_data(self):
        self._examples = [example for example in self._examples]
        for idx, example in enumerate(self._examples):
            example["references"] = example["answers"]["text"]
            example["answer_start"] = example["answers"]["answer_start"]


class P3CommonGenDataset(P3Dataset):
    def process_data(self):
        self._examples = [example for example in self._examples]
        for idx, example in enumerate(self._examples):
            example["references"] = example["target"]


class P3MultinewsDataset(P3Dataset):
    def process_data(self):
        self._examples = [example for example in self._examples]
        for idx, example in enumerate(self._examples):
            example["references"] = [[example["summary"]]]


class P3HotpotQADataset(P3Dataset):
    def process_data(self):
        self._examples = [example for example in self._examples]
        for idx, example in enumerate(self._examples):
            example["references"] = [[example["answer"]]]


# In the modified `process_data` method, for each example that lacks an answer (i.e., when `no_answer` is true),
# a template is selected in a cyclic manner based on the current example's index. This selected template is
# then rendered with the `no_answer` flag set to True, producing an answer. This rendered answer replaces the
# original answer in the example.


class P3DuorcDataset(P3Dataset):
    def process_data(self):
        self._examples = [example for example in self._examples]
        for idx, example in enumerate(self._examples):
            if example["no_answer"]:
                template_idx = idx % len(self._templates)
                template = self._templates[template_idx]
                if template.metadata.original_task:
                    jinja_template = Template(template.jinja)
                    rendered_output = jinja_template.render(no_answer=True)
                    split_output = rendered_output.split("|||", 1)
                    if len(split_output) == 2:
                        _, answer = split_output
                    else:
                        answer = ""
                    example["answers"] = [answer.strip()]
            example["references"] = example["answers"]
            example["answer_start"] = [-1]


class P3RopesDataset(P3Dataset):
    def process_data(self):
        self._examples = [example for example in self._examples]
        for idx, example in enumerate(self._examples):
            example["references"] = example["answers"]["text"]
            example["answer_start"] = [0]


class P3WikiBioDataset(P3Dataset):
    def process_data(self):
        self._examples = [example for example in self._examples]
        for idx, example in enumerate(self._examples):
            example["references"] = [[example["target_text"]]]


class P3CNNDailyMailDataset(P3Dataset):
    def process_data(self):
        self._examples = [example for example in self._examples]
        for idx, example in enumerate(self._examples):
            example["references"] = [[example["highlights"]]]


class P3SamsumDataset(P3Dataset):
    def process_data(self):
        self._examples = [example for example in self._examples]
        for idx, example in enumerate(self._examples):
            example["references"] = [[example["summary"]]]


class P3QuorefDataset(P3Dataset):
    def process_data(self):
        self._examples = [example for example in self._examples]
        for idx, example in enumerate(self._examples):
            example["references"] = example["answers"]["text"]
            example["answer_start"] = example["answers"]["answer_start"]


class P3GigaWordDataset(P3Dataset):
    def process_data(self):
        self._examples = [example for example in self._examples]
        for idx, example in enumerate(self._examples):
            example["references"] = [[example["summary"]]]


class P3XSumDataset(P3Dataset):
    def process_data(self):
        self._examples = [example for example in self._examples]
        for idx, example in enumerate(self._examples):
            example["references"] = [[example["summary"]]]


class P3AppReviewsDataset(P3Dataset):
    def load_data(self):
        if self.dataset_path[0] == "huggingface":
            import pyarrow  # noqa: F401 # to get PowerPC to work
            from datasets import load_dataset as load_huggingface_dataset

            examples = load_huggingface_dataset(
                *self.dataset_path[1:], split=self.load_split
            )
            if self.stage == "train":
                examples = self.split_dataset(
                    examples, return_split="rest", seed=self.seed, val_fraction=10000
                )
            else:
                examples = self.split_dataset(
                    examples,
                    return_split="selected",
                    seed=self.seed,
                    val_fraction=10000,
                )

            if self.stage == "validation":
                self._examples = self.split_dataset(
                    examples, "selected", self.seed, self.config.val_fraction
                )
            elif self.stage == "test":
                self._examples = self.split_dataset(
                    examples, "rest", self.seed, self.config.val_fraction
                )
            else:
                self._examples = examples
            # TODO: it takes a lot of time for FLAN datasets, we can do this inside P3 datasets if needed
            # self._examples = [example for example in self._examples]
        elif self.dataset_path[0] == "datalabs":
            from datalabs import load_dataset as load_datalabs_dataset

            # TODO: add support for train_validation split later when needed.
            self._examples = load_datalabs_dataset(
                *self.dataset_path[1:], split=self.split
            )
            self._examples = [example for example in self._examples]
        else:
            raise NotImplementedError(
                "If you want to use a custom dataset, you need to implement the load_data method in the child class."
            )

        self._templates = self._get_templates(
            DatasetTemplates(*self.dataset_path[1:]),
            self.include_templates,
            self.ignore_templates,
        )


class TrecTemplate(object):
    def __init__(self):
        self.answer_choices = [
            "Abbreviation.",
            "Expression abbreviated.",
            "Animal.",
            "Organ of body.",
            "Color.",
            "Invention, book and other creative piece.",
            "Currency name.",
            "Disease and medicine.",
            "Event.",
            "Food.",
            "Musical instrument.",
            "Language.",
            "Letter like a-z.",
            "Other entity.",
            "Plant.",
            "Product.",
            "Religion.",
            "Sport.",
            "Element and substance.",
            "Symbols and sign.",
            "Techniques and method.",
            "Equivalent term.",
            "Vehicle.",
            "Word with a special property.",
            "Definition of something.",
            "Description of something.",
            "Manner of an action.",
            "Reason.",
            "Group or organization of persons",
            "Individual.",
            "Title of a person.",
            "Description of a person.",
            "City.",
            "Country.",
            "Mountain.",
            "Other location.",
            "State.",
            "Postcode or other code.",
            "Number of something.",
            "Date.",
            "Distance, linear measure.",
            "Price.",
            "Order, rank.",
            "Other number.",
            "Lasting time of something.",
            "Percent, fraction.",
            "Speed.",
            "Temperature.",
            "Size, area and volume.",
            "Weight.",
        ]

    def apply(self, example):
        input_str = f"Question: {example['text']}\nAnswer choices: {self.answer_choices}\nAnswer:"
        return [input_str, self.answer_choices[example["fine_label"]]]

    def get_answer_choices_list(self, example):
        return self.answer_choices


class P3TrecDataset(P3Dataset):
    # P3 templates are not working for this dataset
    def process_data(self):
        self._examples = [example for example in self._examples]
        self._templates = [TrecTemplate()]


class CBKGenTemplate(object):
    def __init__(self):
        self.answer_choices = []

    def apply(self, example):
        return [example["src"], example["tgt"]]

    def get_answer_choices_list(self, example):
        return self.answer_choices


class CBKGenDataset(P3Dataset):
    def load_data(self):
        self._examples = []
        import datasets

        for json_file in self.dataset_path:
            with open(json_file, "r") as f:
                data = json.load(f)
            dataset = datasets.Dataset.from_dict(data)
            self._examples.extend([example for example in dataset])

    def process_data(self):
        self._templates = [CBKGenTemplate()]
        for idx, example in enumerate(self._examples):
            example["references"] = [example["tgt"]]


class CovidqaDataset(CBKGenDataset):
    def process_data(self):
        super().process_data()
        rng = np.random.RandomState(1234)
        rng.shuffle(self._examples)
        if "EVAL" in self.name:
            self._examples = self._examples[0:500]
        else:
            self._examples = self._examples[500:]


class P3WebQuestionsDataset(P3Dataset):
    def process_data(self):
        self._examples = [example for example in self._examples]
        for example in self._examples:
            example["references"] = example["answers"]


class LudwigTemplate(object):
    def __init__(self, template_num=0):
        self.template_num = template_num
        self.answer_choices = ["yes", "no"]

    def apply(self, example):
        if self.template_num == 0:
            input_str = f"Does the following response to the question imply yes or no?\nquestion: {example['utterance']}\nresponse: {example['response']}\nimplicature:"
        elif self.template_num == 1:
            input_str = f"Finish the following text:\nEsther asked \"{example['utterance']}\" and Juan responded \"{example['response']}\", which means"
        elif self.template_num == 2:
            input_str = f"Is the implied meaning of the following response yes or no:\nquestion: {example['utterance']}\nresponse: {example['response']}\nmeaning:"
        elif self.template_num == 3:
            input_str = f"What is the intent of the following response, yes or no?\nquestion: {example['utterance']}\nresponse: {example['response']}\nintent:"
        elif self.template_num == 4:
            input_str = f"Finish the following text:\nKaren asked \"{example['utterance']}\" and William responded \"{example['response']}\", which means"
        elif self.template_num == 5:
            input_str = f"Finish the following text:\nBob asked \"{example['utterance']}\" and Alice responded \"{example['response']}\", which means"
        target_str = example["implicature"]
        return [input_str, target_str]

    def get_answer_choices_list(self, example):
        return self.answer_choices


class LudwigDataset(P3Dataset):
    def load_data(self):
        super().load_data()
        self._templates = [LudwigTemplate(i) for i in range(6)]
