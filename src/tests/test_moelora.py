# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/3/2

import unittest
from copy import deepcopy

import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

from peft import MoLoRAConfig, get_peft_model, LoraConfig
from peft.tuners.molora.utils import extend_molora_experts
from utils.manipulation import gather_lora_state_dicts_to_molora, get_state_dict_for_final_checkpoint


class T5MoLoRAInferenceTest(unittest.TestCase):
    def setUp(self):
        self.num_experts = 4
        self.config = MoLoRAConfig(
            lora_dim=16,
            lora_alpha=1,
            target_modules="all-linear",
            lora_dropout=0,
            bias="none",
            moe_num_experts=self.num_experts,
            moe_router_aux_loss_coef=0,
            top_k=2,
            task_type="SEQ_2_SEQ_LM",
        )

        self.tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

    def test_inference(self):
        model = get_peft_model(self.model, self.config)
        input_text = "Jeder ist der Andere und Keiner er selbst."
        target_text = "All are the other and none is himself."
        inputs = self.tokenizer(
            [input_text, input_text], return_tensors="pt", padding=True, truncation=True
        )
        inputs["labels"] = self.tokenizer(
            [target_text, target_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )["input_ids"]
        outputs = model(**inputs)
        self.assertEqual(
            outputs.logits.shape[:-1],
            inputs["labels"].shape,
        )

    def test_top_p_inference(self):
        config = MoLoRAConfig(
            lora_dim=16,
            lora_alpha=1,
            target_modules="all-linear",
            lora_dropout=0,
            bias="none",
            moe_num_experts=self.num_experts,
            moe_router_aux_loss_coef=0,
            top_p=0.5,
            task_type="SEQ_2_SEQ_LM",
        )
        model = get_peft_model(self.model, config)
        input_text = "Jeder ist der Andere und Keiner er selbst."
        target_text = "All are the other and none is himself."
        inputs = self.tokenizer(
            [input_text, input_text], return_tensors="pt", padding=True, truncation=True
        )
        inputs["labels"] = self.tokenizer(
            [target_text, target_text], return_tensors="pt", padding=True, truncation=True,
        )["input_ids"]
        outputs = model(**inputs)
        self.assertEqual(
            outputs.logits.shape[:-1], inputs["labels"].shape,
        )

    def test_gather_state_dicts(self):
        lora_config = LoraConfig(
            target_modules="all-linear",
            r=16,
            lora_alpha=1,
            lora_dropout=0,
            init_lora_weights=True,
            use_rslora=False,
            inference_mode=False,
            bias="none",
        )
        molora_model = get_peft_model(deepcopy(self.model), self.config)

        lora_state_dict_list = [
            get_peft_model(deepcopy(self.model), lora_config).state_dict()
            for _ in range(self.num_experts)
        ]  # List[Dict[str, torch.Tensor]]
        gathered_state_dict = gather_lora_state_dicts_to_molora(lora_state_dict_list)

        load_missed = molora_model.load_state_dict(gathered_state_dict, strict=True)

        self.assertEqual(len(load_missed.missing_keys), 0)

    def test_extend_new_expert(self):
        model = get_peft_model(self.model, self.config)
        model.apply_layer_norm_to_gate()
        model.reset_routing_counter()

        for _ in range(3):
            input_ids = torch.randint(10, 1000, (2, 5))
            model(input_ids=input_ids, labels=input_ids)

        usage_statistics = model.get_routing_counter_state_dict(normalize=True)
        model = extend_molora_experts(
            model,
            new_expert_task_names="new_expert",
            num_new_experts=1,
            lora_init_method="top-2-used",
            gate_init_method="top-2-used",
            usage_statistics=usage_statistics,
        )
        input_ids = torch.randint(10, 1000, (2, 5))
        model(input_ids=input_ids, labels=input_ids)

    def test_layer_norm_gate(self):
        model = get_peft_model(self.model, self.config)
        model.apply_layer_norm_to_gate()

    def test_final_checkpoint(self):
        model = get_peft_model(self.model, self.config)

        final_state_dict = get_state_dict_for_final_checkpoint(model, num_experts_not_to_save=2)


if __name__ == "__main__":
    unittest.main()
