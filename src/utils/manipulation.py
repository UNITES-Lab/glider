# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/3/2
import os
import re
from collections import OrderedDict
from typing import Dict, List, Optional

import torch
from torch import linalg as LA
from torch.nn import LayerNorm
from tqdm import tqdm

__all__ = [
    "convert_phatgoose_p3_checkpoint_to_peft_lora",
    "gather_lora_state_dicts_to_molora",
    "convert_phatgoose_full_checkpoint_to_peft_lora",
]

p3_dataset_dict = {
    "P3Agnews": "P3AGNEWS",
    "P3Amazonpolarity": "P3AMAZONPOLARITY",
    "P3Cosmosqa": "P3COSMOSQA",
    "P3Samsum": "P3SAMSUM",
    "P3Quartz": "P3QUARTZ",
    "P3Ropes": "P3ROPES",
    "P3Wikibio": "P3WIKIBIO",
    "P3Paws": "P3PAWS",
    "P3Wikiqa": "P3WIKIQA",
    "P3Socialiqa": "P3SOCIALIQA",
    "P3Qasc": "P3QASC",
    "P3Quail": "P3QUAIL",
    "P3Dream": "P3DREAM",
    "P3Wiqa": "P3WIQA",
    "P3Quarel": "P3QUAREL",
    "P3Sciq": "P3SCIQ",
    "P3Quoref": "P3QUOREF",
    "P3Duorc": "P3DUORC",
    "P3Rottentomatoes": "P3ROTTENTOMATOES",
    "P3Yelp": "P3YELP",
    "P3Commongen": "P3COMMONGEN",
    "P3Gigaword": "P3GIGAWORD",
    "P3Xsum": "P3XSUM",
    "P3Mrpc": "P3MRPC",
    "P3Qqp": "P3QQP",
    "P3Commonsenseqa": "P3COMMONSENSEQA",
    "P3Cose": "P3COSE",
    "P3Wikihop": "P3WIKIHOP",
    "P3Hotpotqa": "P3HOTPOTQA",
    "P3Appreviews": "P3APPREVIEWS",
    "P3Trec": "P3TREC",
    "P3Multinews": "P3MULTINEWS",
    "P3Imdb": "P3IMDB",
    "P3Adversarialqa": "P3ADVERSARIALQA",
    "P3Cnndailymail": "P3CNNDAILYMAIL",
    "P3Dbpedia14": "P3DBPEDIA14",
}

flanv2_dataset_dict = {
    "Flanv2Ai2arceasy": "FLAN2021AI2ARCEASY/ZS",
    "Flanv2Ai2arcchallenge": "FLAN2021AI2ARCCHALLENGE/ZS",
    "Flanv2Algebralinear1d": "FLAN2021ALGEBRALINEAR1D/ZS",
    "Flanv2Boolq": "FLAN2021BOOLQ/ZS",
    "Flanv2Coqa": "FLAN2021COQA/ZS",
    "Flanv2Defpronounresolution": "FLAN2021DEFPRONOUNRESOLUTION/ZS",
    "Flanv2Drop": "FLAN2021DROP/ZS",
    "Flanv2Fixpunct": "FLAN2021FIXPUNCT/ZS",
    "Flanv2GemDart": "FLAN2021GEMDART/ZS",
    "Flanv2Geme2enlg": "FLAN2021GEME2ENLG/ZS",
    "Flanv2Gemwebnlgen": "FLAN2021GEMWEBNLGEN/ZS",
    "Flanv2Gemwikilinguaen": "FLAN2021GEMWIKILINGUAEN/ZS",
    "Flanv2Gluesst2": "FLAN2021GLUESST2/ZS",
    "Flanv2Gluecola": "FLAN2021GLUECOLA/ZS",
    "Flanv2Gluemnli": "FLAN2021GLUEMNLI/ZS",
    "Flanv2Glueqnli": "FLAN2021GLUEQNLI/ZS",
    "Flanv2Gluestsb": "FLAN2021GLUESTSB/ZS",
    "Flanv2Gluewnli": "FLAN2021GLUEWNLI/ZS",
    "Flanv2Lambada": "FLAN2021LAMBADA/ZS",
    "Flanv2Naturalquestionsopen": "FLAN2021NATURALQUESTIONSOPEN/ZS",
    "Flanv2Newsroom": "FLAN2021NEWSROOM/ZS",
    "Flanv2Openbookqa": "FLAN2021OPENBOOKQA/ZS",
    "flanv2opinionabstractsidebate": "FLAN2021OPINIONABSTRACTSIDEBATE/ZS",
    "Flanv2Opinionabstractrottentomatoes": "FLAN2021OPINIONABSTRACTSROTTENTOMATOES/ZS",
    "Flanv2Paracrawlenes": "FLAN2021PARACRAWLENES/ZS",
    "Flanv2Piqa": "FLAN2021PIQA/ZS",
    "Flanv2Quac": "FLAN2021QUAC/ZS",
    "Flanv2Sentiment140": "FLAN2021SENTIMENT140/ZS",
    "Flanv2Snli": "FLAN2021SNLI/ZS",
    "Flanv2Squad": "FLAN2021SQUAD/ZS",
    "Flanv2Supergluemultirc": "FLAN2021SUPERGLUEMULTIRC/ZS",
    "Flanv2Supergluerecord": "FLAN2021SUPERGLUERECORD/ZS",
    "Flanv2Triviaqa": "FLAN2021TRIVIAQA/ZS",
    "Flanv2Truecase": "FLAN2021TRUECASE/ZS",
    "Flanv2Unifiedqascienceinst": "FLAN2021UNIFIEDQASCIENCEINST/ZS",
    "Flanv2Wordsegment": "FLAN2021WORDSEGMENT/ZS",
}

niv2_dataset_dict = {
    "Niv2Translation": "NIV2TRANSLATION/ZS",
    "Niv2Programexecution": "NIV2PROGRAMEXECUTION/ZS",
    "Niv2Questiongeneration": "NIV2QUESTIONGENERATION/ZS",
    "Niv2Sentimentanalysis": "NIV2SENTIMENTANALYSIS/ZS",
    "Niv2Textcategorization": "NIV2TEXTCATEGORIZATION/ZS",
    "Niv2Textmatching": "NIV2TEXTMATCHING/ZS",
    "Niv2Toxiclanguagedetection": "NIV2TOXICLANGUAGEDETECTION/ZS",
    "Niv2Causeeffectclassification": "NIV2CAUSEEFFECTCLASSIFICATION/ZS",
    "Niv2Informationextraction": "NIV2INFORMATIONEXTRACTION/ZS",
    "Niv2Textualentailment": "NIV2TEXTUALENTAILMENT/ZS",
    "Niv2Wrongcandidategeneration": "NIV2WRONGCANDIDATEGENERATION/ZS",
    "Niv2Namedentityrecognition": "NIV2NAMEDENTITYRECOGNITION/ZS",
    "Niv2Commonsenseclassification": "NIV2COMMONSENSECLASSIFICATION/ZS",
    "Niv2Fillintheblank": "NIV2FILLINTHEBLANK/ZS",
    "Niv2Textcompletion": "NIV2TEXTCOMPLETION/ZS",
    "Niv2Sentencecomposition": "NIV2SENTENCECOMPOSITION/ZS",
    "Niv2Titlegeneration": "NIV2TITLEGENERATION/ZS",
    "Niv2Languageidentification": "NIV2LANGUAGEIDENTIFICATION/ZS",
    "Niv2Questionunderstanding": "NIV2QUESTIONUNDERSTANDING/ZS",
    "Niv2Sentenceperturbation": "NIV2SENTENCEPERTURBATION/ZS",
    "Niv2Answerabilityclassification": "NIV2ANSWERABILITYCLASSIFICATION/ZS",
    "Niv2Summarization": "NIV2SUMMARIZATION/ZS",
    "Niv2Coreferenceresolution": "NIV2COREFERENCERESOLUTION/ZS",
    "Niv2Textqualityevaluation": "NIV2TEXTQUALITYEVALUATION/ZS",
    "Niv2Texttocode": "NIV2TEXTTOCODE/ZS",
    "Niv2Paraphrasing": "NIV2PARAPHRASING/ZS",
    "Niv2Dialoguegeneration": "NIV2DIALOGUEGENERATION/ZS",
    "Niv2Questionrewriting": "NIV2QUESTIONREWRITING/ZS",
    "Niv2Wordsemantics": "NIV2WORDSEMANTICS/ZS",
    "Niv2Postagging": "NIV2POSTAGGING/ZS",
    "Niv2Linguisticprobing": "NIV2LINGUISTICPROBING/ZS",
    "Niv2Storycomposition": "NIV2STORYCOMPOSITION/ZS",
    "Niv2Speakeridentification": "NIV2SPEAKERIDENTIFICATION/ZS",
    "Niv2Wordanalogy": "NIV2WORDANALOGY/ZS",
    "Niv2Datatotext": "NIV2DATATOTEXT/ZS",
    "Niv2Stereotypedetection": "NIV2STEREOTYPEDETECTION/ZS",
    "Niv2Negotiationstrategydetection": "NIV2NEGOTIATIONSTRATEGYDETECTION/ZS",
    "Niv2Dialogueactrecognition": "NIV2DIALOGUEACTRECOGNITION/ZS",
    "Niv2Genderclassification": "NIV2GENDERCLASSIFICATION/ZS",
    "Niv2Coherenceclassification": "NIV2COHERENCECLASSIFICATION/ZS",
    "Niv2Explanation": "NIV2EXPLANATION/ZS",
    "Niv2Ethicsclassification": "NIV2ETHICSCLASSIFICATION/ZS",
    "Niv2Wordrelationclassification": "NIV2WORDRELATIONCLASSIFICATION/ZS",
    "Niv2Sentenceordering": "NIV2SENTENCEORDERING/ZS",
    "Niv2Answerverification": "NIV2ANSWERVERIFICATION/ZS",
    "Niv2Mathematics": "NIV2MATHEMATICS/ZS",
    "Niv2Intentidentification": "NIV2INTENTIDENTIFICATION/ZS",
    "Niv2Keywordtagging": "NIV2KEYWORDTAGGING/ZS",
    "Niv2Codetotext": "NIV2CODETOTEXT/ZS",
    "Niv2Dialoguestatetracking": "NIV2DIALOGUESTATETRACKING/ZS",
    "Niv2Textsimplification": "NIV2TEXTSIMPLIFICATION/ZS",
    "Niv2Stancedetection": "NIV2STANCEDETECTION/ZS",
    "Niv2Factverification": "NIV2FACTVERIFICATION/ZS",
    "Niv2Grammarerrordetection": "NIV2GRAMMARERRORDETECTION/ZS",
    "Niv2Sectionclassification": "NIV2SECTIONCLASSIFICATION/ZS",
    "Niv2Numberconversion": "NIV2NUMBERCONVERSION/ZS",
    "Niv2Styletransfer": "NIV2STYLETRANSFER/ZS",
    "Niv2Speakerrelationclassification": "NIV2SPEAKERRELATIONCLASSIFICATION/ZS",
    "Niv2Ironydetection": "NIV2IRONYDETECTION/ZS",
    "Niv2Questiondecomposition": "NIV2QUESTIONDECOMPOSITION/ZS",
    "Niv2Overlapextraction": "NIV2OVERLAPEXTRACTION/ZS",
    "Niv2Grammarerrorcorrection": "NIV2GRAMMARERRORCORRECTION/ZS",
    "Niv2Spellingerrordetection": "NIV2SPELLINGERRORDETECTION/ZS",
    "Niv2Entitygeneration": "NIV2ENTITYGENERATION/ZS",
    "Niv2Sentenceexpansion": "NIV2SENTENCEEXPANSION/ZS",
    "Niv2Discourseconnectiveidentification": "NIV2DISCOURSECONNECTIVEIDENTIFICATION/ZS",
    "Niv2Discourserelationclassification": "NIV2DISCOURSERELATIONCLASSIFICATION/ZS",
    "Niv2Poemgeneration": "NIV2POEMGENERATION/ZS",
    "Niv2Entityrelationclassification": "NIV2ENTITYRELATIONCLASSIFICATION/ZS",
    "Niv2Punctuationerrordetection": "NIV2PUNCTUATIONERRORDETECTION/ZS",
    "Niv2Spamclassification": "NIV2SPAMCLASSIFICATION/ZS",
    "Niv2Paperreview": "NIV2PAPERREVIEW/ZS",
    "Niv2Sentencecompression": "NIV2SENTENCECOMPRESSION/ZS",
    "Niv2Prepositionprediction": "NIV2PREPOSITIONPREDICTION/ZS",
    "Niv2Misc": "NIV2MISC/ZS",
}

add1_dataset_dict = {
    "P3Wscfixed": "P3WSCFIXED",
    "P3Copa": "P3COPA",
    "P3Hswag": "P3HSWAG",
    "P3Wic": "P3WIC",
    "P3Racehigh": "P3RACEHIGH",
    "P3Racemiddle": "P3RACEMIDDLE",
    "P3Webquestions": "P3WEBQUESTIONS",
    "Flanv2Qrecc": "DIALOGQRECC/ZS",
    "Flanv2Wikidialog": "DIALOGWIKIDIALOG/ZS",
    "Flanv2Qreccii": "DIALOGQRECCII/ZS",
    "Flanv2Wikidialogii": "DIALOGWIKIDIALOGII/ZS",
    "Flanv2Aeslc": "FLAN2021AESLC/ZS",
    "Flanv2Wmt16translatecsen": "FLAN2021WMT16TRANSLATECSEN/ZS",
    "Flanv2Wmt16translatedeen": "FLAN2021WMT16TRANSLATEDEEN/ZS",
    "Flanv2Wmt16translateruen": "FLAN2021WMT16TRANSLATERUEN/ZS",
    "Flanv2Wmt16translatefien": "FLAN2021WMT16TRANSLATEFIEN/ZS",
    "Flanv2Wmt16translateroen": "FLAN2021WMT16TRANSLATEROEN/ZS",
    "Flanv2Wmt16translatetren": "FLAN2021WMT16TRANSLATETREN/ZS",
    "Flanv2Wmt14translatefren": "FLAN2021WMT14TRANSLATEFREN/ZS",
}

all_dataset_dict = {**p3_dataset_dict, **flanv2_dataset_dict, **niv2_dataset_dict}
full_dataset_dict = {
    **p3_dataset_dict,
    **flanv2_dataset_dict,
    **niv2_dataset_dict,
    **add1_dataset_dict,
}


def convert_phatgoose_p3_checkpoint_to_peft_lora(
        checkpoint_path: str,
        output_dir: str,
        adapter_name: Optional[str] = "default",
):
    """
    Convert the state_dict of PhatGoose's LoRA to PEFT's LoRA.

    Keys in PhatGoose are like:
        -   'encoder.embed_tokens.weight'
        -   'encoder.block.0.layer.0.SelfAttention.q._addons.router.expert_embeddings__0'
        -   'encoder.block.0.layer.0.SelfAttention.q._addons.expert_lora.layer1__0'
        -   'encoder.block.0.layer.0.SelfAttention.q._addons.expert_lora.layer2__0'

    Corresponding keys in PEFT are like:
        -   'base_model.model.encoder.embed_tokens.weight'
        -   'base_model.model.encoder.block.0.layer.0.SelfAttention.q.lora_gate.default'
        -   'base_model.model.encoder.block.0.layer.0.SelfAttention.q.lora_A.default.weight'
        -   'base_model.model.encoder.block.0.layer.0.SelfAttention.q.lora_B.default.weight'

    """

    if not os.path.exists(output_dir):
        print(f"Creating directory {output_dir}")
        os.makedirs(output_dir)

    assert ("p3" in checkpoint_path.lower()
            ), f"checkpoint_path is supposed to contain 'p3' but got {checkpoint_path}."
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    distributed_state_dict_list = [OrderedDict() for _ in range(len(p3_dataset_dict))]

    for key, value in tqdm(state_dict.items(), desc="Converting PhatGoose's LoRA to PEFT's LoRA"):
        if "router.expert_embeddings" in key:
            # lora gate weight
            prefix = key.split("._addons.router")[0]
            expert_idx = int(key.split("expert_embeddings__")[-1])
            new_key = f"base_model.model.{prefix}.lora_gate.{adapter_name}"
            distributed_state_dict_list[expert_idx][new_key] = value
        elif "expert_lora.layer1" in key:
            # lora_A weight
            prefix = key.split("._addons.expert_lora.layer1")[0]
            expert_idx = int(key.split("expert_lora.layer1__")[-1])
            new_key = f"base_model.model.{prefix}.lora_A.{adapter_name}.weight"
            distributed_state_dict_list[expert_idx][new_key] = value.transpose(0, 1)
        elif "expert_lora.layer2" in key:
            # lora_B weight
            prefix = key.split("._addons.expert_lora.layer2")[0]
            expert_idx = int(key.split("expert_lora.layer2__")[-1])
            new_key = f"base_model.model.{prefix}.lora_B.{adapter_name}.weight"
            distributed_state_dict_list[expert_idx][new_key] = value.transpose(0, 1)
        else:
            # for other weight
            for idx in range(len(distributed_state_dict_list)):
                distributed_state_dict_list[idx][f"base_model.model.{key}"] = value

    task_name_list = list(p3_dataset_dict.keys())
    for idx, distributed_state_dict in enumerate(
            tqdm(distributed_state_dict_list, desc="Saving PEFT's LoRA")
    ):
        task_name = task_name_list[idx].lower()
        torch.save(distributed_state_dict, os.path.join(output_dir, f"{task_name}_best.pt"))

        distributed_state_dict_list[idx] = None
        del distributed_state_dict


def convert_phatgoose_full_checkpoint_to_peft_lora(
        checkpoint_path: str,
        output_dir: str,
        adapter_name: Optional[str] = "default",
):
    """
    Convert the state_dict of PhatGoose's LoRA to PEFT's LoRA.

    Keys in PhatGoose are like:
        -   'encoder.embed_tokens.weight'
        -   'encoder.block.0.layer.0.SelfAttention.q._addons.router.expert_embeddings__0'
        -   'encoder.block.0.layer.0.SelfAttention.q._addons.expert_lora.layer1__0'
        -   'encoder.block.0.layer.0.SelfAttention.q._addons.expert_lora.layer2__0'

    Corresponding keys in PEFT are like:
        -   'base_model.model.encoder.embed_tokens.weight'
        -   'base_model.model.encoder.block.0.layer.0.SelfAttention.q.lora_gate.default'
        -   'base_model.model.encoder.block.0.layer.0.SelfAttention.q.lora_A.default.weight'
        -   'base_model.model.encoder.block.0.layer.0.SelfAttention.q.lora_B.default.weight'

    """

    if not os.path.exists(output_dir):
        print(f"Creating directory {output_dir}")
        os.makedirs(output_dir)

    print(f"Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    distributed_state_dict_list = [OrderedDict() for _ in range(len(full_dataset_dict))]

    for key, value in tqdm(state_dict.items(), desc="Converting PhatGoose's LoRA to PEFT's LoRA"):
        if "router.expert_embeddings" in key:
            # lora gate weight
            prefix = key.split("._addons.router")[0]
            expert_idx = int(key.split("expert_embeddings__")[-1])
            new_key = f"base_model.model.{prefix}.lora_gate.{adapter_name}"
            distributed_state_dict_list[expert_idx][new_key] = value
        elif "expert_lora.layer1" in key:
            # lora_A weight
            prefix = key.split("._addons.expert_lora.layer1")[0]
            expert_idx = int(key.split("expert_lora.layer1__")[-1])
            new_key = f"base_model.model.{prefix}.lora_A.{adapter_name}.weight"
            distributed_state_dict_list[expert_idx][new_key] = value.transpose(0, 1)
        elif "expert_lora.layer2" in key:
            # lora_B weight
            prefix = key.split("._addons.expert_lora.layer2")[0]
            expert_idx = int(key.split("expert_lora.layer2__")[-1])
            new_key = f"base_model.model.{prefix}.lora_B.{adapter_name}.weight"
            distributed_state_dict_list[expert_idx][new_key] = value.transpose(0, 1)
        else:
            # for other weight
            for idx in range(len(distributed_state_dict_list)):
                distributed_state_dict_list[idx][f"base_model.model.{key}"] = value

    task_name_list = list(full_dataset_dict.keys())
    for idx, distributed_state_dict in enumerate(
            tqdm(distributed_state_dict_list, desc="Saving PEFT's LoRA")
    ):
        task_name = task_name_list[idx].lower()
        torch.save(distributed_state_dict, os.path.join(output_dir, f"{task_name}_best.pt"))

        distributed_state_dict_list[idx] = None
        del distributed_state_dict


def gather_lora_state_dicts_to_molora(
        state_dict_list: List[Dict[str, torch.Tensor]],
        cl_state_dict=None,
        norm_router_weight: Optional[bool] = True,
        adapter_name: Optional[str] = "default",
) -> Dict[str, torch.Tensor]:
    """
    Gather the state_dicts of LoRA checkpoints to MoLoRA layers.

    Parameters
    ----------
    state_dict_list: List[Dict[str, torch.Tensor]]
        The list of state_dicts of LoRA layers.
    norm_router_weight: Optional[bool]
        Whether to normalize the router weight. Default is True in PHATGOOSE.
    adapter_name: Optional[str]
        The name of the adapter. Default is "default" in PEFT.

    Returns
    -------
    Dict[str, torch.Tensor]
        The state_dict of the MoLoRA model.

    """

    final_state_dict = OrderedDict()
    num_experts = len(state_dict_list)

    for idx, lora_state_dict in enumerate(
            tqdm(state_dict_list, desc=f"Gathering {num_experts} LoRA state_dicts to MoLoRA")
    ):
        for key, value in lora_state_dict.items():
            if ".base_layer." in key:
                # for pretrained weight of LoRAed layers, we remove the ".base_layer." in the key
                new_key = ".".join(key.split(".base_layer."))
                if new_key in final_state_dict:
                    assert torch.allclose(
                        final_state_dict[new_key], value), f"key: {new_key} has different value in layer {idx}"
                else:
                    final_state_dict[new_key] = value
            elif f".lora_A.{adapter_name}" in key:
                # for lora_A weight
                prefix = key.split(".lora_A.")[0]
                new_key = f"{prefix}.lora_experts.{adapter_name}.{idx}.lora_A.weight"
                final_state_dict[new_key] = value
            elif f".lora_B.{adapter_name}" in key:
                # for lora_B weight
                prefix = key.split(".lora_B.")[0]
                new_key = f"{prefix}.lora_experts.{adapter_name}.{idx}.lora_B.weight"
                final_state_dict[new_key] = value
            elif f".lora_gate.{adapter_name}" in key:
                # for lora_gate weight
                new_key = f"{key}.weight"

                if idx == 0:
                    final_state_dict[new_key] = torch.zeros_like(value).repeat(num_experts, 1)

                final_state_dict[new_key][idx] = value

            else:
                # for other weight
                if idx == 0:
                    final_state_dict[key] = value
                else:
                    assert torch.allclose(
                        final_state_dict[key], value), f"key: {key} has different value in layer {idx}"

        state_dict_list[idx] = None
        del lora_state_dict

    if cl_state_dict:
        final_state_dict.update(cl_state_dict)

    if norm_router_weight:
        for key, value in final_state_dict.items():
            if ".lora_gate." in key:
                ln = LayerNorm(value.shape[-1], elementwise_affine=False)
                final_state_dict[key] = ln(final_state_dict[key])
    return final_state_dict


def get_state_dict_for_final_checkpoint(
        model,
        num_experts_not_to_save: int,
        adapter_name: Optional[str] = "default",
):
    all_lora_state_dict = {}

    for name, param in model.state_dict().items():
        if "lora_A" in name or "lora_B" in name:
            re_search = re.search(rf"\.{adapter_name}\.(\d+)\.lora_([AB])\.", name)
            if re_search is None:
                raise ValueError(f"Unexpected key: {name}")
            expert_idx = int(re_search.group(1))
            if expert_idx < num_experts_not_to_save:
                continue
            all_lora_state_dict[name] = param.cpu()
        elif "embed" in name or "lora_gate" in name:
            all_lora_state_dict[name] = param.cpu()

    return all_lora_state_dict


def gather_auto_encoder_state_dicts_to_molora(
        state_dict_list: List[Dict[str, torch.Tensor]],
        adapter_name: Optional[str] = "default",
) -> Dict[str, torch.Tensor]:
    final_state_dict = OrderedDict()
    num_experts = len(state_dict_list)

    for idx, lora_state_dict in enumerate(
            tqdm(state_dict_list, desc=f"Gathering {num_experts} LoRA AE state_dicts to auto-encoders")
    ):
        for key, value in lora_state_dict.items():
            if ".input_auto_encoder." in key:
                prefix, post_fix = key.split(".input_auto_encoder.")
                post_fix = post_fix.split(f"{adapter_name}.")[-1]
                new_key = f"{prefix}.lora_gate.{adapter_name}.auto_encoders.{idx}.{post_fix}"
                final_state_dict[new_key] = value
            else:
                raise ValueError(f"Unexpected key: {key}")

        state_dict_list[idx] = None
        del lora_state_dict

    return final_state_dict


def create_arrow_gate_from_lora(
        lora_state_dict: Dict[str, torch.Tensor],
        adapter_name: Optional[str] = "default",
) -> Dict[str, torch.Tensor]:
    for key, value in lora_state_dict.items():
        if ".lora_gate." in key:
            lora_A_weight = lora_state_dict[key.replace(
                f"lora_gate.{adapter_name}", f"lora_A.{adapter_name}.weight")].cuda()
            lora_B_weight = lora_state_dict[key.replace(
                f"lora_gate.{adapter_name}", f"lora_B.{adapter_name}.weight")].cuda()
            lora_weight = torch.matmul(lora_B_weight, lora_A_weight)
            # SVD
            dtype = lora_state_dict[key].dtype
            U, S, V = LA.svd(lora_weight.float())
            lora_state_dict[key] = V[:, 0].to(dtype).cpu()
    return lora_state_dict
