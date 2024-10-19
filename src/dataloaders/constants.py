from copy import deepcopy

from .bigbench import *
from .flat import *
from .p3 import *

FLAT_DATASET_CONFIGS = {
    "c4": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_length": 512,
            "max_examples_per_dataset": 50_000,
            "input_field": "text",
            "target_field": "text",
            "dataset_path": [
                "huggingface",
                "allenai/c4",
                "en",
            ],  # a placeholder that is not used
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 32,
            "max_length": 512,
            "max_examples_per_dataset": 50_000,
            "metrics": ["custom"],
            "interface": "gen",
            "input_field": "text",
            "target_field": "text",
            "dataset_path": ["huggingface", "allenai/c4", "en"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 32,
            "max_length": 512,
            "max_examples_per_dataset": 50_000,
            "metrics": ["custom"],
            "interface": "gen",
            "input_field": "text",
            "target_field": "text",
            "dataset_path": ["huggingface", "allenai/c4", "en"],
        },
    },
}

P3_DATASET_CONFIGS = {
    "p3paws": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_length": 512,
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "paws", "labeled_final"],
            "ignore_templates": ["paraphrase-task"],
            "include_templates": "all",
            "t0_instruction": "Determine if the following two sentences paraphrase each other or not.",
            "chatgpt_instruction": "Determine whether the two provided sentences are paraphrases of each other by answering 'Yes' or 'No'.",
            "chatgpt_instruction_v2": "Generate a response determining whether the provided input sentences or scenarios are equivalent or describe the same situation.",
            "chatgpt_instruction_v3": "Evaluate whether two given sentences are paraphrases of each other, demonstrating an understanding of paraphrase detection and linguistic equivalence.",
            "chatgpt_instruction_v4": "Determine if the given pairs of sentences or statements are paraphrases of each other or if a given statement is true or false based on the information provided. This task requires the language model to have strong paraphrasing recognition and factual verification skills.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 16,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "paws", "labeled_final"],
            "ignore_templates": ["paraphrase-task"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 16,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "paws", "labeled_final"],
            "ignore_templates": ["paraphrase-task"],
        },
    },
    "p3cnndailymail": {
        "train": {
            "split": "train",
            "batch_size": 1,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "cnn_dailymail", "3.0.0"],
            "t0_instruction": "Condense the article down to the essentials to present it in the form of short cards in mobile news apps.",
            "chatgpt_instruction": "Summarize the main points of the article concisely while retaining key details and implications.",
            "chatgpt_instruction_v2": "Summarize the article by extracting key points and presenting them concisely with emphasis on main events and factual details.",
            "chatgpt_instruction_v3": "Summarize the main points of a given article into concise bullet points that highlight the essential information, suitable for use in brief news updates or mobile news apps.",
            "chatgpt_instruction_v4": "Generate concise, multi-point summaries of news articles, demonstrating knowledge of summarization techniques to extract and condense key information while maintaining the essence of the original content.",
        },
        "val": {
            "split": "validation-validation",
            "metrics": ["bleu", "rouge"],
            "interface": "gen",
            "max_gen_length": 128,
            "batch_size": 1,
            "max_length": 512,
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "cnn_dailymail", "3.0.0"],
        },
        "test": {
            "split": "test-validation",
            "metrics": ["bleu", "rouge"],
            "interface": "gen",
            "max_gen_length": 128,
            "batch_size": 1,
            "max_length": 512,
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "cnn_dailymail", "3.0.0"],
        },
    },
    "p3wikiqa": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "wiki_qa"],
            "ignore_templates": [
                "Direct Answer to Question",
                "Generate Question from Topic",
                "Jeopardy style",
                "Topic Prediction - Answer Only",
                "Topic Prediction - Question Only",
                "Topic Prediction - Question and Answer Pair",
            ],
            "max_length": 512,
            "include_templates": "all",
            "t0_instruction": "Determine the topic of the question-answer pair",
            "chatgpt_instruction": "Determine if the provided answer is correct for the given question and respond with 'Yes' or 'No'.",
            "chatgpt_instruction_v2": "Determine if the provided information correctly answers the question. If yes, respond with 'True', otherwise respond with 'False'.",
            "chatgpt_instruction_v3": "Generate a 'yes' or 'no' response to validate the correctness of the provided answer based on its relevance and accuracy to the given question, specifically requiring knowledge in historical events, geography, and culinary topics.",
            "chatgpt_instruction_v4": "Determine if the provided suggestion or answer correctly addresses the given question, based on factual accuracy and relevance. Respond with 'True' if the suggestion is a correct answer, otherwise respond with 'False'.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 16,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "wiki_qa"],
            "ignore_templates": [
                "Direct Answer to Question",
                "Generate Question from Topic",
                "Jeopardy style",
                "Topic Prediction - Answer Only",
                "Topic Prediction - Question Only",
                "Topic Prediction - Question and Answer Pair",
            ],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 16,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "wiki_qa"],
            "ignore_templates": [
                "Direct Answer to Question",
                "Generate Question from Topic",
                "Jeopardy style",
                "Topic Prediction - Answer Only",
                "Topic Prediction - Question Only",
                "Topic Prediction - Question and Answer Pair",
            ],
        },
    },
    "p3ropes": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "ropes"],
            "max_length": 512,
            "include_templates": "all",
            "t0_instruction": "Answer correctly the following question related to the paragraph below.",
            "chatgpt_instruction": "Given a description of Hepatitis B and a list of children arranged by age, determine which child has a greater chance of developing a chronic Hepatitis B infection based on their relative age.",
            "chatgpt_instruction_v2": "Based on the information about the ages of individuals and their risk of chronic Hepatitis B infection as described, determine which individual has a higher chance of developing a chronic infection.",
            "chatgpt_instruction_v3": "Generate the correct answer to a question by applying given background knowledge to a described situation, requiring comprehension and logical reasoning skills.",
            "chatgpt_instruction_v4": "Generate an answer to a question by applying relevant knowledge from the provided background information to the given situation.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 4,
            "max_length": 512,
            "metrics": ["squad"],
            "interface": "gen",
            "max_gen_length": 16,
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "ropes"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 4,
            "max_length": 512,
            "metrics": ["squad"],
            "interface": "gen",
            "max_gen_length": 16,
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "ropes"],
        },
    },
    "p3agnews": {
        "train": {
            "split": "train",
            "batch_size": 8,
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "ag_news"],
            "max_length": 512,
            "include_templates": "all",
            "t0_instruction": "Determine which section of a newspaper this article would likely appear in.",
            "chatgpt_instruction": "Generate a label that best categorizes the content of the provided news article.",
            "chatgpt_instruction_v2": "Determine the most relevant category or audience for a given text snippet.",
            "chatgpt_instruction_v3": "Classify news articles into appropriate categories based on their content, demonstrating knowledge in distinguishing topics such as sports, world politics, business, and science and technology.",
            "chatgpt_instruction_v4": "Identify the category or target audience for the given text based on its content. Knowledge required: ability to classify text into appropriate categories such as sports, business, politics, or science and technology.",
        },
        "val": {
            "split": "validation-test",
            "batch_size": 8,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "ag_news"],
        },
        "test": {
            "split": "test-test",
            "batch_size": 8,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "ag_news"],
        },
    },
    "p3amazonpolarity": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "amazon_polarity"],
            "max_length": 512,
            "include_templates": "all",
            "t0_instruction": "Determine this product review convey a negative or positive sentiment.",
            "chatgpt_instruction": "Determine whether the sentiment expressed in the provided product review is positive, negative, or if it increases or decreases the likelihood of purchasing the product, and respond accordingly.",
            "chatgpt_instruction_v2": "Determine whether the review expresses a positive or negative sentiment towards the product, and answer 'Yes' or 'No' accordingly.",
            "chatgpt_instruction_v3": "Determine the sentiment or recommendation implied in a product review by analyzing the review content and providing a straightforward yes/no or satisfied/dissatisfied response.",
            "chatgpt_instruction_v4": "Analyze the sentiment or purchasing likelihood from product reviews to determine whether the sentiment is positive, negative, or if the review would increase or decrease the chances of purchasing the product.",
        },
        "val": {
            "split": "validation-test",
            "batch_size": 4,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "amazon_polarity"],
            "max_pretemplate_examples_per_dataset": 10_000,
        },
        "test": {
            "split": "test-test",
            "batch_size": 4,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "amazon_polarity"],
            "max_pretemplate_examples_per_dataset": 10_000,
        },
    },
    "p3wikibio": {
        "train": {
            "split": "train",
            "batch_size": 2,
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "wiki_bio"],
            "max_length": 512,
            "include_templates": "all",
            "t0_instruction": "Based on these bullet points, write a short biography describing the life.",
            "chatgpt_instruction": "Given a list of bullet points with personal and professional details, write a concise biography.",
            "chatgpt_instruction_v2": "Generate a short biography using the provided factual bullet points about an individual's life.",
            "chatgpt_instruction_v3": "Create a short biography using the provided facts, demonstrating knowledge in historical and biographical writing.",
            "chatgpt_instruction_v4": "Write a short biography based on the given factual bullet points, demonstrating proficiency in summarizing and transforming structured data into coherent narrative text.",
        },
        "val": {
            "split": "validation-val",
            "batch_size": 2,
            "max_length": 512,
            "metrics": ["bleu", "rouge"],
            "interface": "gen",
            "max_gen_length": 256,
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "wiki_bio"],
            "max_pretemplate_examples_per_dataset": 10_000,
        },
        "test": {
            "split": "test-val",
            "batch_size": 2,
            "max_length": 512,
            "metrics": ["bleu", "rouge"],
            "interface": "gen",
            "max_gen_length": 256,
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "wiki_bio"],
            "max_pretemplate_examples_per_dataset": 10_000,
        },
    },
    "p3socialiqa": {
        "train": {
            "split": "train",
            "batch_size": 32,
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "social_i_qa"],
            "max_length": 512,
            "include_templates": "all",
            "t0_instruction": "Determine the best answer for the question according to the context",
            "chatgpt_instruction": "Given the context provided in the input, generate a concise response that directly addresses the question asked.",
            "chatgpt_instruction_v2": "Based on the provided context, predict the next action or emotional response of a character.",
            "chatgpt_instruction_v3": "Generate a brief, contextually relevant response that describes an individual's state, others' feelings, or necessary preceding actions based on given contextual information. The task requires understanding of human emotions, logical sequencing, and contextual interpretation.",
            "chatgpt_instruction_v4": "Generate a response that infers the most likely action or emotion of a character in a given scenario. The task requires understanding of context and human social interactions.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 32,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "ignore_templates": ["Check if a random answer is valid or not"],
            "include_templates": "original",
            "dataset_path": ["huggingface", "social_i_qa"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 32,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "ignore_templates": ["Check if a random answer is valid or not"],
            "include_templates": "original",
            "dataset_path": ["huggingface", "social_i_qa"],
        },
    },
    "p3cosmosqa": {
        "train": {
            "split": "train",
            "batch_size": 8,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "cosmos_qa"],
            "t0_instruction": "Generate a question based on the context and the answer.",
            "chatgpt_instruction": "Read a given passage and multiple-choice question, then select the most appropriate answer from the provided options.",
            "chatgpt_instruction_v2": "Given a text passage and a follow-up question, select the most appropriate answer from the provided options.",
            "chatgpt_instruction_v3": "Using your understanding of context and inference, read the given passage and choose the best option to answer the question based on the provided multiple-choice options.",
            "chatgpt_instruction_v4": "Given a text passage and a multiple-choice question related to the passage, select the best answer based on the context. This task requires comprehension and inference skills to understand the context and determine the most appropriate response from the provided options.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 8,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "cosmos_qa"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 8,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "cosmos_qa"],
        },
    },
    "p3quail": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "quail"],
            "t0_instruction": "Read the following context and choose the correct option to answer the question.",
            "chatgpt_instruction": "Read the provided text and select the best answer from the given options to the question about the text.",
            "chatgpt_instruction_v2": "Generate the instruction to command a language model for selecting the correct answer from multiple-choice options based on a provided context and question.",
            "chatgpt_instruction_v3": "Generate a task to evaluate the understanding of a provided text by selecting the most appropriate answer from multiple-choice options, requiring comprehension and inference skills.",
            "chatgpt_instruction_v4": "Generate a task instruction for a language model to accurately answer multiple-choice questions based on a provided context. The task requires the model to comprehend the context and use information extraction skills to identify and select the correct answer from the given options.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 4,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "quail"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 4,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "quail"],
        },
    },
    "p3quartz": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "quartz"],
            "t0_instruction": "Use information from the paragraph to answer the question.",
            "chatgpt_instruction": "Generate a concise answer by interpreting the specific details given in a text passage related to a posed question.",
            "chatgpt_instruction_v2": "Given a text and a question, provide the correct answer based solely on the information provided in the text.",
            "chatgpt_instruction_v3": "Answer the questions based on the provided text by identifying the relevant information that leads to the correct comparison or conclusion. The task requires the ability to comprehend and extract key details from the text to accurately answer comparative or situational questions.",
            "chatgpt_instruction_v4": "Generate a short and accurate answer to the given question by comprehending and interpreting the provided text. This task requires skills in reading comprehension and logical inference.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 16,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "quartz"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 16,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "quartz"],
        },
    },
    "p3qasc": {
        "train": {
            "split": "train",
            "batch_size": 32,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "qasc"],
            "t0_instruction": "Choose the best answer given the fact and the question.",
            "chatgpt_instruction": "Given a set of facts or a quiz question, select the most appropriate answer from a list of options provided.",
            "chatgpt_instruction_v2": "Given a set of facts and multiple answer choices, identify the correct answer based on the information provided.",
            "chatgpt_instruction_v3": "Generate the most relevant answer by extracting and synthesizing key information from the given facts or hints, focusing on comprehension and inference skills.",
            "chatgpt_instruction_v4": "Using knowledge of scientific facts and logical reasoning, determine the most accurate answer to a question from a list of options based on provided information.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 32,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "qasc"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 32,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "qasc"],
        },
    },
    "p3commongen": {
        "train": {
            "split": "train",
            "batch_size": 32,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "common_gen"],
            "t0_instruction": "Construct a sentence given the words.",
            "chatgpt_instruction": "Generate a coherent sentence using the provided abstract concepts.",
            "chatgpt_instruction_v2": "Generate a coherent sentence using the provided list of concepts.",
            "chatgpt_instruction_v3": "Generate a coherent sentence using all the given abstract concepts, requiring the skill of concept integration to form a meaningful sentence.",
            "chatgpt_instruction_v4": "Generate a coherent sentence by creatively combining a given set of abstract concepts.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 32,
            "max_length": 512,
            "metrics": ["bleu", "rouge"],
            "interface": "gen",
            "max_gen_length": 64,
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "common_gen"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 32,
            "max_length": 512,
            "metrics": ["bleu", "rouge"],
            "interface": "gen",
            "max_gen_length": 64,
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "common_gen"],
        },
    },
    "p3adversarialqa": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "adversarial_qa", "adversarialQA"],
            "t0_instruction": "Extract the answer to the question from the following context",
            "chatgpt_instruction": "Generate an answer to the question provided, using information found in the accompanying passage.",
            "chatgpt_instruction_v2": "Generate a concise answer to a specific question using information directly extracted from a given passage.",
            "chatgpt_instruction_v3": "Given a passage, identify and extract specific information to answer a question based on details found within the text. This task requires comprehension and information retrieval skills.",
            "chatgpt_instruction_v4": "Given a passage, identify and extract the relevant information to answer a specified question. Note that the answer can be directly found within the text.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 4,
            "max_length": 512,
            "metrics": ["squad"],
            "interface": "gen",
            "max_gen_length": 128,
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "adversarial_qa", "adversarialQA"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 4,
            "max_length": 512,
            "metrics": ["squad"],
            "interface": "gen",
            "max_gen_length": 128,
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "adversarial_qa", "adversarialQA"],
        },
    },
    "p3appreviews": {
        "train": {
            "split": "train",
            "batch_size": 8,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "app_reviews"],
            "t0_instruction": "Decide whether you would recommend this app to a friend given the review.",
            "chatgpt_instruction": "Given a user review of an app, determine how likely you would recommend the app to a friend, choosing from the options 'Not at all', 'No', 'Maybe', 'Yes', or 'Definitely'.",
            "chatgpt_instruction_v2": "Given a user review, decide whether to recommend the app to a friend by choosing from the options: Not at all, No, Maybe, Yes, or Definitely.",
            "chatgpt_instruction_v3": "Given a user review of an app, classify the level of recommendation to a friend (Not at all, No, Maybe, Yes, or Definitely) based on the sentiment and content of the review. This task requires knowledge of sentiment analysis and understanding of user feedback.",
            "chatgpt_instruction_v4": "Given a user's review of an app, determine the likelihood of recommending the app to a friend by selecting from the options: Not at all, No, Maybe, Yes, or Definitely. This task requires sentiment analysis and understanding of user feedback to accurately assess the recommendation level.",
        },
        "val": {
            "split": "validation-train",
            "batch_size": 8,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "ignore_templates": [
                "generate_review",
                "convert_to_rating",
                "convert_to_star_rating",
                "generate_review",
            ],
            "include_templates": "all",
            "dataset_path": ["huggingface", "app_reviews"],
        },
        "test": {
            "split": "test-train",
            "batch_size": 8,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "ignore_templates": [
                "generate_review",
                "convert_to_rating",
                "convert_to_star_rating",
                "generate_review",
            ],
            "include_templates": "all",
            "dataset_path": ["huggingface", "app_reviews"],
        },
    },
    "p3commonsenseqa": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "commonsense_qa"],
            "t0_instruction": "Select the most suitable answer for the following question given the options.",
            "chatgpt_instruction": "Answer the following question with a brief and precise response.",
            "chatgpt_instruction_v2": "Provide a concise phrase that answers the question posed in the input.",
            "chatgpt_instruction_v3": "Use common sense reasoning to answer the following questions in a concise manner.",
            "chatgpt_instruction_v4": "Generate a concise response that answers the question directly, demonstrating an understanding of contextual cues and specific knowledge.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 16,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "commonsense_qa"],
            "max_pretemplate_examples_per_dataset": 10_000,
        },
        "test": {
            "split": "test-validation",
            "batch_size": 16,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "commonsense_qa"],
            "max_pretemplate_examples_per_dataset": 10_000,
        },
    },
    "p3cose": {
        "train": {
            "split": "train",
            "batch_size": 32,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "cos_e", "v1.11"],
            "t0_instruction": "Choose the most suitable option to answer the above question.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 32,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "cos_e", "v1.11"],
            "max_pretemplate_examples_per_dataset": 10_000,
        },
        "test": {
            "split": "test-validation",
            "batch_size": 32,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "cos_e", "v1.11"],
            "max_pretemplate_examples_per_dataset": 10_000,
        },
    },
    "p3dbpedia14": {
        "train": {
            "split": "train",
            "batch_size": 2,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "dbpedia_14"],
            "t0_instruction": """Pick one category for the following text. The options are: company, 
            educational institution, artist, athlete, office holder, mean of transportation, building, natural place, 
            village, animal, plant, album, film or written work.""",
            "chatgpt_instruction": "Classify the given text into one of the following categories: company, educational institution, artist, athlete, office holder, means of transportation, building, natural place, village, animal, plant, album, film, or written work.",
            "chatgpt_instruction_v2": "Identify the category of the described item from a given list, which includes company, educational institution, artist, athlete, office holder, mean of transportation, building, natural place, village, animal, plant, album, film, or written work.",
            "chatgpt_instruction_v3": "Given a paragraph of text, identify the category it belongs to from a provided list (company, educational institution, artist, athlete, office holder, mean of transportation, building, natural place, village, animal, plant, album, film, or written work). Use knowledge of film descriptions and attributes to classify texts related to films accurately.",
            "chatgpt_instruction_v4": "Classify the given text into one of the following categories: company, educational institution, artist, athlete, office holder, mean of transportation, building, natural place, village, animal, plant, album, film, or written work, based on the contextual information provided.",
        },
        "val": {
            "split": "validation-test",
            "batch_size": 2,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "dbpedia_14"],
            "max_pretemplate_examples_per_dataset": 10_000,
        },
        "test": {
            "split": "test-test",
            "batch_size": 2,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "dbpedia_14"],
            "max_pretemplate_examples_per_dataset": 10_000,
        },
    },
    "p3dream": {
        "train": {
            "split": "train",
            "batch_size": 2,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "dream"],
            "t0_instruction": "Read the following conversation and determine the answer to the question.",
            "chatgpt_instruction": "Select the most appropriate answer based on the given dialogue and question.",
            "chatgpt_instruction_v2": "Generate a response based on the provided dialogue and a follow-up question, choosing the correct answer from a set of multiple-choice options.",
            "chatgpt_instruction_v3": "Given a dialogue and a question with multiple-choice answers, identify the correct answer based on the context of the conversation. This task requires skills in comprehension and inference.",
            "chatgpt_instruction_v4": "Generate the correct answer to a multiple-choice question about a given dialogue by interpreting the context and details of the conversation.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 2,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "dream"],
            "max_pretemplate_examples_per_dataset": 500_000,
        },
        "test": {
            "split": "test-validation",
            "batch_size": 2,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "dream"],
            "max_pretemplate_examples_per_dataset": 500_000,
        },
    },
    "p3duorc": {
        "train": {
            "split": "train",
            "batch_size": 2,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "duorc", "ParaphraseRC"],
            "t0_instruction": "Extract the answer the following question about the movie plot. If it's un-answerable, output 'No answer'.",
            "chatgpt_instruction": "Identify and output the specific answer to the given question based on the provided movie plot summary; if the question is unanswerable, specify accordingly.",
            "chatgpt_instruction_v2": "Generate trivia questions based on specific details about characters or events from a movie plot, and if the question is unanswerable given the information, prompt the model to say a specific phrase.",
            "chatgpt_instruction_v3": "Read the given movie plot summary and answer the specific question about character relationships or roles based on the information in the plot. If the answer cannot be determined from the plot, respond with 'No answer'.",
            "chatgpt_instruction_v4": "Given a detailed movie plot summary and a question about the plot, use your reading comprehension skills to find and provide the specific answer to the question based on the information in the summary. If the answer cannot be found in the summary, respond with 'No answer'.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 2,
            "max_length": 512,
            "metrics": ["squad"],
            "interface": "gen",
            "max_gen_length": 64,
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "duorc", "ParaphraseRC"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 2,
            "max_length": 512,
            "metrics": ["squad"],
            "interface": "gen",
            "max_gen_length": 64,
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "duorc", "ParaphraseRC"],
        },
    },
    "p3gigaword": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "gigaword"],
            "t0_instruction": "Generate a title for the article.",
            "chatgpt_instruction": "Write a brief summary or title for the provided article text.",
            "chatgpt_instruction_v2": "Generate a concise title for a given news article excerpt.",
            "chatgpt_instruction_v3": "Generate a concise title for a news article based on its content, demonstrating summarization and headline creation skills.",
            "chatgpt_instruction_v4": "Generate a concise and informative title for a given news article excerpt. This task requires understanding the main point of the excerpt and summarizing it accurately.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 16,
            "max_length": 512,
            "metrics": ["bleu", "rouge"],
            "interface": "gen",
            "max_gen_length": 128,
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "gigaword"],
            "max_pretemplate_examples_per_dataset": 10_000,
        },
        "test": {
            "split": "test-validation",
            "batch_size": 16,
            "max_length": 512,
            "metrics": ["bleu", "rouge"],
            "interface": "gen",
            "max_gen_length": 128,
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "gigaword"],
            "max_pretemplate_examples_per_dataset": 10_000,
        },
    },
    "p3hotpotqa": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "hotpot_qa", "fullwiki"],
            "t0_instruction": "Combine facts and formulate an answer to the question",
            "chatgpt_instruction": "Extract relevant information from the provided text to answer a specific question.",
            "chatgpt_instruction_v2": "Read the information provided in the paragraphs below and use it to answer the specific question given, then present your answer clearly along with any necessary explanations or details.",
            "chatgpt_instruction_v3": "Identify and extract specific sentences from provided information that directly support the answer to a given question, requiring knowledge of precise information retrieval and text comprehension.",
            "chatgpt_instruction_v4": "Generate concise answers or supporting sentences for given questions based on provided multi-paragraph texts, demonstrating comprehension and synthesis skills to accurately extract relevant information.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 4,
            "max_length": 512,
            "metrics": ["blue", "rouge"],
            "interface": "gen",
            "max_gen_length": 64,
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "hotpot_qa", "fullwiki"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 4,
            "max_length": 512,
            "metrics": ["blue", "rouge"],
            "interface": "gen",
            "max_gen_length": 64,
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "hotpot_qa", "fullwiki"],
        },
    },
    "p3imdb": {
        "train": {
            "split": "train",
            "batch_size": 2,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "imdb"],
            "t0_instruction": "Determine the sentiment expressed by the reviewer for the movie.",
            "chatgpt_instruction": "Determine the sentiment expressed in the movie review provided.",
            "chatgpt_instruction_v2": "Write a review summarizing your thoughts about a movie and indicate the sentiment you feel towards it.",
            "chatgpt_instruction_v3": "Analyze movie review texts to determine the sentiment expressed by the reviewer, identifying whether the sentiment is positive, negative, or otherwise characterized.",
            "chatgpt_instruction_v4": "Determine the sentiment expressed in the given movie review, which requires knowledge in sentiment analysis and understanding of context and nuances in language.",
        },
        "val": {
            "split": "validation-test",
            "batch_size": 2,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "imdb"],
        },
        "test": {
            "split": "test-test",
            "batch_size": 2,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "imdb"],
        },
    },
    "p3mrpc": {
        "train": {
            "split": "train",
            "batch_size": 8,  # fixme: 16
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "glue", "mrpc"],
            "t0_instruction": "Determine whether the following two sentences mean the same thing.",
            "chatgpt_instruction": "Determine if the two provided sentences are semantically equivalent.",
            "chatgpt_instruction_v2": "Determine if the following two sentences express the same meaning.",
            "chatgpt_instruction_v3": "Determine if the given pairs of sentences are semantically equivalent by applying knowledge of paraphrase detection.",
            "chatgpt_instruction_v4": "Determine whether the following pairs of sentences have the same meaning, utilizing skills in semantic analysis and paraphrase detection to identify if they are 'equivalent' or 'not equivalent'.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 16,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "glue", "mrpc"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 16,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "glue", "mrpc"],
        },
    },
    "p3multinews": {
        "train": {
            "split": "train",
            "batch_size": 1,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "multi_news"],
            "t0_instruction": "Write a summary of the articles.",
            "chatgpt_instruction": "Summarize multiple detailed articles into concise, focused news briefs or summaries.",
            "chatgpt_instruction_v2": "Summarize these texts into a cohesive narrative that emphasizes the main points and key details.",
            "chatgpt_instruction_v3": "Condense the provided articles into a single concise summary while preserving the essential details and context. This task requires strong summarization skills and the ability to distill complex information.",
            "chatgpt_instruction_v4": "Generate concise summaries of multiple related news articles by synthesizing key information, ensuring coverage of major events, notable quotes, and significant developments, demonstrating proficiency in summarizing and integrating content from diverse sources.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 1,
            "max_length": 512,
            "metrics": ["bleu", "rouge"],
            "interface": "gen",
            "max_gen_length": 256,
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "multi_news"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 1,
            "max_length": 512,
            "metrics": ["bleu", "rouge"],
            "interface": "gen",
            "max_gen_length": 256,
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "multi_news"],
        },
    },
    "p3qqp": {
        "train": {
            "split": "train",
            "batch_size": 8,  # fixme: 16
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "glue", "qqp"],
            "t0_instruction": "Determine whether two questions are asking the same thing.",
            "chatgpt_instruction": "Determine if two questions are duplicates or convey the same meaning, and respond with 'yes', 'no', or 'not duplicates' accordingly.",
            "chatgpt_instruction_v2": "Determine whether two questions are sufficiently similar to be considered duplicates and provide a binary 'yes' or 'no' answer.",
            "chatgpt_instruction_v3": "Determine if two questions are duplicates or convey the same meaning based on their content and context.",
            "chatgpt_instruction_v4": "Given a pair of questions, determine if they are asking the same thing by applying knowledge of semantic similarity and duplicate detection.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 16,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "glue", "qqp"],
            "max_pretemplate_examples_per_dataset": 10_000,
        },
        "test": {
            "split": "test-validation",
            "batch_size": 16,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "glue", "qqp"],
            "max_pretemplate_examples_per_dataset": 10_000,
        },
    },
    "p3quarel": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "quarel"],
            "t0_instruction": "Determine the answer to the logic test question",
            "chatgpt_instruction": "Generate a response by choosing between two provided options based on the context or explanation given in the input.",
            "chatgpt_instruction_v2": "Provide the most sensible choice between the two options provided, based on the context given in the question.",
            "chatgpt_instruction_v3": "Generate the most appropriate answer to each question by analyzing the context and understanding the concepts of resistance and distance, using the provided choices instead of the given options A and B.",
            "chatgpt_instruction_v4": "Generate the appropriate response based on physical reasoning by choosing between provided options without using labels such as ''A' or 'B'.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 16,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "all",
            "dataset_path": ["huggingface", "quarel"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 16,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "all",
            "dataset_path": ["huggingface", "quarel"],
        },
    },
    "p3quoref": {
        "train": {
            "split": "train",
            "batch_size": 2,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "quoref"],
            "t0_instruction": "Extract the answer to the question from the context.",
            "chatgpt_instruction": "Read the provided text and answer the question by identifying specific characters or entities mentioned within it.",
            "chatgpt_instruction_v2": "Extract the specific answer to a given question from the provided text passage.",
            "chatgpt_instruction_v3": "Generate answers to questions based on information extracted from provided text passages, demonstrating comprehension and information retrieval skills.",
            "chatgpt_instruction_v4": "Generate the answer to a given question by extracting specific information from a provided context, demonstrating the ability to comprehend and locate relevant details within the text.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 2,
            "max_length": 512,
            "metrics": ["squad"],
            "interface": "gen",
            "max_gen_length": 32,
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "quoref"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 2,
            "max_length": 512,
            "metrics": ["squad"],
            "interface": "gen",
            "max_gen_length": 32,
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "quoref"],
        },
    },
    "p3rottentomatoes": {
        "train": {
            "split": "train",
            "batch_size": 32,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "rotten_tomatoes"],
            "t0_instruction": "Decide the review is positive or negative.",
            "chatgpt_instruction": "Generate the sentiment expressed by the reviewer for the movie based on the provided description.",
            "chatgpt_instruction_v2": "Determine the sentiment expressed about the movie from the given text and respond with 'positive' or 'negative'.",
            "chatgpt_instruction_v3": "Identify the sentiment expressed by the reviewer for the movie, using sentiment analysis skills.",
            "chatgpt_instruction_v4": "Generate the sentiment (positive or negative) for a given movie review based on the sentiment analysis.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 32,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "rotten_tomatoes"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 32,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "rotten_tomatoes"],
        },
    },
    "p3samsum": {
        "train": {
            "split": "train",
            "batch_size": 2,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "samsum"],
            "t0_instruction": "Generate a summary for the dialogue.",
            "chatgpt_instruction": "Summarize the provided dialogue.",
            "chatgpt_instruction_v2": "Summarize the dialogue.",
            "chatgpt_instruction_v3": "Generate a concise summary of the given dialogues. The task requires the language model to demonstrate comprehension and summarization skills by extracting key information from casual conversational exchanges.",
            "chatgpt_instruction_v4": "Generate a concise summary of dialogues, showcasing the ability to extract and condense key information from conversational exchanges.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 2,
            "max_length": 512,
            "metrics": ["rouge"],
            "interface": "gen",
            "max_gen_length": 128,
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "samsum"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 2,
            "max_length": 512,
            "metrics": ["rouge"],
            "interface": "gen",
            "max_gen_length": 128,
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "samsum"],
        },
    },
    "p3sciq": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "sciq"],
            "t0_instruction": "Read the paragraph and choose the correct option from the provided answers.",
            "chatgpt_instruction": "Generate a response to a given question by extracting relevant information from the provided paragraph.",
            "chatgpt_instruction_v2": "Generate a one-word answer from a given paragraph by extracting information relevant to the specified question.",
            "chatgpt_instruction_v3": "Generate a concise answer to a given question using the information provided in a paragraph, demonstrating comprehension of biological and physiological concepts.",
            "chatgpt_instruction_v4": "Generate a one-word answer to the given question by extracting relevant information from the provided paragraph. The task requires reading comprehension and concise summarization skills.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 4,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "sciq"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 4,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "sciq"],
        },
    },
    "p3trec": {
        "train": {
            "split": "train",
            "batch_size": 1,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "trec"],
            "t0_instruction": "Determine the category that best describes the question.",
            "chatgpt_instruction": "Generate the category most suitable for the given question from a list of predefined answer choices.",
            "chatgpt_instruction_v2": "Identify and select the most appropriate category from the list of answer choices that accurately describes the information requested in the question.",
            "chatgpt_instruction_v3": "Given a question and categories, determine the most appropriate category that classifies the answer, demonstrating knowledge in categorization and context comprehension.",
            "chatgpt_instruction_v4": "Classify the question based on its context into one of the provided categories, demonstrating knowledge in identifying the nature of queries and their corresponding types.",
        },
        "val": {
            "split": "validation-test",
            "batch_size": 1,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": [],
            "dataset_path": ["huggingface", "trec"],
        },
        "test": {
            "split": "test-test",
            "batch_size": 1,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": [],
            "dataset_path": ["huggingface", "trec"],
        },
    },
    "p3wikihop": {
        "train": {
            "split": "train",
            "batch_size": 1,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "MoE-UNC/wikihop"],
            "t0_instruction": "Determine which object entity has the relation of the question.",
            "chatgpt_instruction": "Identify and select the correct entity from a list that stands in a specified relation to a given subject based on the context provided.",
            "chatgpt_instruction_v2": "Read the information provided and answer the specific relationship question about the mentioned entities by selecting the correct choice from the list given.",
            "chatgpt_instruction_v3": "Given detailed textual information about various subjects, identify the correct related entity based on specified relationships such as 'continent' for 'Blanca Peak', 'occupation' for 'Alessandro della Via', and 'military branch' for 'Paul Conrath'.",
            "chatgpt_instruction_v4": "Read the provided text and determine the correct relationship between specified entities based on given context. The task requires knowledge of historical, geographical, and biographical information to accurately identify relationships such as 'continent', 'occupation', and 'military branch'.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 1,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "MoE-UNC/wikihop"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 1,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "MoE-UNC/wikihop"],
        },
    },
    "p3wiqa": {
        "train": {
            "split": "train",
            "batch_size": 8,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "wiqa"],
            "t0_instruction": "Determine how does the supposed perturbation influence the second effect mentioned. Answer by 'more', 'less' or 'no effect'.",
            "chatgpt_instruction": "Given a detailed process description, predict how a specified change in one step of the process will affect a subsequent step. Choose from 'more', 'less', or 'no effect' to indicate the impact.",
            "chatgpt_instruction_v2": "Provide an explanation of a process and ask how a specific change in one element of that process would affect another element, offering multiple choice answers for the response.",
            "chatgpt_instruction_v3": "Generate a multiple-choice question that assesses the effect of a change in a process, based on a given sequence of events. The model should analyze the described process, identify how a specific alteration impacts the outcome, and select the correct answer from the provided options.",
            "chatgpt_instruction_v4": "Given a process description, evaluate how a specific change in conditions affects the outcome, selecting from multiple choice options (A: more, B: less, C: no effect).",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 8,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "wiqa"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 8,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "wiqa"],
        },
    },
    "p3xsum": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "xsum"],
            "t0_instruction": "Write a summary of the text.",
            "chatgpt_instruction": "Generate concise summaries for given texts.",
            "chatgpt_instruction_v2": "Generate a concise summary of the detailed information provided.",
            "chatgpt_instruction_v3": "Generate concise summaries of complex texts, leveraging advanced summarization skills to distill key information and main points accurately.",
            "chatgpt_instruction_v4": "Summarize detailed news articles into concise, clear sentences that capture the main points, demonstrating comprehension and summarization skills.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 4,
            "max_length": 512,
            "metrics": ["bleu", "rouge"],
            "interface": "gen",
            "max_gen_length": 256,
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "xsum"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 4,
            "max_length": 512,
            "metrics": ["bleu", "rouge"],
            "interface": "gen",
            "max_gen_length": 256,
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "xsum"],
        },
    },
    "p3yelp": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "yelp_review_full"],
            "t0_instruction": "Give a review score between 1 and 5.",
            "chatgpt_instruction": "Generate a star rating based on the sentiment and content of the given restaurant review.",
            "chatgpt_instruction_v2": "Generate a numerical rating based on the sentiment expressed in a detailed review of a restaurant or food item.",
            "chatgpt_instruction_v3": "Generate a review rating from 1 to 5 stars based on the sentiment and details of the given review text.",
            "chatgpt_instruction_v4": "Generate a review rating (between 1 and 5 stars) based on the provided review text, demonstrating the ability to understand and evaluate subjective feedback.",
        },
        "val": {
            "split": "validation-test",
            "batch_size": 4,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "yelp_review_full"],
        },
        "test": {
            "split": "test-test",
            "batch_size": 4,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "yelp_review_full"],
        },
    },
    # ====== belows are HELD-OUT tasks ======
    "p3rte": {
        "train": {
            "split": "train",
            "batch_size": 32,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "super_glue", "rte"],
            "t0_instruction": "Determine whether the hypothesis is true or false given the premise.",
            "chatgpt_instruction": "Determine the truthfulness of a given statement based on the provided passage, responding with 'True,' 'False,' or 'Yes,' 'No' as appropriate.",
            "chatgpt_instruction_v2": "Write a sentence to determine if a given statement is true based on the context provided.",
            "chatgpt_instruction_v3": "Determine the truthfulness of statements based on provided passages, using comprehension and inference skills to evaluate whether the statement is true, false, or can be answered with yes or no.",
            "chatgpt_instruction_v4": "Given a statement and a subsequent claim about the statement, determine whether the claim is guaranteed to be true, based on the information provided. The language model must use skills in logical reasoning and comprehension to accurately assess the truthfulness of the claim. Respond with ''Yes' or 'No' for yes/no questions, and 'True' or 'False' for true/false questions.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 32,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "include_templates": "original",
            "dataset_path": ["huggingface", "super_glue", "rte"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 32,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "include_templates": "original",
            "dataset_path": ["huggingface", "super_glue", "rte"],
        },
    },
    "p3hswag": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "hellaswag"],
            "t0_instruction": "Complete the description with an appropriate ending.",
            "chatgpt_instruction": "Complete the sentence based on the given context and topic.",
            "chatgpt_instruction_v2": "Generate a conclusion for the given scenario by selecting the most appropriate ending from the provided options.",
            "chatgpt_instruction_v3": "Complete the given sentences with the most appropriate ending based on the provided context and hint about the topic, demonstrating knowledge in contextual understanding and sentence completion.",
            "chatgpt_instruction_v4": "Generate the most appropriate sentence ending based on the given context and specified topic, ensuring logical consistency and relevance to the topic provided.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 4,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "include_templates": "original",
            "dataset_path": ["huggingface", "hellaswag"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 4,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "include_templates": "original",
            "dataset_path": ["huggingface", "hellaswag"],
        },
    },
    "p3copa": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "super_glue", "copa"],
            "t0_instruction": "Choose the most plausible alternative",
            "chatgpt_instruction": "Select the option that best explains the given context and completes the sentence logically.",
            "chatgpt_instruction_v2": "Choose the sentence that logically follows from the given context.",
            "chatgpt_instruction_v3": "Identify the most logically consistent sentence from two given options based on the provided context, demonstrating reasoning and causal relationship skills.",
            "chatgpt_instruction_v4": "Generate the most likely outcome for a given scenario by choosing between two provided options based on contextual clues and causal reasoning.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 16,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "ignore_templates": [
                "…which may be caused by",
                "…What could happen next, C1 or C2?",
                "…As a result, C1 or C2?",
                "…why? C1 or C2",
            ],
            "include_templates": "original",
            "dataset_path": ["huggingface", "super_glue", "copa"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 16,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "ignore_templates": [
                "…which may be caused by",
                "…What could happen next, C1 or C2?",
                "…As a result, C1 or C2?",
                "…why? C1 or C2",
            ],
            "include_templates": "original",
            "dataset_path": ["huggingface", "super_glue", "copa"],
        },
    },
    "p3wic": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "super_glue", "wic"],
            "t0_instruction": "Determine whether the word has a similar meaning in sentences A and B.",
            "chatgpt_instruction": "Determine if the given word or phrase has the same meaning in the provided pair of sentences and respond with 'Yes' or 'No'.",
            "chatgpt_instruction_v2": "Determine if the specified word is used in the same sense in both sentences and respond with 'Yes' or 'No'.",
            "chatgpt_instruction_v3": "Determine whether a specific word or phrase is used in the same sense in two provided sentences. This task requires semantic analysis and understanding of word meanings in different contexts.O",
            "chatgpt_instruction_v4": "Generate a response to determine whether a word is used in the same sense in two given sentences. This task requires knowledge of word sense disambiguation and contextual meaning analysis.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 16,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "include_templates": "original",
            "dataset_path": ["huggingface", "super_glue", "wic"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 16,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "include_templates": "original",
            "dataset_path": ["huggingface", "super_glue", "wic"],
        },
    },
    "p3winogrande": {
        "train": {
            "split": "train",
            "batch_size": 32,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "winogrande", "winogrande_xl"],
            "t0_instruction": "Determine what does the _ in the sentence refer to.",
            "chatgpt_instruction": "Determine the correct pronoun antecedent by selecting the appropriate name to replace the underscore in the given sentence.",
            "chatgpt_instruction_v2": "Identify the correct referent or option for the placeholder (_) in a given sentence.",
            "chatgpt_instruction_v3": "Determine the correct pronoun or proper noun that completes a sentence based on context. Choose the appropriate option from the given choices to replace the blank. This task requires understanding of pronoun reference and contextual clues.",
            "chatgpt_instruction_v4": "Generate the correct antecedent or reference for the pronoun or placeholder in the given sentence, demonstrating your understanding of pronoun resolution and contextual analysis.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 32,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "include_templates": "original",
            "dataset_path": ["huggingface", "winogrande", "winogrande_xl"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 32,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "include_templates": "original",
            "dataset_path": ["huggingface", "winogrande", "winogrande_xl"],
        },
    },
    "p3cb": {
        "train": {
            "split": "train",
            "batch_size": 32,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "super_glue", "cb"],
            "t0_instruction": "Determine if the premise is 'true', 'false', or 'maybe' given the hypothesis.",
            "chatgpt_instruction": "Determine the truth value (True, False, Neither, Always, Sometimes, Never, Yes, No, Maybe) of a statement or inference based on the provided context.",
            "chatgpt_instruction_v2": "Determine if a given statement is 'always', 'sometimes', or 'never' correct, 'yes', 'no', or 'maybe', or 'guaranteed', 'possible', or 'impossible' based on the provided conversational context or text.",
            "chatgpt_instruction_v3": "Determine the truth value or inference of a given statement based on the provided text, utilizing comprehension and logical reasoning skills to choose the correct option among True/False/Neither, Always/Sometimes/Never, or Yes/No/Maybe.",
            "chatgpt_instruction_v4": "Determine the truth value (always, sometimes, never; yes, no, maybe; guaranteed, possible, impossible) of given statements based on conversational or narrative context, requiring understanding of implications and logical reasoning.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 32,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "include_templates": "original",
            "dataset_path": ["huggingface", "super_glue", "cb"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 32,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "include_templates": "original",
            "dataset_path": ["huggingface", "super_glue", "cb"],
        },
    },
    "p3storycloze": {
        "train": {
            "split": "train",
            "batch_size": 32,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "MoE-UNC/story_cloze"],
            "t0_instruction": "Determine which is a possible continuation for the story.",
            "chatgpt_instruction": "Generate the most plausible continuation for the given story from the provided options.",
            "chatgpt_instruction_v2": "Choose the most likely continuation for a given story from the provided options.",
            "chatgpt_instruction_v3": "Generate the most plausible continuation for a given story by selecting the appropriate option from multiple choices, demonstrating an understanding of narrative coherence and logical progression.",
            "chatgpt_instruction_v4": "Generate the most probable and logical ending for a given story from multiple provided options, using knowledge of narrative structure and coherence.",
        },
        "val": {
            "split": "validation-test",
            "batch_size": 32,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "include_templates": "original",
            "dataset_path": ["huggingface", "MoE-UNC/story_cloze"],
        },
        "test": {
            "split": "test-test",
            "batch_size": 32,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "include_templates": "original",
            "dataset_path": ["huggingface", "MoE-UNC/story_cloze"],
        },
    },
    "p3wscfixed": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "super_glue", "wsc.fixed"],
            "t0_instruction": "Determine whether the two pronouns refer to the same in the sentence.",
            "chatgpt_instruction": "Determine the referent of the pronoun in the given passage and respond with 'Yes' or 'No'.",
            "chatgpt_instruction_v2": "Determine whether the pronoun in the given sentence or passage refers to a specified antecedent, and answer 'Yes' or 'No'.",
            "chatgpt_instruction_v3": "Determine the referent of a pronoun in a given passage by analyzing the context and understanding pronoun antecedent relationships.",
            "chatgpt_instruction_v4": "Determine if a given pronoun in a passage refers to a specific antecedent, and answer ''Yes' or 'No'. This task requires knowledge of pronoun resolution and an understanding of context within sentences.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 16,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "include_templates": "original",
            "dataset_path": ["huggingface", "super_glue", "wsc.fixed"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 16,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "include_templates": "original",
            "dataset_path": ["huggingface", "super_glue", "wsc.fixed"],
        },
    },
    "p3mnli": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "glue", "mnli"],
            "t0_instruction": "Determine whether the hypothesis is 'true', 'false', or 'maybe' based on the premise.",
        },
        "val": {
            "split": "validation-validation_matched",
            "batch_size": 16,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "glue", "mnli"],
            "max_pretemplate_examples_per_dataset": 10_000,
        },
        "test": {
            "split": "test-validation",
            "batch_size": 16,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "glue", "mnli"],
            "max_pretemplate_examples_per_dataset": 10_000,
        },
    },
    "p3snli": {
        "train": {
            "split": "train",
            "batch_size": 32,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "snli"],
            "t0_instruction": "Determine whether the hypothesis is 'true', 'false', or 'maybe' based on the premise.",
        },
        "val": {
            "split": "validation-validation_matched",
            "batch_size": 32,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "snli"],
            "max_pretemplate_examples_per_dataset": 10_000,
        },
        "test": {
            "split": "test-validation_matched",
            "batch_size": 32,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "snli"],
            "max_pretemplate_examples_per_dataset": 10_000,
        },
    },
    "p3cola": {
        "train": {
            "split": "train",
            "batch_size": 32,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "glue", "cola"],
            "t0_instruction": "Determine whether the sentence make sense and use correct English",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 32,
            "max_length": 512,
            "metrics": ["matthews_correlation"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "glue", "cola"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 32,
            "max_length": 512,
            "metrics": ["matthews_correlation"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "glue", "cola"],
        },
    },
    "p3racehigh": {
        "train": {
            "split": "train",
            "batch_size": 2,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "race", "high"],
            "t0_instruction": "Select the right answer to the question given the article",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 2,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "race", "high"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 2,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "race", "high"],
        },
    },
    "p3racemiddle": {
        "train": {
            "split": "train",
            "batch_size": 2,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "race", "middle"],
            "t0_instruction": "Select the right answer to the question given the article.",
        },
        "val": {
            "split": "validation-validation",
            "batch_size": 2,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "race", "middle"],
        },
        "test": {
            "split": "test-validation",
            "batch_size": 2,
            "max_length": 512,
            "metrics": ["accuracy"],
            "interface": "mc",
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "race", "middle"],
        },
    },
    "p3webquestions": {
        "train": {
            "split": "train",
            "batch_size": 32,
            "max_length": 512,
            "include_templates": "all",
            "max_examples_per_dataset": 500_000,
            "dataset_path": ["huggingface", "web_questions"],
            "t0_instruction": "Generate the correct facts to answer the question.",
        },
        "val": {
            "split": "validation-test",
            "batch_size": 32,
            "max_length": 512,
            "metrics": ["rouge"],
            "interface": "gen",
            "max_gen_length": 32,
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "web_questions"],
        },
        "test": {
            "split": "test-test",
            "batch_size": 32,
            "max_length": 512,
            "metrics": ["rouge"],
            "interface": "gen",
            "max_gen_length": 32,
            "round_robin_template": True,
            "include_templates": "original",
            "dataset_path": ["huggingface", "web_questions"],
        },
    },
    "p3anlir1": {
        "train": {
            "split": "train-train_r1",
            "dataset_path": ["huggingface", "anli"],
            "batch_size": 32,
            "max_length": 512,
            "include_templates": "original",
            "interface": "mc",
            "metrics": ["accuracy"],
            "t0_instruction": "Determine whether the hypothesis is 'true', 'false', or 'maybe' based on the premise.",
            "chatgpt_instruction": "Determine the truth value (True, False, Neither, Always, Sometimes, Never, Yes, No, Maybe) of the given statements based on the provided context.",
            "chatgpt_instruction_v2": "Determine the validity of a statement based on provided textual information, choosing between 'always', 'sometimes', 'never', 'yes', 'no', 'maybe', 'guaranteed', 'possible', or 'impossible'.",
            "chatgpt_instruction_v3": "Generate a task instruction to evaluate a language model's ability to comprehend nuanced information and logical inference based on given statements. The model must determine whether a provided claim is true, false, sometimes true, never true, or if an inference can be made as yes, no, or maybe.",
            "chatgpt_instruction_v4": "Given a passage, determine whether a given statement is always, sometimes, never, yes, no, maybe, guaranteed, possible, or impossible based on the provided information. This task requires critical reading and comprehension skills to accurately interpret the relationship between the passage and the statement.",
        },
        "val": {
            "split": "validation-dev_r1",
            "dataset_path": ["huggingface", "anli"],
            "batch_size": 32,
            "max_length": 512,
            "include_templates": "original",
            "interface": "mc",
            "metrics": ["accuracy"],
        },
        "test": {
            "split": "test-dev_r1",
            "dataset_path": ["huggingface", "anli"],
            "batch_size": 32,
            "max_length": 512,
            "include_templates": "original",
            "interface": "mc",
            "metrics": ["accuracy"],
        },
    },
    "p3anlir2": {
        "train": {
            "split": "train-train_r2",
            "dataset_path": ["huggingface", "anli"],
            "batch_size": 32,
            "max_length": 512,
            "include_templates": "original",
            "interface": "mc",
            "metrics": ["accuracy"],
            "t0_instruction": "Determine whether the hypothesis is 'true', 'false', or 'maybe' based on the premise.",
            "chatgpt_instruction": "Determine the truth value (True, False, Neither, Always, Sometimes, Never, Yes, No, Maybe) of a given statement based on the provided information.",
            "chatgpt_instruction_v2": "Determine whether a given statement is always, sometimes, or never correct based on a provided text.",
            "chatgpt_instruction_v3": "Generate responses to questions by evaluating the truth value of statements using provided context, demonstrating skills in reading comprehension and logical inference.",
            "chatgpt_instruction_v4": "Given a text passage, analyze the provided information and determine the accuracy of a specific statement based on the text. Your response should classify the statement as 'always,' 'sometimes,' or 'never correct,' 'yes,' 'no,' or 'maybe,' or 'guaranteed,' 'possible,' or 'impossible,' depending on the context. This task requires comprehension and logical reasoning skills.",
        },
        "val": {
            "split": "validation-dev_r2",
            "dataset_path": ["huggingface", "anli"],
            "batch_size": 32,
            "max_length": 512,
            "include_templates": "original",
            "interface": "mc",
            "metrics": ["accuracy"],
        },
        "test": {
            "split": "test-dev_r2",
            "dataset_path": ["huggingface", "anli"],
            "batch_size": 32,
            "max_length": 512,
            "include_templates": "original",
            "interface": "mc",
            "metrics": ["accuracy"],
        },
    },
    "p3anlir3": {
        "train": {
            "split": "train-train_r3",
            "dataset_path": ["huggingface", "anli"],
            "batch_size": 32,
            "max_length": 512,
            "include_templates": "original",
            "interface": "mc",
            "metrics": ["accuracy"],
            "t0_instruction": "Determine whether the hypothesis is 'true', 'false', or 'maybe' based on the premise.",
            "chatgpt_instruction": "Determine the appropriate truth value (True, False, Neither, Always, Sometimes, Never, Yes, No, Maybe) for a statement based on the given information.",
            "chatgpt_instruction_v2": "Determine if a given statement is always, sometimes, or never correct based on the provided text.",
            "chatgpt_instruction_v3": "Generate a True/False/Neither, Always/Sometimes/Never, or Yes/No/Maybe response to a given statement based on the provided context, demonstrating comprehension and logical reasoning.",
            "chatgpt_instruction_v4": "Generate a task that evaluates the logical consistency and comprehension of given biographical and factual texts by determining the correctness of provided statements as always, sometimes, or never; yes, no, or maybe; or guaranteed, possible, or impossible.",
        },
        "val": {
            "split": "validation-dev_r3",
            "dataset_path": ["huggingface", "anli"],
            "batch_size": 32,
            "max_length": 512,
            "include_templates": "original",
            "interface": "mc",
            "metrics": ["accuracy"],
        },
        "test": {
            "split": "test-dev_r3",
            "dataset_path": ["huggingface", "anli"],
            "batch_size": 32,
            "max_length": 512,
            "include_templates": "original",
            "interface": "mc",
            "metrics": ["accuracy"],
        },
    },
}

BB_DATASET_CONFIGS = {
    "bbbooleanexpressions": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 32,
            "split": "test",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "lukaemon/bbh", "boolean_expressions"],
            "answer_choices": ["True", "False"]
        }
    },
    "bbcausaljudgement": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 16,
            "split": "train_validation",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "causal_judgment"]
        }
    },
    "bbdateunderstanding": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 32,
            "split": "train_validation",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "date_understanding"]
        }
    },
    "bbdisambiguationqa": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 32,
            "split": "train_validation",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "disambiguation_qa"]
        }
    },
    "bbdycklanguages": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 1,
            "split": "train_validation",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "dyck_languages"]
        }
    },
    "bbformalfallacies": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 32,
            "split": "train_validation",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "formal_fallacies_syllogisms_negation"]
        }
    },
    "bbgeometricshapes": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 8,
            "split": "train_validation",
            "metrics": ["accuracy_multiple_ans"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "geometric_shapes"]
        }
    },
    "bbhyperbaton": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 32,
            "split": "train_validation",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "hyperbaton"]
        }
    },
    "bblogicaldeduction": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 8,
            "split": "train_validation",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "logical_deduction"]
        }
    },
    "bbmovierecommendation": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 16,
            "split": "train_validation",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "movie_recommendation"]
        }
    },
    "bbmultisteparithmetictwo": {
        "test": {
            "interface": "gen",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 32,
            "split": "test",
            "metrics": ["exact_match"],
            "dataset_path": ["huggingface", "lukaemon/bbh", "multistep_arithmetic_two"],
            "max_gen_length": 8
        }
    },
    "bbnavigate": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 32,
            "split": "train_validation",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "navigate"]
        }
    },
    "bbobjectcounting": {
        "test": {
            "interface": "gen",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 32,
            "split": "train_validation",
            "metrics": ["exact_match_multiple_ans"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "object_counting"],
            "max_gen_length": 8
        }
    },
    "bbpenguinsinatable": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 16,
            "split": "train_validation",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "penguins_in_a_table"]
        }
    },
    "bbreasoningaboutcoloredobjects": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 4,
            "split": "train_validation",
            "metrics": ["accuracy_multiple_ans"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "reasoning_about_colored_objects"]
        }
    },
    "bbruinnames": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 16,
            "split": "train_validation",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "ruin_names"]
        }
    },
    "bbsalienttranslationerrordetection": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 4,
            "split": "train_validation",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "salient_translation_error_detection"]
        }
    },
    "bbsnarks": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 32,
            "split": "train_validation",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "snarks"]
        }
    },
    "bbsportsunderstanding": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 32,
            "split": "train_validation",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "sports_understanding"]
        }
    },
    "bbtemporalsequences": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 16,
            "split": "train_validation",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "temporal_sequences"]
        }
    },
    "bbtrackingshuffledobjects": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 8,
            "split": "train_validation",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "tracking_shuffled_objects"]
        }
    },
    "bbweboflies": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 32,
            "split": "test",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "lukaemon/bbh", "web_of_lies"],
            "answer_choices": ["Yes", "No"]
        }
    },
    "bbwordsorting": {
        "test": {
            "interface": "gen",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 32,
            "split": "train_validation",
            "metrics": ["exact_match"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "word_sorting"],
            "max_gen_length": 128
        }
    },
    "bbautodebugging": {
        "test": {
            "interface": "gen",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 32,
            "split": "train_validation",
            "metrics": ["exact_match_multiple_ans"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "auto_debugging"],
            "max_gen_length": 32
        }
    },
    "bbbbqlitejson": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 16,
            "split": "train_validation",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "bbq_lite_json"]
        }
    },
    "bbcodelinedescription": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 16,
            "split": "train_validation",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "code_line_description"]
        }
    },
    "bbconceptualcombinations": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 32,
            "split": "train_validation",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "conceptual_combinations"]
        }
    },
    "bbconlangtranslation": {
        "test": {
            "interface": "gen",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 32,
            "split": "train_validation",
            "metrics": ["rouge"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "conlang_translation"],
            "max_gen_length": 64
        }
    },
    "bbemojimovie": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 32,
            "split": "train_validation",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "emoji_movie"]
        }
    },
    "bbhinduknowledge": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 32,
            "split": "train_validation",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "hindu_knowledge"]
        }
    },
    "bbknownunknowns": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 32,
            "split": "train_validation",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "known_unknowns"]
        }
    },
    "bblanguageidentification": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 2,
            "split": "train_validation",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "language_identification"]
        }
    },
    "bblinguisticspuzzles": {
        "test": {
            "interface": "gen",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 32,
            "split": "train_validation",
            "metrics": ["exact_match"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "linguistics_puzzles"],
            "max_gen_length": 128
        }
    },
    "bblogicgridpuzzle": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 4,
            "split": "train_validation",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "logic_grid_puzzle"]
        }
    },
    "bbmisconceptionsrussian": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 32,
            "split": "train_validation",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "misconceptions_russian"]
        }
    },
    "bbnovelconcepts": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 16,
            "split": "train_validation",
            "metrics": ["accuracy_multiple_ans"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "novel_concepts"]
        }
    },
    "bboperators": {
        "test": {
            "interface": "gen",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 32,
            "split": "train_validation",
            "metrics": ["exact_match"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "operators"],
            "max_gen_length": 4
        }
    },
    "bbparsinlureadingcomprehension": {
        "test": {
            "interface": "gen",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 32,
            "split": "train_validation",
            "metrics": ["exact_match_multiple_ans"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "parsinlu_reading_comprehension"],
            "max_gen_length": 128
        }
    },
    "bbplaydialogsameordifferent": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 16,
            "split": "train_validation",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "play_dialog_same_or_different"]
        }
    },
    "bbrepeatcopylogic": {
        "test": {
            "interface": "gen",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 32,
            "split": "train_validation",
            "metrics": ["exact_match"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "repeat_copy_logic"],
            "max_gen_length": 64
        }
    },
    "bbstrangestories": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 16,
            "split": "train_validation",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "strange_stories"]
        }
    },
    "bbstrategyqa": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 32,
            "split": "train_validation",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "strategyqa"]
        }
    },
    "bbsymbolinterpretation": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 8,
            "split": "train_validation",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "symbol_interpretation"]
        }
    },
    "bbvitamincfactverification": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 8,
            "split": "train_validation",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "vitaminc_fact_verification"]
        }
    },
    "bbwinowhy": {
        "test": {
            "interface": "mc",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "max_length": 512,
            "batch_size": 32,
            "split": "train_validation",
            "metrics": ["accuracy"],
            "dataset_path": ["huggingface", "tasksource/bigbench", "winowhy"]
        }
    }
}

FLAN_DATASET_CONFIGS = {
    "flanv2ai2arceasy": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "ai2_arc",
                "ARC-Easy:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "interface": "lm",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "ai2_arc",
                "ARC-Easy:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "ai2_arc",
                "ARC-Easy:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        }
    },
    "flanv2ai2arcchallenge": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "ai2_arc",
                "ARC-Challenge:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "interface": "lm",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "ai2_arc",
                "ARC-Challenge:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "ai2_arc",
                "ARC-Challenge:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        }
    },
    "flanv2algebralinear1d": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "algebra__linear_1d:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "interface": "lm",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 4,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "algebra__linear_1d:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 4,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "algebra__linear_1d:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        }
    },
    "flanv2boolq": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "bool_q:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "interface": "lm",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "bool_q:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "bool_q:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        }
    },
    "flanv2coqa": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "coqa:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "interface": "lm",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 256,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "coqa:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 256,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "coqa:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        }
    },
    "flanv2defpronounresolution": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "definite_pronoun_resolution:1.1.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "interface": "lm",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "definite_pronoun_resolution:1.1.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "definite_pronoun_resolution:1.1.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        }
    },
    "flanv2drop": {
        "train": {
            "split": "train",
            "batch_size": 2,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "drop:2.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "interface": "lm",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "drop:2.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "drop:2.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        }
    },
    "flanv2fixpunct": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "fix_punct"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "interface": "lm",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 256,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "fix_punct"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 256,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "fix_punct"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        }
    },
    "flanv2gemdart": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "gem",
                "dart:1.1.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "interface": "lm",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "gem",
                "dart:1.1.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "gem",
                "dart:1.1.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        }
    },
    "flanv2geme2enlg": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "gem",
                "e2e_nlg:1.1.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "interface": "lm",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "gem",
                "e2e_nlg:1.1.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "gem",
                "e2e_nlg:1.1.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        }
    },
    "flanv2gemwebnlgen": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "gem",
                "web_nlg_en:1.1.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "interface": "lm",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "gem",
                "web_nlg_en:1.1.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "gem",
                "web_nlg_en:1.1.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        }
    },
    "flanv2gemwikilinguaen": {
        "train": {
            "split": "train",
            "batch_size": 2,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "gem",
                "wiki_lingua_english_en:1.1.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "interface": "lm",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "gem",
                "wiki_lingua_english_en:1.1.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "gem",
                "wiki_lingua_english_en:1.1.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        }
    },
    "flanv2gluesst2": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "glue",
                "sst2:2.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "interface": "lm",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "glue",
                "sst2:2.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "glue",
                "sst2:2.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        }
    },
    "flanv2gluecola": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "glue",
                "cola:2.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "interface": "lm",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "glue",
                "cola:2.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "glue",
                "cola:2.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        }
    },
    "flanv2gluemnli": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "glue",
                "mnli:2.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "interface": "lm",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "glue",
                "mnli:2.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "glue",
                "mnli:2.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        }
    },
    "flanv2glueqnli": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "glue",
                "qnli:2.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "glue",
                "qnli:2.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "glue",
                "qnli:2.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "flanv2gluestsb": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "glue",
                "stsb:2.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "glue",
                "stsb:2.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "glue",
                "stsb:2.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "flanv2gluewnli": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "glue",
                "wnli:2.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "glue",
                "wnli:2.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "glue",
                "wnli:2.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "flanv2lambada": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "lambada:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "lambada:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "lambada:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "flanv2naturalquestionsopen": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "natural_questions_open:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 32,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "natural_questions_open:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 32,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "natural_questions_open:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "flanv2newsroom": {
        "train": {
            "split": "train",
            "batch_size": 2,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "newsroom:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "newsroom:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "newsroom:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "flanv2openbookqa": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "openbookqa:0.1.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 32,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "openbookqa:0.1.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 32,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "openbookqa:0.1.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "flanv2opinionabstractsidebate": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "opinion_abstracts_idebate"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 256,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "opinion_abstracts_idebate"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 256,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "opinion_abstracts_idebate"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "flanv2opinionabstractrottentomatoes": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "opinion_abstracts_rotten_tomatoes"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "opinion_abstracts_rotten_tomatoes"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "opinion_abstracts_rotten_tomatoes"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "flanv2paracrawlenes": {
        "train": {
            "split": "train",
            "batch_size": 8,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "para_crawl_enes"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 256,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "para_crawl_enes"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 256,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "para_crawl_enes"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "flanv2piqa": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "piqa:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 256,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "piqa:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 256,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "piqa:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "flanv2quac": {
        "train": {
            "split": "train",
            "batch_size": 2,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "quac:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "quac:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "quac:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "flanv2sentiment140": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "sentiment140:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "sentiment140:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "sentiment140:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "flanv2snli": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "snli:1.1.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 32,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "snli:1.1.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 32,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "snli:1.1.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "flanv2squad": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "squad",
                "v2.0:3.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "squad",
                "v2.0:3.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "squad",
                "v2.0:3.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "flanv2supergluemultirc": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "super_glue",
                "multirc:1.0.2"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "super_glue",
                "multirc:1.0.2"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "super_glue",
                "multirc:1.0.2"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "flanv2supergluerecord": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "super_glue",
                "record:1.0.2"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "super_glue",
                "record:1.0.2"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "super_glue",
                "record:1.0.2"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "flanv2triviaqa": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "trivia_qa",
                "rc:1.1.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "trivia_qa",
                "rc:1.1.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "trivia_qa",
                "rc:1.1.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "flanv2truecase": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "true_case"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 256,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "true_case"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 256,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "true_case"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "flanv2unifiedqascienceinst": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "unified_qa_science_inst"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 32,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "unified_qa_science_inst"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 32,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "unified_qa_science_inst"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "flanv2wordsegment": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "word_segment"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 256,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "word_segment"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 256,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "word_segment"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "flanv2wmt14translatefren": {
        "train": {
            "split": "train",
            "batch_size": 2,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "wmt14_translate",
                "fr-en:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "wmt14_translate",
                "fr-en:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "wmt14_translate",
                "fr-en:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "flanv2wmt16translatecsen": {
        "train": {
            "split": "train",
            "batch_size": 2,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "wmt16_translate",
                "cs-en:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "wmt16_translate",
                "cs-en:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "wmt16_translate",
                "cs-en:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "flanv2wmt16translatedeen": {
        "train": {
            "split": "train",
            "batch_size": 2,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "wmt16_translate",
                "de-en:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "wmt16_translate",
                "de-en:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "wmt16_translate",
                "de-en:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "flanv2wmt16translateruen": {
        "train": {
            "split": "train",
            "batch_size": 2,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "wmt16_translate",
                "ru-en:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "wmt16_translate",
                "ru-en:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "wmt16_translate",
                "ru-en:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "flanv2wmt16translatefien": {
        "train": {
            "split": "train",
            "batch_size": 2,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "wmt16_translate",
                "fi-en:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "wmt16_translate",
                "fi-en:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "wmt16_translate",
                "fi-en:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "flanv2wmt16translateroen": {
        "train": {
            "split": "train",
            "batch_size": 2,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "wmt16_translate",
                "ro-en:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "wmt16_translate",
                "ro-en:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "wmt16_translate",
                "ro-en:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "flanv2wmt16translatetren": {
        "train": {
            "split": "train",
            "batch_size": 2,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "wmt16_translate",
                "tr-en:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "wmt16_translate",
                "tr-en:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "wmt16_translate",
                "tr-en:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "flanv2aeslc": {
        "train": {
            "split": "train",
            "batch_size": 2,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "aeslc:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "aeslc:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "Flan2021",
                "aeslc:1.0.0"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2translation": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Translation"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "interface": "lm",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Translation"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Translation"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        }
    },
    "niv2programexecution": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Program Execution"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 256,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Program Execution"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 256,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Program Execution"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2questiongeneration": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Question Generation"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Question Generation"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Question Generation"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2sentimentanalysis": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Sentiment Analysis"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Sentiment Analysis"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Sentiment Analysis"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2textcategorization": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Text Categorization"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 16,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Text Categorization"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 16,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Text Categorization"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2textmatching": {
        "train": {
            "split": "train",
            "batch_size": 8,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Text Matching"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Text Matching"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Text Matching"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2toxiclanguagedetection": {
        "train": {
            "split": "train",
            "batch_size": 8,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Toxic Language Detection"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Toxic Language Detection"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Toxic Language Detection"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2causeeffectclassification": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Cause Effect Classification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Cause Effect Classification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Cause Effect Classification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2informationextraction": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Information Extraction"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 256,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Information Extraction"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 256,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Information Extraction"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2textualentailment": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Textual Entailment"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Textual Entailment"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Textual Entailment"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2wrongcandidategeneration": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Wrong Candidate Generation"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 256,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Wrong Candidate Generation"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 256,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Wrong Candidate Generation"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2namedentityrecognition": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Named Entity Recognition"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 256,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Named Entity Recognition"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 256,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Named Entity Recognition"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2commonsenseclassification": {
        "train": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Commonsense Classification"
            ],
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Commonsense Classification"
            ],
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 4,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Commonsense Classification"
            ],
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 4,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2fillintheblank": {
        "train": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Fill in The Blank"
            ],
            "split": "train",
            "batch_size": 8,
            "max_examples_per_dataset": 500000,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Fill in The Blank"
            ],
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Fill in The Blank"
            ],
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2textcompletion": {
        "train": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Text Completion"
            ],
            "split": "train",
            "batch_size": 8,
            "max_examples_per_dataset": 500000,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Text Completion"
            ],
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Text Completion"
            ],
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2sentencecomposition": {
        "train": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Sentence Composition"
            ],
            "split": "train",
            "batch_size": 4,
            "max_examples_per_dataset": 500000,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Sentence Composition"
            ],
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Sentence Composition"
            ],
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2titlegeneration": {
        "train": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Title Generation"
            ],
            "split": "train",
            "batch_size": 8,
            "max_examples_per_dataset": 500000,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Title Generation"
            ],
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Title Generation"
            ],
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2languageidentification": {
        "train": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Language Identification"
            ],
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Language Identification"
            ],
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Language Identification"
            ],
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2questionunderstanding": {
        "train": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Question Understanding"
            ],
            "split": "train",
            "batch_size": 2,
            "max_examples_per_dataset": 500000,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Question Understanding"
            ],
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 32,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Question Understanding"
            ],
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 32,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2sentenceperturbation": {
        "train": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Sentence Perturbation"
            ],
            "split": "train",
            "batch_size": 4,
            "max_examples_per_dataset": 500000,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Sentence Perturbation"
            ],
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 256,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Sentence Perturbation"
            ],
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 256,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2answerabilityclassification": {
        "train": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Answerability Classification"
            ],
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Answerability Classification"
            ],
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 4,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Answerability Classification"
            ],
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 4,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2summarization": {
        "train": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Summarization"
            ],
            "split": "train",
            "batch_size": 2,
            "max_examples_per_dataset": 500000,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Summarization"
            ],
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Summarization"
            ],
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2coreferenceresolution": {
        "train": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Coreference Resolution"
            ],
            "split": "train",
            "batch_size": 2,
            "max_examples_per_dataset": 500000,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Coreference Resolution"
            ],
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 16,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Coreference Resolution"
            ],
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 16,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2textqualityevaluation": {
        "train": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Text Quality Evaluation"
            ],
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Text Quality Evaluation"
            ],
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 32,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Text Quality Evaluation"
            ],
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 32,
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2texttocode": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Text to Code"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Text to Code"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Text to Code"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2paraphrasing": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Paraphrasing"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Paraphrasing"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Paraphrasing"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2dialoguegeneration": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Dialogue Generation"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Dialogue Generation"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Dialogue Generation"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2questionrewriting": {
        "train": {
            "split": "train",
            "batch_size": 8,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Question Rewriting"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Question Rewriting"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Question Rewriting"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2wordsemantics": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Word Semantics"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 16,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Word Semantics"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 16,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Word Semantics"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2postagging": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Pos Tagging"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 256,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Pos Tagging"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 256,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Pos Tagging"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2linguisticprobing": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Linguistic Probing"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 4,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Linguistic Probing"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 4,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Linguistic Probing"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2storycomposition": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Story Composition"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Story Composition"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Story Composition"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2speakeridentification": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Speaker Identification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 16,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Speaker Identification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 16,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Speaker Identification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2wordanalogy": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Word Analogy"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Word Analogy"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Word Analogy"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2datatotext": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Data to Text"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Data to Text"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Data to Text"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2stereotypedetection": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Stereotype Detection"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Stereotype Detection"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Stereotype Detection"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2negotiationstrategydetection": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Negotiation Strategy Detection"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 2,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Negotiation Strategy Detection"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 2,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Negotiation Strategy Detection"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2dialogueactrecognition": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Dialogue Act Recognition"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Dialogue Act Recognition"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Dialogue Act Recognition"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2genderclassification": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Gender Classification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Gender Classification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Gender Classification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2coherenceclassification": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Coherence Classification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 4,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Coherence Classification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 4,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Coherence Classification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2explanation": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Explanation"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Explanation"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Explanation"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2ethicsclassification": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Ethics Classification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Ethics Classification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Ethics Classification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2wordrelationclassification": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Word Relation Classification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Word Relation Classification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Word Relation Classification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2sentenceordering": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Sentence Ordering"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Sentence Ordering"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Sentence Ordering"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2answerverification": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Answer Verification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 4,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Answer Verification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 4,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Answer Verification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2mathematics": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Mathematics"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Mathematics"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Mathematics"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2intentidentification": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Intent Identification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 32,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Intent Identification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 32,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Intent Identification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2keywordtagging": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Keyword Tagging"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 16,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Keyword Tagging"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 16,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Keyword Tagging"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2codetotext": {
        "train": {
            "split": "train",
            "batch_size": 8,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Code to Text"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Code to Text"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Code to Text"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2dialoguestatetracking": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Dialogue State Tracking"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Dialogue State Tracking"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Dialogue State Tracking"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2textsimplification": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Text Simplification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Text Simplification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Text Simplification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2stancedetection": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Stance Detection"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 4,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Stance Detection"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 4,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Stance Detection"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2factverification": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Fact Verification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 4,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Fact Verification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 4,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Fact Verification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2grammarerrordetection": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Grammar Error Detection"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Grammar Error Detection"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Grammar Error Detection"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2sectionclassification": {
        "train": {
            "split": "train",
            "batch_size": 8,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Section Classification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Section Classification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Section Classification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2numberconversion": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Number Conversion"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Number Conversion"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Number Conversion"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2styletransfer": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Style Transfer"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Style Transfer"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Style Transfer"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2speakerrelationclassification": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Speaker Relation Classification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 4,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Speaker Relation Classification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 4,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Speaker Relation Classification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2ironydetection": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Irony Detection"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 4,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Irony Detection"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 4,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Irony Detection"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2questiondecomposition": {
        "train": {
            "split": "train",
            "batch_size": 8,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Question Decomposition"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 256,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Question Decomposition"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 256,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Question Decomposition"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2overlapextraction": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Overlap Extraction"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Overlap Extraction"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 64,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Overlap Extraction"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2grammarerrorcorrection": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Grammar Error Correction"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Grammar Error Correction"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Grammar Error Correction"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2spellingerrordetection": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Spelling Error Detection"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 16,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Spelling Error Detection"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 16,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Spelling Error Detection"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2entitygeneration": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Entity Generation"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 16,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Entity Generation"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 16,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Entity Generation"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2sentenceexpansion": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Sentence Expansion"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Sentence Expansion"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Sentence Expansion"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2discourseconnectiveidentification": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Discourse Connective Identification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Discourse Connective Identification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 8,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Discourse Connective Identification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2discourserelationclassification": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Discourse Relation Classification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 16,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Discourse Relation Classification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 16,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Discourse Relation Classification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2poemgeneration": {
        "train": {
            "split": "train",
            "batch_size": 8,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Poem Generation"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Poem Generation"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Poem Generation"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2entityrelationclassification": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Entity Relation Classification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 2,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Entity Relation Classification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 2,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Entity Relation Classification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2punctuationerrordetection": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Punctuation Error Detection"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 4,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Punctuation Error Detection"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 4,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Punctuation Error Detection"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2spamclassification": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Spam Classification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 4,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Spam Classification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 4,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Spam Classification"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2paperreview": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Paper Review"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 2,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Paper Review"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 2,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Paper Review"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2sentencecompression": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Sentence Compression"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Sentence Compression"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Sentence Compression"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2prepositionprediction": {
        "train": {
            "split": "train",
            "batch_size": 16,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Preposition Prediction"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 2,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Preposition Prediction"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 2,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Preposition Prediction"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "niv2misc": {
        "train": {
            "split": "train",
            "batch_size": 8,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Misc."
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 256,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Misc."
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 256,
            "dataset_path": [
                "FLANV2",
                "NIV2",
                "Misc."
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "flanv2qrecc": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Dialog",
                "qrecc"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "interface": "lm",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "dataset_path": [
                "FLANV2",
                "Dialog",
                "qrecc"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 128,
            "dataset_path": [
                "FLANV2",
                "Dialog",
                "qrecc"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        }
    },
    "flanv2qreccii": {
        "train": {
            "split": "train",
            "batch_size": 4,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Dialog",
                "qrecc_ii"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets"
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "Dialog",
                "qrecc_ii"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "Dialog",
                "qrecc_ii"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "batch_size": 32
        }
    },
    "flanv2wikidialog": {
        "train": {
            "split": "train",
            "batch_size": 2,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Dialog",
                "wiki_dialog"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "interface": "lm",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "Dialog",
                "wiki_dialog"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "Dialog",
                "wiki_dialog"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        }
    },
    "flanv2wikidialogii": {
        "train": {
            "split": "train",
            "batch_size": 2,
            "max_examples_per_dataset": 500000,
            "dataset_path": [
                "FLANV2",
                "Dialog",
                "wiki_dialog_ii"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "interface": "lm",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1
        },
        "val": {
            "split": "validation-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "Dialog",
                "wiki_dialog_ii"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        },
        "test": {
            "split": "test-validation",
            "metrics": [
                "rouge"
            ],
            "interface": "gen",
            "max_gen_length": 512,
            "dataset_path": [
                "FLANV2",
                "Dialog",
                "wiki_dialog_ii"
            ],
            "max_length": 512,
            "input_field": "inputs",
            "target_field": "targets",
            "length_normalization": True,
            "multiple_choice_loss": 1,
            "unlikelihood_loss": 1,
            "num_beams": 1,
            "batch_size": 32
        }
    }
}

FULL_DATASET_CONFIGS = {
    **FLAN_DATASET_CONFIGS,
    **BB_DATASET_CONFIGS,
    **P3_DATASET_CONFIGS,
    **FLAT_DATASET_CONFIGS,
}

MAX_VAL_SAMPLES = 500
MAX_TEST_SAMPLES = 2000
for task, config in P3_DATASET_CONFIGS.items():
    P3_DATASET_CONFIGS[task]["val"]["max_pretemplate_examples_per_dataset"] = MAX_VAL_SAMPLES
    P3_DATASET_CONFIGS[task]["val"]["t0_instruction"] = P3_DATASET_CONFIGS[task]["train"]["t0_instruction"]
    P3_DATASET_CONFIGS[task]["test"]["max_pretemplate_examples_per_dataset"] = MAX_TEST_SAMPLES
    P3_DATASET_CONFIGS[task]["test"]["t0_instruction"] = P3_DATASET_CONFIGS[task]["train"]["t0_instruction"]

    if "chatgpt_instruction" in P3_DATASET_CONFIGS[task]["train"]:
        P3_DATASET_CONFIGS[task]["val"]["chatgpt_instruction"] = P3_DATASET_CONFIGS[task]["train"][
            "chatgpt_instruction"]
        P3_DATASET_CONFIGS[task]["test"]["chatgpt_instruction"] = P3_DATASET_CONFIGS[task]["train"][
            "chatgpt_instruction"]

    if "chatgpt_instruction_v2" in P3_DATASET_CONFIGS[task]["train"]:
        P3_DATASET_CONFIGS[task]["val"]["chatgpt_instruction_v2"] = P3_DATASET_CONFIGS[task]["train"][
            "chatgpt_instruction_v2"]
        P3_DATASET_CONFIGS[task]["test"]["chatgpt_instruction_v2"] = P3_DATASET_CONFIGS[task]["train"][
            "chatgpt_instruction_v2"]

    if "chatgpt_instruction_v3" in P3_DATASET_CONFIGS[task]["train"]:
        P3_DATASET_CONFIGS[task]["val"]["chatgpt_instruction_v3"] = P3_DATASET_CONFIGS[task]["train"][
            "chatgpt_instruction_v3"]
        P3_DATASET_CONFIGS[task]["test"]["chatgpt_instruction_v3"] = P3_DATASET_CONFIGS[task]["train"][
            "chatgpt_instruction_v3"]

    if "chatgpt_instruction_v4" in P3_DATASET_CONFIGS[task]["train"]:
        P3_DATASET_CONFIGS[task]["val"]["chatgpt_instruction_v4"] = P3_DATASET_CONFIGS[task]["train"][
            "chatgpt_instruction_v4"]
        P3_DATASET_CONFIGS[task]["test"]["chatgpt_instruction_v4"] = P3_DATASET_CONFIGS[task]["train"][
            "chatgpt_instruction_v4"]

    # P3_DATASET_CONFIGS[task]["val"]["batch_size"] *= 2
    # P3_DATASET_CONFIGS[task]["test"]["batch_size"] *= 2

P3_DATASET_CONFIGS_MOE = deepcopy(P3_DATASET_CONFIGS)
BB_DATASET_CONFIGS_MOE = deepcopy(BB_DATASET_CONFIGS)
FULL_DATASET_CONFIGS_MOE = deepcopy(FULL_DATASET_CONFIGS)
_molora_train_batch_size = {
    "p3cosmosqa": 8,
    "p3quoref": 2,
    "p3wikiqa": 16,
    "p3imdb": 2,
    "p3agnews": 1,
    "p3dbpedia14": 2,
    "p3commongen": 32,
    "p3cnndailymail": 1,
    "p3paws": 16,
    "p3gigaword": 16,
    "p3agnews": 8,
    "p3rottentomatoes": 32,
    "p3adversarialqa": 4,
}
for _t, _bsz in _molora_train_batch_size.items():
    P3_DATASET_CONFIGS_MOE[_t]["train"]["batch_size"] = _bsz

FLATTASK2CLASS = {
    "c4": C4Dataset,
}

P3TASK2CLASS = {
    "p3socialiqa": P3Dataset,
    "p3paws": P3Dataset,
    "p3wikiqa": P3Dataset,
    "p3ropes": P3RopesDataset,
    "p3agnews": P3Dataset,
    "p3amazonpolarity": P3Dataset,
    "p3wikibio": P3WikiBioDataset,
    "p3cnndailymail": P3CNNDailyMailDataset,
    "p3cosmosqa": P3Dataset,
    "p3quail": P3Dataset,
    "p3quartz": P3Dataset,
    "p3qasc": P3Dataset,
    "p3commongen": P3CommonGenDataset,
    "p3adversarialqa": P3AdversarialQADataset,
    "p3appreviews": P3AppReviewsDataset,
    "p3commonsenseqa": P3Dataset,
    "p3cose": P3Dataset,
    "p3dbpedia14": P3Dataset,
    "p3dream": P3Dataset,
    "p3duorc": P3DuorcDataset,
    "p3gigaword": P3GigaWordDataset,
    "p3hotpotqa": P3HotpotQADataset,
    "p3imdb": P3Dataset,
    "p3mrpc": P3Dataset,
    "p3multinews": P3MultinewsDataset,
    "p3qqp": P3Dataset,
    "p3quarel": P3Dataset,
    "p3quoref": P3QuorefDataset,
    "p3rottentomatoes": P3Dataset,
    "p3samsum": P3SamsumDataset,
    "p3sciq": P3Dataset,
    "p3trec": P3TrecDataset,
    "p3wikihop": P3WikiHopDataset,
    "p3wiqa": P3Dataset,
    "p3xsum": P3XSumDataset,
    "p3yelp": P3Dataset,
    "p3rte": P3Dataset,
    "p3hswag": P3Dataset,
    "p3copa": P3Dataset,
    "p3wic": P3Dataset,
    "p3winogrande": P3Dataset,
    "p3cb": P3Dataset,
    "p3storycloze": P3StoryClozeDataset,
    "p3wscfixed": P3Dataset,
    "p3mnli": P3MnliDataset,
    "p3snli": P3Dataset,
    "p3cola": P3Dataset,
    "p3racehigh": P3Dataset,
    "p3racemiddle": P3Dataset,
    "p3webquestions": P3WebQuestionsDataset,
    "p3anlir1": P3Dataset,
    "p3anlir2": P3Dataset,
    "p3anlir3": P3Dataset,
}

BigBenchTASK2CLASS = {
    "bbbooleanexpressions": BigBenchSampleDataset,
    "bbcausaljudgement": BigBenchDataset,
    "bbdateunderstanding": BigBenchDataset,
    "bbdisambiguationqa": BigBenchDataset,
    "bbdycklanguages": BigBenchDataset,
    "bbformalfallacies": BigBenchDataset,
    "bbgeometricshapes": BigBenchDataset,
    "bbhyperbaton": BigBenchDataset,
    "bblogicaldeduction": BigBenchDataset,
    "bbmovierecommendation": BigBenchDataset,
    "bbmultisteparithmetictwo": BigBenchSampleDataset,
    "bbnavigate": BigBenchDataset,
    "bbobjectcounting": BigBenchDataset,
    "bbpenguinsinatable": BigBenchDataset,
    "bbreasoningaboutcoloredobjects": BigBenchDataset,
    "bbruinnames": BigBenchDataset,
    "bbsalienttranslationerrordetection": BigBenchDataset,
    "bbsnarks": BigBenchDataset,
    "bbsportsunderstanding": BigBenchDataset,
    "bbtemporalsequences": BigBenchDataset,
    "bbtrackingshuffledobjects": BigBenchDataset,
    "bbweboflies": BigBenchSampleDataset,
    "bbwordsorting": BigBenchDataset,
    "bbautodebugging": BigBenchDataset,
    "bbbbqlitejson": BigBenchDataset,
    "bbcodelinedescription": BigBenchDataset,
    "bbconceptualcombinations": BigBenchDataset,
    "bbconlangtranslation": BigBenchDataset,
    "bbemojimovie": BigBenchDataset,
    "bbhinduknowledge": BigBenchDataset,
    "bbknownunknowns": BigBenchDataset,
    "bblanguageidentification": BigBenchDataset,
    "bblinguisticspuzzles": BigBenchDataset,
    "bblogicgridpuzzle": BigBenchDataset,
    "bbmisconceptionsrussian": BigBenchDataset,
    "bbnovelconcepts": BigBenchDataset,
    "bboperators": BigBenchDataset,
    "bbparsinlureadingcomprehension": BigBenchDataset,
    "bbplaydialogsameordifferent": BigBenchDataset,
    "bbrepeatcopylogic": BigBenchDataset,
    "bbstrangestories": BigBenchDataset,
    "bbstrategyqa": BigBenchDataset,
    "bbsymbolinterpretation": BigBenchDataset,
    "bbvitamincfactverification": BigBenchDataset,
    "bbwinowhy": BigBenchDataset
}

FullTASK2CLASS = {
    **FLATTASK2CLASS,
    **P3TASK2CLASS,
    **BigBenchTASK2CLASS,
}

OTHER_TASKS = [
    "p3cose",
    "p3mnli",
    "p3snli",
    "p3cola",
    "p3racehigh",
    "p3racemiddle",
    "p3webquestions",
]

T0_MCQ_TASKS = [
    "p3socialiqa",
    "p3wiqa",
    "p3cosmosqa",
    "p3quail",
    "p3quartz",
    "p3qasc",
    "p3commonsenseqa",
    "p3quarel",
    "p3dream",
    "p3sciq",
    "p3wikihop",
]
T0_EQA_TASKS = [
    "p3ropes",
    "p3adversarialqa",
    "p3duorc",
    "p3quoref",
]
T0_CBQA_TASKS = [
    "p3hotpotqa",
    "p3wikiqa",
]
T0_SENTIMENT_TASKS = [
    "p3amazonpolarity",
    "p3appreviews",
    "p3rottentomatoes",
    "p3imdb",
    "p3yelp",
]
T0_TC_TASKS = [
    "p3agnews",
    "p3dbpedia14",
    "p3trec",
]
T0_STT_TASKS = [
    "p3wikibio",
    "p3commongen",
]
T0_SUMM_TASKS = [
    "p3cnndailymail",
    "p3multinews",
    "p3gigaword",
    "p3samsum",
    "p3xsum",
]
T0_PI_TASKS = [
    "p3paws",
    "p3qqp",
    "p3mrpc",
]

T0_SC_TASKS = [
    "p3hswag",
    "p3copa",
    "p3storycloze",
]
T0_NLI_TASKS = ["p3cb", "p3rte", "p3anlir1", "p3anlir2", "p3anlir3"]
T0_CR_TASKS = [
    "p3winogrande",
    "p3wscfixed",
]
T0_WSE_TASKS = [
    "p3wic",
]
BIGBENCH_HARD_TASKS = [
    'bbbooleanexpressions',
    'bbcausaljudgement',
    'bbdateunderstanding',
    'bbdisambiguationqa',
    # 'bbdycklanguages',
    'bbformalfallacies',
    'bbgeometricshapes',
    'bbhyperbaton',
    'bblogicaldeduction',
    'bbmovierecommendation',
    'bbmultisteparithmetictwo',
    'bbnavigate',
    'bbobjectcounting',
    'bbpenguinsinatable',
    'bbreasoningaboutcoloredobjects',
    'bbruinnames',
    'bbsalienttranslationerrordetection',
    'bbsnarks',
    'bbsportsunderstanding',
    'bbtemporalsequences',
    'bbtrackingshuffledobjects',
    'bbweboflies',
    'bbwordsorting',
]

BIGBENCH_LITE_TASKS = [
    # 'bbautodebugging',
    'bbbbqlitejson',
    # 'bbcodelinedescription',
    'bbconceptualcombinations',
    'bbconlangtranslation',
    # 'bbemojimovie',
    'bbformalfallacies',
    'bbhinduknowledge',
    'bbknownunknowns',
    # 'bblanguageidentification',
    'bblinguisticspuzzles',
    'bblogicgridpuzzle',
    'bblogicaldeduction',
    # 'bbmisconceptionsrussian',
    'bbnovelconcepts',
    'bboperators',
    # 'bbparsinlureadingcomprehension',
    'bbplaydialogsameordifferent',
    'bbrepeatcopylogic',
    'bbstrangestories',
    'bbstrategyqa',
    # 'bbsymbolinterpretation',
    'bbvitamincfactverification',
    'bbwinowhy'
]

T0_HELDIN_TASKS = (
        T0_MCQ_TASKS
        + T0_EQA_TASKS
        + T0_CBQA_TASKS
        + T0_SENTIMENT_TASKS
        + T0_TC_TASKS
        + T0_STT_TASKS
        + T0_SUMM_TASKS
        + T0_PI_TASKS
)

BIGBENCH_TASKS = BIGBENCH_HARD_TASKS + BIGBENCH_LITE_TASKS

T0_HELDOUT_TASKS = (
        T0_SC_TASKS + T0_NLI_TASKS + T0_CR_TASKS + T0_WSE_TASKS
)
T0_TASKS = T0_HELDIN_TASKS + T0_HELDOUT_TASKS

T0_CL_TRAIN_1 = [
    "p3commongen",
    "p3wikiqa",
    "p3paws",
    "p3cosmosqa",
    "p3quoref",
    "p3imdb",
    "p3dbpedia14",
    "p3agnews",
    "p3cnndailymail",
]
T0_CL_TRAIN_2 = [
    "p3gigaword",
    "p3paws",
    "p3commongen",
    "p3agnews",
    "p3rottentomatoes",
    "p3wikiqa",
    "p3adversarialqa",
    "p3cosmosqa",
]
T0_CL_TRAIN_3 = [
    "p3commongen",
    "p3wikiqa",
    "p3paws",
    "p3cosmosqa",
    "p3quoref",
    "p3imdb",
    "p3agnews",
    "p3cnndailymail",
]

T0_INIT_POOL_1 = list(set(T0_HELDIN_TASKS) - set(T0_CL_TRAIN_1))
T0_INIT_POOL_2 = list(set(T0_HELDIN_TASKS) - set(T0_CL_TRAIN_2))
T0_INIT_POOL_3 = list(set(T0_HELDIN_TASKS) - set(T0_CL_TRAIN_3))

P3_TASKS = ['p3agnews', 'p3amazonpolarity', 'p3cosmosqa', 'p3samsum', 'p3quartz', 'p3ropes', 'p3wikibio', 'p3paws',
            'p3wikiqa', 'p3socialiqa', 'p3qasc', 'p3quail', 'p3dream', 'p3wiqa', 'p3quarel', 'p3sciq', 'p3quoref',
            'p3duorc', 'p3rottentomatoes', 'p3yelp', 'p3commongen', 'p3gigaword', 'p3xsum', 'p3mrpc', 'p3qqp',
            'p3commonsenseqa', 'p3cose', 'p3wikihop', 'p3hotpotqa', 'p3appreviews', 'p3trec', 'p3multinews', 'p3imdb',
            'p3adversarialqa', 'p3cnndailymail', 'p3dbpedia14']

FLANV2_TASKS = ['flanv2ai2arceasy', 'flanv2ai2arcchallenge', 'flanv2algebralinear1d', 'flanv2boolq',
                'flanv2coqa', 'flanv2defpronounresolution', 'flanv2drop', 'flanv2fixpunct', 'flanv2gemdart',
                'flanv2geme2enlg', 'flanv2gemwebnlgen', 'flanv2gemwikilinguaen', 'flanv2gluesst2', 'flanv2gluecola',
                'flanv2gluemnli', 'flanv2glueqnli', 'flanv2gluestsb', 'flanv2gluewnli', 'flanv2lambada',
                'flanv2naturalquestionsopen', 'flanv2newsroom', 'flanv2openbookqa', 'flanv2opinionabstractsidebate',
                'flanv2opinionabstractrottentomatoes', 'flanv2paracrawlenes', 'flanv2piqa', 'flanv2quac',
                'flanv2sentiment140', 'flanv2snli', 'flanv2squad', 'flanv2supergluemultirc', 'flanv2supergluerecord',
                'flanv2triviaqa', 'flanv2truecase', 'flanv2unifiedqascienceinst', 'flanv2wordsegment']

NIV2_TASKS = ['niv2translation', 'niv2programexecution', 'niv2questiongeneration', 'niv2sentimentanalysis',
              'niv2textcategorization', 'niv2textmatching', 'niv2toxiclanguagedetection',
              'niv2causeeffectclassification', 'niv2informationextraction', 'niv2textualentailment',
              'niv2wrongcandidategeneration', 'niv2namedentityrecognition', 'niv2commonsenseclassification',
              'niv2fillintheblank', 'niv2textcompletion', 'niv2sentencecomposition', 'niv2titlegeneration',
              'niv2languageidentification', 'niv2questionunderstanding', 'niv2sentenceperturbation',
              'niv2answerabilityclassification', 'niv2summarization', 'niv2coreferenceresolution',
              'niv2textqualityevaluation', 'niv2texttocode', 'niv2paraphrasing', 'niv2dialoguegeneration',
              'niv2questionrewriting', 'niv2wordsemantics', 'niv2postagging', 'niv2linguisticprobing',
              'niv2storycomposition', 'niv2speakeridentification', 'niv2wordanalogy', 'niv2datatotext',
              'niv2stereotypedetection', 'niv2negotiationstrategydetection', 'niv2dialogueactrecognition',
              'niv2genderclassification', 'niv2coherenceclassification', 'niv2explanation',
              'niv2ethicsclassification', 'niv2wordrelationclassification', 'niv2sentenceordering',
              'niv2answerverification', 'niv2mathematics', 'niv2intentidentification', 'niv2keywordtagging',
              'niv2codetotext', 'niv2dialoguestatetracking', 'niv2textsimplification', 'niv2stancedetection',
              'niv2factverification', 'niv2grammarerrordetection', 'niv2sectionclassification',
              'niv2numberconversion', 'niv2styletransfer', 'niv2speakerrelationclassification',
              'niv2ironydetection', 'niv2questiondecomposition', 'niv2overlapextraction',
              'niv2grammarerrorcorrection', 'niv2spellingerrordetection', 'niv2entitygeneration',
              'niv2sentenceexpansion', 'niv2discourseconnectiveidentification',
              'niv2discourserelationclassification', 'niv2poemgeneration', 'niv2entityrelationclassification',
              'niv2punctuationerrordetection', 'niv2spamclassification', 'niv2paperreview',
              'niv2sentencecompression', 'niv2prepositionprediction', 'niv2misc']

ADD1_TASKS = ['p3wscfixed', 'p3copa', 'p3hswag', 'p3wic', 'p3racehigh', 'p3racemiddle', 'p3webquestions',
              'flanv2qrecc', 'flanv2wikidialog', 'flanv2qreccii', 'flanv2wikidialogii',
              'flanv2aeslc', 'flanv2wmt16translatecsen', 'flanv2wmt16translatedeen',
              'flanv2wmt16translateruen', 'flanv2wmt16translatefien', 'flanv2wmt16translateroen',
              'flanv2wmt16translatetren', 'flanv2wmt14translatefren']

FULL_TASKS = P3_TASKS + FLANV2_TASKS + NIV2_TASKS + ADD1_TASKS

FullTASK2CLASS = {
    **FullTASK2CLASS,
    **{task: FlanDataset for task in FULL_TASKS if task not in FullTASK2CLASS}
}

TAG2TASK_LIST = {
    "t0-heldin": T0_HELDIN_TASKS,
    "t0-heldout": T0_HELDOUT_TASKS,
    "t0": T0_TASKS,
    "t0-mcq": T0_MCQ_TASKS,
    "t0-eqa": T0_EQA_TASKS,
    "t0-cbqa": T0_CBQA_TASKS,
    "t0-sentiment": T0_SENTIMENT_TASKS,
    "t0-tc": T0_TC_TASKS,
    "t0-stt": T0_STT_TASKS,
    "t0-summ": T0_SUMM_TASKS,
    "t0-pi": T0_PI_TASKS,
    "t0-sc": T0_SC_TASKS,
    "t0-nli": T0_NLI_TASKS,
    "t0-cr": T0_CR_TASKS,
    "t0-wse": T0_WSE_TASKS,
    "bigbench": BIGBENCH_TASKS,
    "bigbench-hard": BIGBENCH_HARD_TASKS,
    "bigbench-lite": BIGBENCH_LITE_TASKS,
    "t0-cl-train1": T0_CL_TRAIN_1,
    "t0-cl-init1": T0_INIT_POOL_1,
    "t0-cl-train2": T0_CL_TRAIN_2,
    "t0-cl-init2": T0_INIT_POOL_2,
    "t0-cl-train3": T0_CL_TRAIN_3,
    "t0-cl-init3": T0_INIT_POOL_3,
    "p3": P3_TASKS,
    "flanv2": FLANV2_TASKS,
    "niv2": NIV2_TASKS,
    "add1": ADD1_TASKS,
    "full": FULL_TASKS,
}
