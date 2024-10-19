# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/3/5

CHECKPOINTING_CACHE_NAME = "_checkpointing_cache.pt"
FINAL_CHECKPOINT_NAME = "final_checkpoint.pt"
METRICS_FILE_NAME = "metrics.json"

MODEL_TO_ZEROSHOT_SCORE = {
    "google/t5-xl-lm-adapt": {
        "p3wiqa": 0.333,
        "p3cosmosqa": 0.287,
        "p3qasc": 0.231,
        "p3sciq": 0.522,
        "p3wikihop": 0.225,
        "p3ropes": 0.0365,
        "p3duorc": 0.0135,
        "p3wikiqa": 0.61,
        "p3imdb": 0.617,
        "p3rottentomatoe": 0.558,
    }
}

MODEL_TO_UPPERBOUND_SCORE = {
    "google/t5-xl-lm-adapt": {
        "p3dream": 0.755,
        "p3imdb": 0.564,
        "p3socialiqa": 0.515,
        "p3wikiqa": 0.945,
        "p3mrpc": 0.778,
        "p3commongen": 0.22349999999999998,
        "p3xsum": 0.219,
        "p3amazonpolarity": 0.972,
        "p3wikihop": 0.268,
        "p3wikibio": 0.237,
        "p3quartz": 0.566,
        "p3qqp": 0.842,
        "p3agnews": 0.932,
        "p3sciq": 0.84,
        "p3duorc": 0.26749999999999996,
        "p3appreviews": 0.73,
        "p3yelp": 0.725,
        "p3gigaword": 0.28200000000000003,
        "p3cosmosqa": 0.504,
        "p3quoref": 0.7184999999999999,
        "p3multinews": 0.154,
        "p3ropes": 0.459,
        "p3wiqa": 0.519,
        "p3cnndailymail": 0.212,
        "p3qasc": 0.951,
        "p3quarel": 0.545,
        "p3samsum": 0.417,
        "p3adversarialqa": 0.3825,
        "p3hotpotqa": 0.3,
        "p3rottentomatoes": 1.0,
        "p3paws": 0.881,
        "p3trec": 0.25,
        "p3commonsenseqa": 0.492,
        "p3quail": 0.593,
        "p3dbpedia14": 0.0,
    }
}
