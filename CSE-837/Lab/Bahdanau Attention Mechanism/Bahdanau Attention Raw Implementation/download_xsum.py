from datasets import load_dataset
import json


dataset = load_dataset("EdinburghNLP/xsum")


with open("xsum_train.json", "w") as f:
    json.dump(
        dataset["train"].shuffle(seed=42).select(range(8000)).to_list(),
        f,
        indent=2
    )


with open("xsum_validation.json", "w") as f:
    json.dump(
        dataset["validation"].shuffle(seed=42).select(range(500)).to_list(),
        f,
        indent=2
    )


with open("xsum_test.json", "w") as f:
    json.dump(
        dataset["test"].shuffle(seed=42).select(range(1500)).to_list(),
        f,
        indent=2
    )


print("XSum download complete")