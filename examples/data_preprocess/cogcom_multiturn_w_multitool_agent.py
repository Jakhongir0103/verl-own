# Copyright 2023-2025 SGLang Team
# Copyright Amazon.com, Inc. or its affiliates.
# Copyright 2025 Reallm Labs Ltd. or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Preprocess the Geometry3k dataset to parquet format
"""

import argparse
import os

import datasets
import pandas

from verl.utils.hdfs_io import copy, makedirs
from PIL import Image

def filter_image_on_size(img_path, min_size=28):
    with open(img_path, "rb") as f:
        pil_image = Image.open(f).convert("RGBA")
        if pil_image.size[0] >= min_size and pil_image.size[1] >= min_size:
            return True
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/workspace/verl/data/geo3k_multiturn_w_tool")
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()
    # data_source = "qijimrc/CoMDataset"
    # dataset = datasets.load_dataset(data_source, 'com_math')
    data_dir = '/users/jsaydali/scratch/data/comdataset/'
    data = pandas.read_json(os.path.join(data_dir, 'com_math.jsonl'), lines=True)
    data_new = data.explode("metadata", ignore_index=True)
    data_metadata = pandas.json_normalize(data_new["metadata"])
    data = pandas.concat([data_new, data_metadata], axis=1)[['pid', 'image_path', 'question', 'answer']]
    data['image_path'] = data['image_path'].apply(lambda path: os.path.join(data_dir, path))
    data = data[data['image_path'].map(filter_image_on_size)]
    dataset = datasets.Dataset.from_pandas(data)

    # Split the loaded dataset into train and test if not already split
    if not isinstance(dataset, dict) and not hasattr(dataset, "keys"):
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags. "
        r"The final answer MUST BE put in \boxed{}."
    )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            # TODO: create [problem, answer, images]
            problem = example.pop("question")
            prompt = problem + " " + instruction_following
            answer = example.pop("answer")
            image_path = example.pop("image_path")
            data = {
                "data_source": 'custom_multimodal',
                "agent_name": "tool_agent",
                "prompt": [
                    {
                        "role": "system",
                        "content": (
                            "You are given an image and a question about it, and you need to answer the question based on the image. "
                            "You should use the available tools to help you answer the question, if necessary: `image_bbox_tool`, `image_flip_tool`, `image_line_tool`, `image_rotate_tool`, `image_crop_tool`. "
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "images": [image_path],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "image_bbox_tool": {
                            "create_kwargs": {"image": image_path},
                            # "execute_kwargs": {},
                            # "calc_reward_kwargs": {},
                            # "release_kwargs": {},
                        },
                        "image_flip_tool": {
                            "create_kwargs": {"image": image_path},
                            # "execute_kwargs": {},
                            # "calc_reward_kwargs": {},
                            # "release_kwargs": {},
                        },
                        "image_line_tool": {
                            "create_kwargs": {"image": image_path},
                            # "execute_kwargs": {},
                            # "calc_reward_kwargs": {},
                            # "release_kwargs": {},
                        },
                        "image_rotate_tool": {
                            "create_kwargs": {"image": image_path},
                            # "execute_kwargs": {},
                            # "calc_reward_kwargs": {},
                            # "release_kwargs": {},
                        },
                        "image_crop_tool": {
                            "create_kwargs": {"image": image_path},
                            # "execute_kwargs": {},
                            # "calc_reward_kwargs": {},
                            # "release_kwargs": {},
                        },
                    },
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)
    train_dataset = train_dataset.cast_column("images", datasets.Sequence(datasets.Image()))
    test_dataset = test_dataset.cast_column("images", datasets.Sequence(datasets.Image()))
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
