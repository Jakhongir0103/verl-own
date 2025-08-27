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

from verl.utils.hdfs_io import copy, makedirs
from PIL import Image

# Demo data
demo_data = [
    {
        'image': '/users/jsaydali/scratch/data/debug/000000000070.jpg',
        'width': 640,
        'height': 480,
        'left': 52,
        'top': 372,
        'right': 636,
        'bottom': 443,
        'question': "What equipment is used for snowboarding?"
    },
    {
        'image': '/users/jsaydali/scratch/data/debug/000000000092.jpg',
        'width': 640,
        'height': 427,
        'left': 417,
        'top': 120,
        'right': 583,
        'bottom': 426,
        'question': "What do you use to eat a cake?"
    },
    {
        'image': '/users/jsaydali/scratch/data/debug/000000000141.jpg',
        'width': 640,
        'height': 399,
        'left': 3,
        'top': 33,
        'right': 110,
        'bottom': 216,
        'question': "Where do we control the water from?"
    },
    {
        'image': '/users/jsaydali/scratch/data/debug/000000000257.jpg',
        'width': 640,
        'height': 480,
        'left': 101,
        'top': 415,
        'right': 176,
        'bottom': 479,
        'question': "What is used for carrying a baby?"
    },
    {
        'image': '/users/jsaydali/scratch/data/debug/000000000335.jpg',
        'width': 640,
        'height': 480,
        'left': 270,
        'top': 16,
        'right': 345,
        'bottom': 86,
        'question': "What do we use to watch movies?"
    },
    {
        'image': '/users/jsaydali/scratch/data/debug/000000000410.jpg',
        'width': 640,
        'height': 480,
        'left': 38,
        'top': 103,
        'right': 303,
        'bottom': 362,
        'question': "What can cut paper easily?"
    },
    {
        'image': '/users/jsaydali/scratch/data/debug/000000000412.jpg',
        'width': 640,
        'height': 426,
        'left': 471,
        'top': 162,
        'right': 638,
        'bottom': 423,
        'question': "What living creature can speak?"
    },
    {
        'image': '/users/jsaydali/scratch/data/debug/000000000436.jpg',
        'width': 427,
        'height': 640,
        'left': 221,
        'top': 237,
        'right': 321,
        'bottom': 310,
        'question': "What do we eat for dessert?"
    },
    {
        'image': '/users/jsaydali/scratch/data/debug/000000000459.jpg',
        'width': 516,
        'height': 640,
        'left': 372,
        'top': 245,
        'right': 475,
        'bottom': 316,
        'question': "What can be used to take pictures?"
    },
    {
        'image': '/users/jsaydali/scratch/data/debug/000000000492.jpg',
        'width': 640,
        'height': 383,
        'left': 166,
        'top': 248,
        'right': 281,
        'bottom': 353,
        'question': "What do parents love and care for?"
    },
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/workspace/verl/data/geo3k_multiturn_w_tool")
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    # data_source = "/iopsstor/scratch/cscs/jsaydali/text_test/generate_images/text_sizes/text_recognition_dataset"
    # dataset = datasets.load_from_disk(data_source)

    demo_data = demo_data * 500
    dataset = datasets.Dataset.from_list(demo_data)

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
            # problem = example.pop("problem")
            # prompt = problem + " " + instruction_following
            # answer = example.pop("answer")
            # images = example.pop("images")

            problem = example.pop("question")
            prompt = problem + " " + instruction_following
            answer = f"[{example.pop('left')}, {example.pop('top')}, {example.pop('right')}, {example.pop('bottom')}]"
            images = [example.pop("image")]

            # read each image in images as PIL
            pil_images = []
            for img_path in images:
                with open(img_path, "rb") as f:
                    pil_images.append(Image.open(f).convert("RGBA"))                
            data = {
                "data_source": 'custom_multimodal',
                "agent_name": "tool_agent",
                "prompt": [
                    {
                        "role": "system",
                        "content": (
                            "You are given an image with a text in it and you need to read the text. "
                            "You should use the `image_crop_tool` tool to crop a specific area of the image, if necessary. "
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "images": images,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "image_crop_tool": {
                            "create_kwargs": {"image": pil_images[0]},
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
