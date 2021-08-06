#!/bin/bash
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

pip3 install svgwrite playsound
pip3 install --extra-index-url https://google-coral.github.io/py-repo/ pycoral
[ ! -d "models" ] && mkdir -p "models"
cd models
rm mobilenet_v2_1.0_224_quant_edgetpu.tflite
wget https://dl.google.com/coral/canned_models/mobilenet_v2_1.0_224_quant_edgetpu.tflite
cd ..
[ ! -d "labels" ] && mkdir -p "labels"
cd labels
rm imagenet_labels.txt
wget https://dl.google.com/coral/canned_models/imagenet_labels.txt
cd ..
[ ! -d "sdcard_directory" ] && mkdir -p "sdcard_directory"