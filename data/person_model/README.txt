Person detection models for the "Whole person" detector.

yolox_tiny.*  YOLOX-Tiny  (accurate, default)
yolox_nano.*  YOLOX-Nano  (fast)

Source: https://github.com/Megvii-BaseDetection/YOLOX (release 0.1.1rc0, files yolox_tiny.onnx and yolox_nano.onnx),
Copyright (c) Megvii Inc., licensed under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0).

Converted to the ncnn format with pnnx. The first layer of the network (the "focus" slicing) was removed from the
model file and is done by the plugin instead (src/person-detect-core.hpp), so the model input is 12 x 208 x 208.
