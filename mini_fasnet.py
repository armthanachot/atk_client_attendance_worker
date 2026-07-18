from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


# Adapted from minivision-ai/Silent-Face-Anti-Spoofing
# src/model_lib/MiniFASNet.py (Apache-2.0).


class Flatten(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.view(value.size(0), -1)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: tuple[int, int] = (1, 1),
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.prelu(self.bn(self.conv(value)))


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: tuple[int, int] = (1, 1),
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(value))


class DepthWise(nn.Module):
    def __init__(
        self,
        c1: tuple[int, int],
        c2: tuple[int, int],
        c3: tuple[int, int],
        residual: bool = False,
        kernel: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (2, 2),
        padding: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        self.conv = ConvBlock(c1_in, c1_out)
        self.conv_dw = ConvBlock(
            c2_in,
            c2_out,
            groups=c2_in,
            kernel=kernel,
            padding=padding,
            stride=stride,
        )
        self.project = LinearBlock(c3_in, c3_out)
        self.residual = residual

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        shortcut = value
        value = self.project(self.conv_dw(self.conv(value)))
        return shortcut + value if self.residual else value


class Residual(nn.Module):
    def __init__(
        self,
        c1: list[tuple[int, int]],
        c2: list[tuple[int, int]],
        c3: list[tuple[int, int]],
        num_block: int,
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            *[
                DepthWise(c1[index], c2[index], c3[index], residual=True, stride=(1, 1))
                for index in range(num_block)
            ],
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model(value)


class SEModule(nn.Module):
    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        module_input = value
        value = self.avg_pool(value)
        value = self.relu(self.bn1(self.fc1(value)))
        value = self.sigmoid(self.bn2(self.fc2(value)))
        return module_input * value


class DepthWiseSE(DepthWise):
    def __init__(
        self,
        c1: tuple[int, int],
        c2: tuple[int, int],
        c3: tuple[int, int],
        residual: bool = False,
        kernel: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (2, 2),
        padding: tuple[int, int] = (1, 1),
        se_reduct: int = 8,
    ) -> None:
        super().__init__(c1, c2, c3, residual, kernel, stride, padding)
        self.se_module = SEModule(c3[1], se_reduct)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        shortcut = value
        value = self.project(self.conv_dw(self.conv(value)))
        if self.residual:
            value = self.se_module(value)
            return shortcut + value
        return value


class ResidualSE(nn.Module):
    def __init__(
        self,
        c1: list[tuple[int, int]],
        c2: list[tuple[int, int]],
        c3: list[tuple[int, int]],
        num_block: int,
        se_reduct: int = 4,
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        for index in range(num_block):
            block_type = DepthWiseSE if index == num_block - 1 else DepthWise
            kwargs = {"se_reduct": se_reduct} if block_type is DepthWiseSE else {}
            modules.append(
                block_type(
                    c1[index],
                    c2[index],
                    c3[index],
                    residual=True,
                    stride=(1, 1),
                    **kwargs,
                ),
            )
        self.model = nn.Sequential(*modules)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model(value)


KEEP_DICT = {
    "1.8M": [
        32,
        32,
        103,
        103,
        64,
        13,
        13,
        64,
        26,
        26,
        64,
        13,
        13,
        64,
        52,
        52,
        64,
        231,
        231,
        128,
        154,
        154,
        128,
        52,
        52,
        128,
        26,
        26,
        128,
        52,
        52,
        128,
        26,
        26,
        128,
        26,
        26,
        128,
        308,
        308,
        128,
        26,
        26,
        128,
        26,
        26,
        128,
        512,
        512,
    ],
    "1.8M_": [
        32,
        32,
        103,
        103,
        64,
        13,
        13,
        64,
        13,
        13,
        64,
        13,
        13,
        64,
        13,
        13,
        64,
        231,
        231,
        128,
        231,
        231,
        128,
        52,
        52,
        128,
        26,
        26,
        128,
        77,
        77,
        128,
        26,
        26,
        128,
        26,
        26,
        128,
        308,
        308,
        128,
        26,
        26,
        128,
        26,
        26,
        128,
        512,
        512,
    ],
}


class MiniFASNet(nn.Module):
    def __init__(
        self,
        keep: list[int],
        embedding_size: int,
        conv6_kernel: tuple[int, int],
        drop_p: float = 0.2,
        num_classes: int = 3,
        img_channel: int = 3,
    ) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.conv1 = ConvBlock(img_channel, keep[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = ConvBlock(keep[0], keep[1], kernel=(3, 3), padding=(1, 1), groups=keep[1])
        self.conv_23 = DepthWise((keep[1], keep[2]), (keep[2], keep[3]), (keep[3], keep[4]))
        self.conv_3 = Residual(
            [(keep[4], keep[5]), (keep[7], keep[8]), (keep[10], keep[11]), (keep[13], keep[14])],
            [(keep[5], keep[6]), (keep[8], keep[9]), (keep[11], keep[12]), (keep[14], keep[15])],
            [(keep[6], keep[7]), (keep[9], keep[10]), (keep[12], keep[13]), (keep[15], keep[16])],
            4,
        )
        self.conv_34 = DepthWise((keep[16], keep[17]), (keep[17], keep[18]), (keep[18], keep[19]))
        self.conv_4 = Residual(
            [
                (keep[19], keep[20]),
                (keep[22], keep[23]),
                (keep[25], keep[26]),
                (keep[28], keep[29]),
                (keep[31], keep[32]),
                (keep[34], keep[35]),
            ],
            [
                (keep[20], keep[21]),
                (keep[23], keep[24]),
                (keep[26], keep[27]),
                (keep[29], keep[30]),
                (keep[32], keep[33]),
                (keep[35], keep[36]),
            ],
            [
                (keep[21], keep[22]),
                (keep[24], keep[25]),
                (keep[27], keep[28]),
                (keep[30], keep[31]),
                (keep[33], keep[34]),
                (keep[36], keep[37]),
            ],
            6,
        )
        self.conv_45 = DepthWise((keep[37], keep[38]), (keep[38], keep[39]), (keep[39], keep[40]))
        self.conv_5 = Residual(
            [(keep[40], keep[41]), (keep[43], keep[44])],
            [(keep[41], keep[42]), (keep[44], keep[45])],
            [(keep[42], keep[43]), (keep[45], keep[46])],
            2,
        )
        self.conv_6_sep = ConvBlock(keep[46], keep[47])
        self.conv_6_dw = LinearBlock(keep[47], keep[48], groups=keep[48], kernel=conv6_kernel)
        self.conv_6_flatten = Flatten()
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.drop = nn.Dropout(p=drop_p)
        self.prob = nn.Linear(embedding_size, num_classes, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.conv1(value)
        value = self.conv2_dw(value)
        value = self.conv_23(value)
        value = self.conv_3(value)
        value = self.conv_34(value)
        value = self.conv_4(value)
        value = self.conv_45(value)
        value = self.conv_5(value)
        value = self.conv_6_sep(value)
        value = self.conv_6_dw(value)
        value = self.conv_6_flatten(value)
        if self.embedding_size != 512:
            value = self.linear(value)
        value = self.bn(value)
        value = self.drop(value)
        return self.prob(value)


class MiniFASNetSE(MiniFASNet):
    def __init__(
        self,
        keep: list[int],
        embedding_size: int,
        conv6_kernel: tuple[int, int],
        drop_p: float = 0.75,
        num_classes: int = 3,
        img_channel: int = 3,
    ) -> None:
        super().__init__(keep, embedding_size, conv6_kernel, drop_p, num_classes, img_channel)
        self.conv_3 = ResidualSE(
            [(keep[4], keep[5]), (keep[7], keep[8]), (keep[10], keep[11]), (keep[13], keep[14])],
            [(keep[5], keep[6]), (keep[8], keep[9]), (keep[11], keep[12]), (keep[14], keep[15])],
            [(keep[6], keep[7]), (keep[9], keep[10]), (keep[12], keep[13]), (keep[15], keep[16])],
            4,
        )
        self.conv_4 = ResidualSE(
            [
                (keep[19], keep[20]),
                (keep[22], keep[23]),
                (keep[25], keep[26]),
                (keep[28], keep[29]),
                (keep[31], keep[32]),
                (keep[34], keep[35]),
            ],
            [
                (keep[20], keep[21]),
                (keep[23], keep[24]),
                (keep[26], keep[27]),
                (keep[29], keep[30]),
                (keep[32], keep[33]),
                (keep[35], keep[36]),
            ],
            [
                (keep[21], keep[22]),
                (keep[24], keep[25]),
                (keep[27], keep[28]),
                (keep[30], keep[31]),
                (keep[33], keep[34]),
                (keep[36], keep[37]),
            ],
            6,
        )
        self.conv_5 = ResidualSE(
            [(keep[40], keep[41]), (keep[43], keep[44])],
            [(keep[41], keep[42]), (keep[44], keep[45])],
            [(keep[42], keep[43]), (keep[45], keep[46])],
            2,
        )


def get_kernel(height: int, width: int) -> tuple[int, int]:
    return ((height + 15) // 16, (width + 15) // 16)


def minifasnet_v1(height: int, width: int, num_classes: int = 3) -> MiniFASNet:
    return MiniFASNet(KEEP_DICT["1.8M"], 128, get_kernel(height, width), num_classes=num_classes)


def minifasnet_v2(height: int, width: int, num_classes: int = 3) -> MiniFASNet:
    return MiniFASNet(KEEP_DICT["1.8M_"], 128, get_kernel(height, width), num_classes=num_classes)


def minifasnet_v1se(height: int, width: int, num_classes: int = 3) -> MiniFASNetSE:
    return MiniFASNetSE(KEEP_DICT["1.8M"], 128, get_kernel(height, width), num_classes=num_classes)


def minifasnet_v2se(height: int, width: int, num_classes: int = 4) -> MiniFASNetSE:
    return MiniFASNetSE(KEEP_DICT["1.8M_"], 128, get_kernel(height, width), num_classes=num_classes)


MODEL_MAPPING = {
    "MiniFASNetV1": minifasnet_v1,
    "MiniFASNetV2": minifasnet_v2,
    "MiniFASNetV1SE": minifasnet_v1se,
    "MiniFASNetV2SE": minifasnet_v2se,
}


def build_minifasnet(model_type: str, height: int, width: int, num_classes: int = 3) -> nn.Module:
    if model_type not in MODEL_MAPPING:
        raise ValueError(f"Unsupported MiniFASNet model type: {model_type}")
    return MODEL_MAPPING[model_type](height, width, num_classes)


def parse_model_name(model_name: str) -> tuple[int, int, str]:
    info = model_name.split("_")[0:-1]
    height, width = info[-1].split("x")
    model_type = model_name.split(".pth")[0].split("_")[-1]
    return int(height), int(width), model_type


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    first_key = next(iter(state_dict))
    if not first_key.startswith("module."):
        return state_dict
    return {key[7:]: value for key, value in state_dict.items()}
