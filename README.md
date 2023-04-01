# SELDNET-DOA-pytorch

![PyTorch Version](https://img.shields.io/badge/PyTorch-1.x-red) ![Python Version](https://img.shields.io/badge/Python-3.7+-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Windows](https://img.shields.io/badge/OS-Windows-blue) ![macOS](https://img.shields.io/badge/OS-macOS-lightgrey) ![Linux](https://img.shields.io/badge/OS-Linux-green)

## 目录

1. [项目简介](#项目简介)
2. [依赖环境](#依赖环境)
3. [安装指南](#安装指南)
4. [数据准备](#数据准备)
5. [训练](#训练)
6. [测试](#测试)
7. [结果可视化](#结果可视化)
8. [致谢](#致谢)
9. [参考文献](#参考文献)
10. [许可证](#许可证)

## 项目简介

SELDNET-DOA-pytorch是一个基于PyTorch实现的开源项目，专注于声源事件定位、检测和跟踪（SELDT）。该项目旨在帮助研究人员和开发人员更轻松地探索SELDT任务的实现，并进行实验。本项目的灵感来自于IEEE
AASP Workshop DCASE上的研究挑战。 在SELDNET-DOA-pytorch中，我们提供了以下特点：

- 适用于静态和动态场景的声源事件定位、检测和跟踪（SELDT）的深度学习模型。

- 使用PyTorch框架构建，易于理解和修改。

- 包括用于训练和测试模型的示例数据集。

## 功能概述

- **静态场景**：项目提供了一种基于SELDnet的方法，用于处理具有空间静态源的静态场景。这是对DCASE 2019年SELD挑战数据集的应用。
- **动态场景**：项目还展示了一种基于SELDnet的方法，用于处理具有不同角速度移动的源的动态场景。这是对DCASE 2020年SELD挑战数据集的应用。
- **易用性**：我们提供了简洁易懂的代码和相关数据集，方便您进行二次开发和实验。

## 相关研究

本项目基于一系列与声源检测（SED）、方位角（DOA）估计以及综合任务声源事件定位、检测和跟踪（SELDT）相关的论文。详细论文列表可以参考[这里](https://github.com/Sharathadavanne/seld-dcase2020)。

## 依赖环境

## 安装指南

## 数据准备

论文中使用的数据集，都已发布在 zenodo.org 上。这些数据集的大小在 30-45 GB 之间。

前五个数据集由静止点源组成，每个点源与一个空间坐标相关联。而最后两个数据集包含具有不同角速度的移动点源。

发布的数据集包括：

1. ANSIM（TUT Sound Events 2018 - Ambisonic, Anechoic and Synthetic Impulse Response Dataset）
2. RESIM（TUT Sound Events 2018 - Ambisonic, Reverberant and Synthetic Impulse Response Dataset）
3. CANSIM（TUT Sound Events 2018 - Circular array, Anechoic and Synthetic Impulse Response Dataset）
4. CRESIM（TUT Sound Events 2018 - Circular array, Reverberant and Synthetic Impulse Response Dataset）
5. REAL（TUT Sound Events 2018 - Ambisonic, Reverberant and Real-life Impulse Response Dataset）
6. Real-life impulse responses to simulate custom SELD datasets
7. MANSIM（TAU Moving Sound Events 2019 - Ambisonic, Anechoic, Synthetic Impulse Response and Moving Sources Dataset）
8. MREAL（TAU Moving Sound Events 2019 - Ambisonic, Reverberant, Real-life Impulse Response and Moving Sources Dataset）

所有数据集都包含三个子数据集，最多有一个（ov1）、两个（ov2）和三个（ov3）时间重叠的声音事件。每个子数据集都有三个交叉验证分割（split1、split2
和 split3）。总共每个数据集有九个分割，保存为单独的 zip 文件。为了测试 SELDnet 代码，您不必下载整个数据集。您只需下载其中一个
zip 文件，然后针对相应的重叠（ov）和分割（split）训练 SELDnet。

例如：如果要下载第一个数据集ANSIM（TUT Sound Events 2018 - Ambisonic, Anechoic and Synthetic Impulse Response
Dataset），只要下载以下九个.zip中的一个即可：

![image-20230402012739565](https://image-1304916025.cos.ap-nanjing.myqcloud.com/typora-typora-image-20230402012739565.png)

## 训练

## 测试

## 结果可视化

## 致谢

## 参考文献

## 许可证
