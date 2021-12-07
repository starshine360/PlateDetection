# 检测车牌

## Introduction

该项目简略实现了在图片中检测车牌的功能，仅适用于车牌亮度较强且与背景有较高对比度的场景。算法设计和代码实现上大多参考了[知乎文章](https://zhuanlan.zhihu.com/p/102203294)，特此注明。


## Prerequisite

运行本项目的程序前，请先检查是否满足如下条件：

* python (above version 3.7)
* opencv-python (above version 4)
* numpy (latest version)

## Usage

在当前目录下新建目录：`data/test/` 和 `data/temp/`. 将需检测的图片放置在目录 `data/test/` 下，然后修改 `main.py` 中 `img_name` 的值为该图片的文件名。键入 `python main.py`，等待程序运行完毕。检测结果将存于目录 `data/result/` 中。