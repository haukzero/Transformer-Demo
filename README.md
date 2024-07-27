# Transformer Demo

跟着原论文 "Attention Is All You Need" 的主要框架走，一步一步用 pytorch 搭建一个简单的 Transformer 示例。

本项目主要借鉴了 [nlp-tutorial/5-1.Transformer](https://github.com/graykode/nlp-tutorial/tree/master/5-1.Transformer) ，并在其基础上进行模块文件拆分及重要代码注释，使其结构更加清晰。

- 2024.7.27 更新 :
  - 输入序列无需手动填充空位，更新`utils/sen2vec()`函数，自动将输入序列用填充字符填充至`max_len`大小
  - 更新`utils/vec2sen()`函数，自动去除末尾多余占位符和结束符
  - 更新模型的`greedy_decoder()`，自动解码直到遇到全结束符或全占位符或长度超限

## How to Start

```bash
python main.py
```

## 项目结构

- `model`:
  - `config.json`: 包含模型构建中需要用到的参数
  - `vocab.json`: 示例词表
- `utils.py`: 杂项函数
- `pe.py`: 位置编码模块
- `mha.py`: 多头注意力机制模块
- `ffn.py`: 前馈神经网络模块
- `enc.py`: 编码模块
- `dec.py`: 解码模块
- `trm.py`: Transformer 模型
- `main.py`: 主函数文件

## 原论文中关于模型搭建的重要图片

- 主要框架

![arch](img/arch.png)

- 位置编码

![pe](img/position_encoding.png)

- 注意力机制

![ai](img/attention_img.png)
![af](img/attention_formula.png)
![mha](img/multi-head.png)

- 前馈神经网络

![ffn](img/ffn.png)
