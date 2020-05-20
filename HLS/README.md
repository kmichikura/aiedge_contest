# 高位合成ソースコード

## ファイル説明
- dlaip.py: NNgenを使用してDNNのハードウェアを生成するための高位合成コード
- forceDSP.py: 演算器を強制的にDSPへ割り当てるよう合成属性を付与するコード
- setenv.sh: 環境変数設定用スクリプト
- nngen/ : A Fully-Customizable Hardware Synthesis Compiler for Deep Neural Network
- verilogen/ : [A Mixed-Paradigm Hardware Construction Framework](https://github.com/PyHDI/veriloggen) NNgenで使用
- Pyverilog/ : [Python-based Hardware Design Processing Toolkit for Verilog HDL](https://github.com/PyHDI/Pyverilog) NNgenで使用

## セットアップ
```bash
$ python3.7 -m venv .venv
$ source .venv/bin/activate
$ pip install jinja2 numpy onnx
```

## 実行
```bash
$ source .venv/bin/activate
$(.venv) source setenv.sh
$(.venv) python3.7 dlaip.py
```

