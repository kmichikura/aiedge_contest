NNgen
==============================

A Fully-Customizable Hardware Synthesis Compiler for Deep Neural Network

Copyright 2017, Shinya Takamaeda-Yamazaki and Contributors


ライセンス
==============================

Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)


NNgenとは?
==============================

NNgenはディープニューラルネットワークのモデル特化ハードウェアアクセラレータを合成するオープンソースのコンパイラです。NNgenは入力モデル定義からDNNアクセラレータのVerilog HDLのソースコードとIPコアのパッケージを生成します。

生成されたハードウェアは、プロセッシングエンジン、オンチップメモリ、オンチップネットワーク、DMAコントローラ、制御回路といったすべてが含まれています。そのため、生成されたハードウェアは、処理を開始した後は、外部の回路やCPUなどからの追加の制御をまったく必要としません。

NNgenのバックエンドには、複数の記述パラダイムが利用できるオープンソースの高位合成コンパイラVeriloggenを用いています。そのため、新しいアルゴリズムやアプリケーションに応じて、NNgenの構成をカスタマイズすることができます。


NNgenにコントリビュートするには
==============================

NNgenプロジェクトは質問、バグ報告、新しい機能提案、[GitHub](https://github.com/NNgen/nngen)上でのプルリクエストを常に歓迎します。

コミュニティ管理者 
--------------------

このプロジェクトの管理者として、コミュニティ管理を主導し、ソフトウェアの開発と普及を促進する役割を担います。

コミッタ
--------------------

コミッタは、プロジェクトへの書き込み権限を付与されている個人です。コミッタとして貢献するためには、コミュニティ管理者の承認が必要です。貢献の分野は、コードの貢献とレビュー、ドキュメント、教育、そしてアウトリーチなど、あらゆる形態をとることができます。高品質で健全なプロジェクトにはコミッタが不可欠です。コミュニティは貢献者からの新しいコミッタを積極的に探しています。

レビュアー
--------------------

レビュアーは、プロジェクトに積極的に貢献し、新しい貢献のコードレビューに積極的に参加する個人です。積極的な貢献者の中から査読者を決定します。 コミッタはレビュアーにレビューを明示的に依頼してください。 高品質のコードレビューは長期にわたる技術的債務を防ぎ、プロジェクトの成功に不可欠です。 プロジェクトへのプルリクエストは、マージするために少なくとも1人のレビュアーによってレビューされる必要があります。

質問、バグ報告、新しい機能提案について
--------------------

GitHubの[issue tracker](https://github.com/NNgen/nngen/issues)にコメントを残してください。

プルリクエストについて
--------------------

プルリクエストを提供してくれた貢献者を確認するためには "CONTRIBUTORS.md"をチェックしてください。

NNgenは統合テストにpytestフレームワークを使います。 プルリクエストを送信するときは、pytestにテスト例を含めてください。 テストコードを書くには、 "tests"ディレクトリにある既存のテスト例を参照してください。

プルリクエスト対象のコードがすべてのテストをパスし、明白な問題がなければ、それはコミッタによってdevelopブランチにマージされます。


インストール
==============================

要件
--------------------

- Python3: 3.6 or later
- Icarus Verilog: 10.1 or later

```
sudo apt install python3 python3-pip iverilog
```

- Pyverilog: 1.1.4 or later
- Veriloggen: 1.7.3 or later
- Jinja2: 2.10 or later
- NumPy: 1.14 or later
- ONNX: 1.6.0 or later

```
pip3 install veriloggen pyverilog jinja2 numpy onnx
```

ドキュメント生成のための要件
--------------------

- TeX Live: 2015 or later
- dvipng: 1.15 or later

```
sudo apt install texlive-science texlive-fonts-recommended texlive-fonts-extra dvipng
```

- Sphinx: 2.10 or later
- sphinx_rtd_theme : 0.4.3 or later

```
pip3 install sphinx sphinx_rtd_theme
```

別のインストール方法
--------------------
    
現在のNNgenとVeriloggenは活発に開発が進んでいるので、最新の機能を利用する場合には、GitHubから最新バージョンのNNgen、Veriloggen、その他のライブラリをダウンロード（またはgit clone）し、適宜パスを通して利用することもできます。

### GitHubから最新のNNgen、Veriloggen、その他のライブラリをダウンロード

```
git clone https://github.com/NNgen/nngen.git
git clone https://github.com/PyHDI/veriloggen.git
git clone https://github.com/PyHDI/Pyverilog.git
```

### VeriloggenとNNgenから他のライブラリへのシンボリックリンクを作成

システムにインストールする代わりに、依存ライブラリへのシンボリックリンクを作成してください。

```
cd nngen
ln -s ../veriloggen/veriloggen
ln -s ../Pyverilog/pyverilog
```


Getting Started
==============================

準備中です。


関連プロジェクト
==============================

[Veriloggen](https://github.com/PyHDI/veriloggen)
- PythonでVerilog HDLソースコードを構築するためのライブラリ

[Pyverilog](https://github.com/PyHDI/Pyverilog)
- PythonベースのVerilog HDLハードウェア設計処理ツールキット
