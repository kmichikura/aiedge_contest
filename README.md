# aiedge_contest
[第2回AIエッジコンテスト](https://signate.jp/competitions/191) チームFTの最終提出物

## 動作確認済み環境
### 高位合成
- CPU: x86 CPU
- OS: LinuxOS (Distribution: Ubuntu 16.04 LTS)
### アプリケーションソフト, JSONフォーマット変換ソフト
- Platform: Avnet社Ultra96（Zynq UltraScale+ MPSoC ZAU3EG SBVA484）FPGAボード
- OS: Linux Kernel linux-xlnx tag=xilinx-v2018.3
- RFS: debian9

## 提出物
- [高位合成用ソースコード](https://github.com/kmichikura/aiedge_contest/tree/master/HLS)
- [RTL](https://github.com/kmichikura/aiedge_contest/tree/master/RTL)
- [アプリケーションソースコード](https://github.com/kmichikura/aiedge_contest/tree/master/APP)
- [DNN出力を指定JSONフォーマット変換ソフト](https://github.com/kmichikura/aiedge_contest/tree/master/OTHER_APP)

## インストール
```
$ git clone https://github.com/kmichikura/aiedge_contest.git
$ cd HLS/
$ git submodule init
$ git submodule update
```
