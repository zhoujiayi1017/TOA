# TOA_throat_regression
喉位置回帰のモバイル版開発用リポジトリ

# Contents
1. BlazeFace  
   顔検出用モデル
2. FaceMesh  
   顔の詳細特徴取得用モデル
3. posenet  
   頭の位置を含めた身体の17個のキーポイントを検出
4. BlazePose  
   高精度なポーズ検出モデル(未実装)
5. throat_regression  
   メインで開発する場所。
   動画に対する喉回帰モデルを作成
6. C++_model  
   python実装をC++に換装する（予定）
7. samples  
   色々なサンプルを入れておく

# TODO
- C++でfacemeshの推論を動作させる(速度の比較も)
- pythonモデルを喉回帰ができるように書き換える
- 手の回帰モデルについて検討する
