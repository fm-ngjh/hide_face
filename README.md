# 概要
画像認識系の講義の課題で作成しているプログラムです．オープンなSNSへの写真投稿を想定し，写真に写った複数の顔を検出して自動で隠すのですが，その際画像内での重要度に応じて被写体とその他(写りこんでしまった人)を区別して隠し方を変えたい(顔が隠れていてもどの人物を撮った写真なのかというのがパッと見でわかるようにしたい)というコンセプトで作成しています．

deepfaceのライブラリを使用し，顔の検出と表情検出を行っています．現在は「画像全体に対し顔が占める割合が大きいほど重要度が高い」として評価し，重要度が高い人物は検出された感情別のスタンプで，それ以外の人物はモザイクで顔を隠しています．

# 実行例

<table>
  <tr>
    <td><img src="https://github.com/fm-ngjh/hide_face/assets/135797163/b671be5d-ebcc-4583-b1c5-4e4343af5312"></td>
    <td><img src="https://github.com/fm-ngjh/hide_face/assets/135797163/48f0c17e-7d5b-4038-a53f-0d7df5e4b45b"></td>
  </tr>
</table> 

現在は重要度を被写体とそれ以外で区切るラインを画像ごとに指定する(例えば事前に被写体の数を入力しておく等)必要があり，汎用性に欠けます．この問題を解決するために，データセットを自作して機械学習による最適な閾値の設定が行えるようにすることを考えています．
(写真はフリー素材を使用しています)

