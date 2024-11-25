# waypoint modifier

waypoint を編集及び可視化するツール

## 前提環境

- python (>=3.10)
- pip (>= 22.0.2)

## Getting Started

### インストール

```shell
https://github.com/AbudoriLab-TC2024/waypoint-modifier.git
cd waypoint-modifier
pip install .

# 必要に応じて
export PATH=$PATH:$HOME/.local/bin
```

### 点列の編集

```shell
resampler <path/to/localization-pose.csv> [-o <path/to/output.csv>]
```

### 点列の描画

```shell
viewer <path/to/waypoint.csv>
```

以下のようなメッセージが出るので、URLをブラウザでアクセスする。

```
Dash is running on http://127.0.0.1:8050/

 * Serving Flask app 'waypoint_modifier.viewer'
 * Debug mode: on
```


## 詳細

### 点列の編集

入力の（例えば自己位置推定の）Poseリストを間引く。0番目の点から走査していき、以下のOR条件を満たした時にその点を採用する。

- 点の間隔の累和が `MAX_ACCUM_DIST` を越えた時
- 曲率の累和が `MAX_ACCUM_CURV` を越えた時

ただし、 停止時に点が重複するのを除去するために `MIN_ACCUM_DIST` 以上の距離を走行していることを制約に加えている。

出力時は座標の列名を `x` と `y` に変更する。また、入力のCSVに`action` 列がある場合、それをコピーする。ない場合空の`action`列を追加する。


### 点列の描画

waypoint の CSVを可視化するWebアプリ。pip インストール時にインターネットアクセスが必要だが、一度インストールすればオフラインで動く(はず)。
以下の列が必要。

- x : x座標
- y : y座標
- action : その点でのアクション (空白 or `continue` or `stop`)

`i-1` → `i` の点のベクトルを使って yaw角を計算している。これは [follow_path.py](https://github.com/AbudoriLab-TC2024/penguin_nav?tab=readme-ov-file#follow_pathpy)と同様の実装となる。0番目の点のyaw角は0になる。

`viewer` 自体に点列を編集機能はない。読み込んだファイルを監視しているので、ファイルを編集すればそれが反映される。
