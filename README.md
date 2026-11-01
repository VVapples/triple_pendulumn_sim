# Triple pendulum simulation

このフォルダは、平面内を運動する三重振り子を計算し、初期角度
\((\theta_1,\theta_2,\theta_3)\) の空間でカオスの強さを可視化するための
プログラムです。フォルダ名の `triple` は課題で使用している表記に合わせています。

## 理論と数値計算法

各棒の角度を、鉛直下向きから測った絶対角度とします。一般化座標を

```text
q = (theta1, theta2, theta3)
```

とし、質量行列 `M(q)` を用いてハミルトニアンを

```text
H(q, p) = 1/2 p^T M(q)^(-1) p + V(q)
```

と表します。状態は角度と正準運動量を合わせた
`z = (q, p)` です。

通常の陽的Euler法では、長時間計算でエネルギーがドリフトし、軌道が
物理的に不自然になることがあります。ここではGonzalez型の離散勾配法を
使います。離散勾配 `gbar` は

```text
gbar · (z1 - z0) = H(z1) - H(z0)
```

を満たすように作られるため、正準構造と組み合わせると、Newton反復が
十分収束した場合に各ステップでエネルギーを丸め誤差程度に保存します。
この実装では、Newton法の初期値にRK4の予測値を使い、ヤコビ行列は中心差分
で計算しています。したがって、エネルギー保存を優先しつつ、単純な
陽的法より長時間積分の数値誤差を抑えられます。
Newton反復には減衰更新も入れており、RK4予測値が陰的方程式の解から
少し離れている初期条件でも、残差を減少させながら収束させます。
さらに、特定の状態でNewton反復が局所的に失敗した場合は、そのステップだけを
最大6回まで半分に分割して再試行します。各分割ステップ自体は同じ離散勾配法
なので、単純に別の積分法へ切り替えているわけではありません。

## カオスの評価

ここでいうカオスの強さは、厳密な最大リアプノフ指数ではありません。
有限時間後の初期条件への感度を次で評価します。

```text
C(i,j,k) = mean(||wrap(theta_final(neighbor)
                         - theta_final(center))||)
```

`neighbor` は初期角度格子における6方向
`(+/-theta1, +/-theta2, +/-theta3)` の隣接点です。角度は周期変数なので、
差は `atan2(sin(delta), cos(delta))` で `[-pi, pi]` に折り返します。
`C` が大きいほど、初期角度の小さな違いが有限時間後に大きな違いになった
ことを意味し、カオス的な領域として描画します。

## ファイル

### `triple_pendulum.py`

`TriplePendulumSimulator` が単一の三重振り子を担当します。

- `simulate(...)`: 時系列、角度、角速度、エネルギーを返す
- `simulate_final(...)`: 終了時の角度だけを返す
- `positions(...)`: 棒と質点の平面位置を返す
- `finite_time_sensitivity(...)`: 6方向の摂動を直接計算する
- `finite_time_sensitivity_from_state(...)`: 現在状態の周囲を短時間評価する
- `SimulationResult.energy_error`: 初期エネルギーとの差

### `plot_chaos_volume.py`

初期角度空間の格子を `ProcessPoolExecutor` で並列計算します。計算後、
6近傍との差を求め、3次元散布図の色と透明度に変換します。境界点は
6近傍が揃わないため描画対象から除外しています。

### `realtime_triple_pendulum.py`

単一の三重振り子をMatplotlib/Tkinterでリアルタイム表示します。
各フレームで `TriplePendulumSimulator.step()` を呼び出すため、
`triple_pendulum.py` と同じ離散勾配法を実際のアニメーションで確認できます。
右側にはエネルギー誤差を表示します。Newton反復が失敗した場合は計算を停止し、
ウィンドウ下部に例外の種類とメッセージを表示します。エラーを無視して別の
積分法へ切り替えることはしません。
また、現在状態の6方向に微小な角度摂動を加えたコピーを短時間進め、
実数のカオス評価値を表示します。リアルタイム表示では、これまでに得た
評価値の最小値と最大値を使って次のように正規化し、棒と質点の透明度に
反映します。

```text
alpha = clip((chaos - min_chaos) / (max_chaos - min_chaos), 0.08, 1.0)
```

単一振り子では「領域内の全振り子」との比較はできないため、ここでの
最小値・最大値はリアルタイム履歴の範囲です。3次元初期角度空間の全点で
比較する場合は、`plot_chaos_volume.py` のように全格子点のカオス評価値
から最小値・最大値を求めます。

## 実行方法

NumPyとMatplotlibをインストールしたPython環境で、PowerShellから次を実行します。

```powershell
cd "C:\Users\iamso\OneDrive - 国立大学法人東海国立大学機構\講義\シミュレーションサイエンス\1022\tripple_pendulumn_sim"
python plot_chaos_volume.py
```

単一振り子のリアルタイム表示:

```powershell
py -3.10 realtime_triple_pendulum.py
```

初期角度をラジアンで指定できます。

```powershell
py -3.10 realtime_triple_pendulum.py --angles 1.2 1.8 2.4 --dt 0.002
```

カオス評価の摂動軌道を少し先まで進める場合:

```powershell
py -3.10 realtime_triple_pendulum.py --dt 0.002 --chaos-horizon-steps 3
```

Newton反復エラーが出る場合は、まず `--dt 0.001` のように時間刻みを
小さくしてください。内部でも局所的なステップ分割を行いますが、極端な
初期条件や長時間計算ではそれでも収束しない場合があります。
`--steps-per-frame` は1フレームで進める計算ステップ数
であり、表示速度と計算負荷を調整します。画面上の `Pause / Resume` と
`Reset` も利用できます。

### `realtime_triple_pendulum2.py`

`realtime_triple_pendulum.py` の拡張版です。終端質点の軌跡を左側の物理空間に
描画し、古い軌跡は時間経過で消えます。

```powershell
py -3.10 realtime_triple_pendulum2.py
```

実行中の操作:

- `Faster` または右矢印: シミュレーション速度を2倍
- `Slower` または左矢印: シミュレーション速度を半分
- `Pause / Resume` またはスペース: 一時停止/再開
- `Clear trace`: 軌跡を消去
- `Reset`: 初期状態へ戻す

軌跡の設定:

```powershell
py -3.10 realtime_triple_pendulum2.py `
  --fade-start-dt 10 `
  --fade-time 20
```

これはシミュレーション時刻10秒までは軌跡を完全に表示し、その後20秒かけて
徐々に透明にし、合計30秒後に削除します。`fade-start-dt` と `fade-time` は
シミュレーション時間の単位で、画面のフレーム時間ではありません。

### `plot_csv_volume.py`

シミュレーションを再実行せず、保存済みCSVから同じ3Dカオス散布図を作ります。
引数なしで起動するとCSV選択ダイアログが開きます。

```powershell
py -3.10 plot_csv_volume.py
```

特定のCSVを直接指定することもできます。

```powershell
py -3.10 plot_csv_volume.py triple_pendulum_volume.csv
```

デフォルトでは `triple_pendulum_chaos_from_csv.png` に保存してから表示します。
保存先を指定する場合:

```powershell
py -3.10 plot_csv_volume.py triple_pendulum_volume.csv `
  --output my_chaos_plot.png
```

画像を保存せず表示だけ行う場合は `--output none` を指定します。

### `plot_csv_volume2.py`

保存済みCSVを、点ではなく3Dボクセルとして表示します。各ボクセルは
初期角度空間の1点に対応します。カオス値は色で表し、透明度は通常の
散布図とは逆に設定しています。

- カオス値が小さい: 不透明
- カオス値が大きい: 透明

ファイル選択ダイアログを使う場合:

```powershell
py -3.10 plot_csv_volume2.py
```

CSVを直接指定する場合:

```powershell
py -3.10 plot_csv_volume2.py triple_pendulum_volume.csv
```

PNGを保存せず表示だけ行う場合:

```powershell
py -3.10 plot_csv_volume2.py triple_pendulum_volume.csv --output none
```

境界点は6近傍のカオス値を持たないため、ボクセルとして表示しません。
デフォルトでは、正規化カオス値が `0.75` 以上のボクセルを完全に非表示に
します。閾値を変更する場合:

```powershell
py -3.10 plot_csv_volume2.py triple_pendulum_volume.csv `
  --chaos-threshold 0.60
```

閾値を大きくすると、より多くのボクセルが表示されます。すべて表示する場合は
`--chaos-threshold 1.0` を指定してください。閾値は生の `chaos_value` ではなく、
領域内の最小値・最大値を使って0から1へ正規化した値です。
起動時には、復元した格子サイズ、有効なボクセル数、カオス値の範囲も
ターミナルに表示します。CSVに `i,j,k` 格子インデックスが含まれる場合は、
それを使ってボクセル位置を復元します。
内部の箱も確認できるように、`plot_csv_volume2.py` ではMatplotlibの
`voxels()` が省略する隣接ボクセル間の内部面も含め、各ボクセルを個別の
立方体として描画しています。ただし、3D表示では視線方向に外側の面が
重なるため、すべての内部面が同時に見えるわけではありません。回転すると
内部構造を確認できます。

### HTML/WebGLボクセルビューア

Matplotlibより高速に回転・拡大縮小したい場合は、ブラウザのWebGLを使う
ビューアを使用できます。関連ファイルは[`web_voxel_viewer/`](<./web_voxel_viewer/>)に
まとめています。CSVを埋め込んだHTMLを作成し、既定ではブラウザで開きます。

```powershell
py -3.10 web_voxel_viewer\create_voxel_viewer.py `
  triple_pendulum_volume.csv `
  --output triple_pendulum_voxels.html
```

HTMLを開いた後は、ドラッグで回転、ホイールで拡大縮小、右ドラッグで移動できます。
閾値と透明度は画面上のスライダーで変更できます。また、HTMLの`Open CSV`から
別のCSVを読み込めます。描画にはThree.jsの`InstancedMesh`を使うため、各ボクセルを
Matplotlibの個別ポリゴンとして作る方式より大量データに適しています。
このビューアはThree.jsをCDNから読み込むため、初回表示時にはインターネット接続が
必要です。

以前に生成したHTMLには修正が反映されないため、ビューアを更新した場合はHTMLを
再生成してください。ブラウザが`file:` URLのモジュールを制限する場合は、生成した
HTMLのあるフォルダで簡易HTTPサーバーを起動します。

```powershell
py -3.10 -m http.server 8000
```

その後、ブラウザで`http://localhost:8000/triple_pendulum_voxels.html`を開きます。

### 図の読み方

CSVから作る3D図では、各軸は初期角度です。

- x軸: `theta1_initial_rad`、第1リンクの初期角度
- y軸: `theta2_initial_rad`、第2リンクの初期角度
- z軸: `theta3_initial_rad`、第3リンクの初期角度

各点は1つの初期角度の組
`(theta1_initial_rad, theta2_initial_rad, theta3_initial_rad)` を表します。
色と透明度は `chaos_value` を表します。この値は、その初期条件から計算した
終了時の角度と、6方向に微小に異なる初期条件から計算した終了時の角度との差
の平均です。

- 小さい値: 近傍の初期条件でも終了状態が似ている
- 大きい値: 小さな初期角度の差が終了状態の大きな差になっている

したがって、大きい値ほど有限時間の意味でカオスへの感度が強いと解釈します。
これは厳密なリアプノフ指数ではありません。カラーバーは正規化前の
`chaos_value` の実際の値を表示し、透明度はその値を領域内の最小値・最大値で
0.08から1.0へ正規化したものです。

初期設定は `9^3` 点、`t_max=8`、`dt=0.04` です。最初の動作確認では
`build_chaos_volume(grid_size=5, t_max=2.0, dt=0.05, workers=2)` のように
小さくしてください。Windowsのマルチプロセス起動に対応するため、実行部は
`if __name__ == "__main__":` で保護しています。
関数を別のスクリプトから呼ぶ場合、`workers=1` にすると逐次実行になり、
対話的なPython入力やノートブックでも利用できます。

コマンドライン引数で計算量を調整できます。

```powershell
py -3.10 plot_chaos_volume.py --grid-size 5 --t-max 3 --dt 0.03 --workers 4
```

各初期条件の終了状態はデフォルトで
`triple_pendulum_checkpoint.npz` に逐次保存されます。同じパラメータで
再実行すると、完了済みの点を再計算せずに続きから実行します。
計算が長時間止まって見える場合は、`Ctrl+C`で中断しても、最後に完了した
ジョブまでのチェックポイントを残せます。同じコマンドを再実行してください。
グリッドサイズ、角度範囲、`t_max`、`dt`はチェックポイント作成時と一致している
必要があります。チェックポイントは一時ファイルから原子的に置き換えるため、
保存中に中断しても直前の完全なファイルを維持します。
一時ファイルが残っている場合は、再開時に完了件数が多い方の新しい
チェックポイントを自動的に選びます。
別の条件で計算するときは別のチェックポイント名を指定してください。

```powershell
py -3.10 plot_chaos_volume.py --grid-size 7 --t-max 5 --dt 0.02 `
  --checkpoint triple_pendulum_grid7.npz
```

1件が長時間かかりすぎる場合は、ジョブ単位の制限時間を指定できます。
制限に達した初期条件は `triple_pendulum_failures.json` に記録されます。
未完了点が残る場合は全体の描画を行わず、制限値を調整して同じチェック
ポイントで再開してください。

```powershell
py -3.10 plot_chaos_volume.py --max-job-seconds 60
```

特定の初期条件で処理が長時間かかる場合は、`--max-job-seconds`を指定すると
そのジョブを失敗として記録して次へ進みます。計算全体が停止したように見える
場合は、例えば次のように再開してください。

```powershell
py -3.10 plot_chaos_volume.py --grid-size 64 --t-max 3 --dt 0.03 `
  --workers 8 --max-job-seconds 120 --no-plot
```

## CSV出力

### GPU版

### Native CUDA runtime

[`gpu runtime/`](<./gpu%20runtime/>) contains a native C++/CUDA implementation.
Unlike the CuPy version, each CUDA thread keeps one complete pendulum
trajectory on the GPU, and both simulation and chaos evaluation run in CUDA
kernels. The host only transfers the initial-angle array before execution and
the final arrays after execution.

Install the NVIDIA CUDA Toolkit, ensure `nvcc` is available, and build it with:

```powershell
cd "gpu runtime"
cmake -S . -B build -G "NMake Makefiles"
cmake --build build --config Release
```

Run it with:

```powershell
.\build\triple_pendulum_cuda.exe `
  --grid-size 32 --t-max 3 --dt 0.03 `
  --csv ..\data\triple_pendulum_volume_cuda.csv
```

The generated CSV is compatible with the existing viewers. This environment
does not currently have `nvcc`, so compilation must be verified on the CUDA
machine.

`plot_chaos_volume_gpu.py` は、CuPyで多数の初期条件をGPU上で同時に計算する
実験的なCUDA版です。Gonzalez型離散勾配法を保ったまま、状態・質量行列・
Newton反復をバッチ化します。RTX 5070 TiではCUDA対応のNVIDIAドライバと
CuPyのCUDA 12系wheelが必要です。

```powershell
py -3.10 -m pip install cupy-cuda12x
py -3.10 plot_chaos_volume_gpu.py `
  --grid-size 9 --t-max 8 --dt 0.02 --batch-size 1024
```

まず小さい実行でGPUが認識されることを確認してください。

```powershell
py -3.10 plot_chaos_volume_gpu.py `
  --grid-size 5 --t-max 0.5 --dt 0.03 --batch-size 128
```

`--batch-size` はGPUメモリ使用量を調整します。CUDAメモリ不足なら小さくし、
計算が細かすぎる場合は大きくします。結果は通常のCSVと同じ列構成で
`triple_pendulum_volume_gpu.csv` に保存されます。GPU版は現時点では通常版の
チェックポイント・ジョブ単位timeoutは使わず、バッチ単位で進捗を表示します。
`--plot` を付けると計算後に通常の散布図も表示します。

GPUとCPUの通信を減らすため、各バッチの入力角度はバッチ開始時に一度だけ
GPUへ転送し、終了角度はバッチ終了時に一度だけCPUへ戻します。Newton反復中の
収束マスクと固定状態のハミルトニアンはGPU上に保持し、反復ごとのCPU同期を
避けています。十分なGPUメモリがある場合は、まず`--batch-size 2048`または
`4096`を試してください。バッチを大きくすると転送回数は減りますが、Newtonの
一部の難しい初期条件がバッチ全体を遅くする可能性があります。

### HTMLリアルタイムビューア

[`web_realtime_viewer/`](<./web_realtime_viewer/>)に、リアルタイム振り子ビューアの
ブラウザ版があります。Three.jsで描画するため、Tkinter/Matplotlib版より軽く、
速度変更、再生/一時停止、リセット、トレース消去、トレースのフェードを操作できます。

[`triple_pendulum_realtime.html`](<./web_realtime_viewer/triple_pendulum_realtime.html>)を
ブラウザで開いてください。初回表示にはThree.js CDNへの接続が必要です。入力欄で
初期角度、`dt`、フェード開始時間、フェード時間を変更できます。Spaceキーで一時停止、
左右矢印キーで速度変更もできます。

### 統合HTMLビューア

ボクセル表示とリアルタイム振り子を同時に使う場合は、
[`web_combined_viewer/`](<./web_combined_viewer/>)を使用します。左側の可視ボクセルを
クリックすると、その初期角度が右側の振り子へ設定されます。

```powershell
py -3.10 web_combined_viewer\create_combined_viewer.py `
  triple_pendulum_volume.csv `
  --output triple_pendulum_combined.html
```

左側ではカオス閾値・透明度・カメラを操作でき、右側では速度、`dt`、トレースの
フェード条件を操作できます。初回表示にはThree.js CDNへの接続が必要です。

GPU版のNewton反復は、バッチ内の一部の初期条件だけが収束しない場合に、
その軌道だけ時間刻みを自動的に半分へ分割して再試行します。そのため、
Newtonの収束性は `--batch-size` では本質的に変わらず、値を小さくしても
同じ初期条件で失敗することがあります。8段階の分割後も収束しない場合は、
`--dt` を小さくしてください。

`ImportError: DLL load failed while importing cublas` が出る場合は、CuPyの
CUDA wheelが混在しているか、CUDA runtimeとwheelの組み合わせが一致していません。
同じPython環境でCuPyを入れ直してから、ターミナルを再起動してください。

```powershell
py -3.10 -m pip uninstall -y cupy cupy-cuda11x cupy-cuda12x
py -3.10 -m pip install --no-cache-dir cupy-cuda12x
```

計算結果はデフォルトで `triple_pendulum_volume.csv` に保存されます。
PNGを開かず、CSVだけを作る場合は次のように実行します。

```powershell
py -3.10 plot_chaos_volume.py `
  --grid-size 5 `
  --t-max 3 `
  --dt 0.03 `
  --workers 4 `
  --csv triple_pendulum_volume.csv `
  --no-plot
```

CSVは初期角度空間の全格子点について、次の列を持ちます。

- `i`, `j`, `k`: 格子インデックス
- `theta1_initial_rad`, `theta2_initial_rad`, `theta3_initial_rad`: 初期角度
- `theta1_final_rad`, `theta2_final_rad`, `theta3_final_rad`: 終了時の角度
- `chaos_value`: 6近傍との有限時間角度差
- `chaos_value_normalized`: 領域内のカオス値を0から1へ正規化した値
- `alpha`: PNG描画で使う透明度
- `is_interior`: 6近傍が揃う内部点なら1、境界点なら0

角度はすべてラジアンです。境界点は6近傍がないため、
`chaos_value`、`chaos_value_normalized`、`alpha` が空欄になります。
CSVはUTF-8 BOM付きで保存するため、Excelでも開きやすくなっています。

Pythonでは例えば次のように読み込めます。

```python
import pandas as pd

data = pd.read_csv("triple_pendulum_volume.csv")
interior = data[data["is_interior"] == 1]
```

実行中はターミナルに、初期条件の準備、シミュレーションの完了数・割合・
経過時間・推定残り時間、6近傍カオス量の計算、画像保存の各段階を表示します。
シミュレーション中の表示がしばらく進まない場合でも、Newton反復で1件を
計算中の可能性があります。並列実行では10秒ごとに `Still working` の
heartbeatも表示し、完了数、未完了数、待機中のタスク番号を確認できます。
各ジョブの完了時には、そのジョブの計算時間も表示します。完了数が増え続ける、
またはheartbeatが出続けていれば、ターミナルが固まったとは限りません。
ワーカープロセスの例外は、該当するタスク番号と初期角度を付けて表示します。

単一シミュレーションの例:

```python
from triple_pendulum import TriplePendulumSimulator

simulator = TriplePendulumSimulator()
result = simulator.simulate(
    angles=(1.2, 1.8, 2.4),
    t_max=5.0,
    dt=0.02,
)
print(result.energy_error[-1])
print(simulator.finite_time_sensitivity((1.2, 1.8, 2.4), t_max=5.0))
```

なお、3次元でプロットする座標は物理空間の `(x,y,z)` ではなく、
初期角度空間の `(theta1(0), theta2(0), theta3(0))` です。物理的な三次元
振り子は自由度と拘束条件が増えるため、別のモデルとして拡張する必要があります。
