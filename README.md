# PCEN-CNN 反芻聲紋辨識（PyTorch）
 
用 PyTorch 重寫我碩士論文裡的 PCEN × Baseline CNN 訓練流程，辨識乳牛採食（Eating）與反芻（Rumination）的聲音。下面的「原程式」指論文使用的 TensorFlow/Keras 版 Colab notebook，這個版本只用公開資料集，結果無法直接和論文結果比較。
 
## 結果
 
| 指標 | 平均 ± 標準差（10 個 seed） |
|---|---|
| Accuracy | 0.9995 ± 0.0010 |
| Macro-F1 | 0.9995 ± 0.0010 |
| Weighted-F1 | 0.9995 ± 0.0010 |
 
測試集共 427 個窗口（Eating 256、Rumination 171）。10 個 seed 裡 8 個是 1.0000，另外 2 個是 0.9977。
 
## 資料集
 
使用 [Martinez-Rau et al. (2023)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10632420/) 的公開資料集（[GitLab](https://gitlab.com/luciano.mrau/acoustic_dairy_cow_dataset)），用到兩個音檔：
 
- `JM_rumination.wav`：反芻
- `JM_grazing.wav`：採食

Eating 的樣本全部來自 JM_grazing，Rumination 的樣本全部來自 JM_rumination。
音檔附有逐次下顎運動的標注（`b`、`c`、`x` 是採食，`r` 是反芻）。原始音檔不放在這個 repo，notebook 執行時會自動下載，使用前請先看資料集頁面的授權條款。
 
與原程式的不同：
- 原程式另外使用自己蒐集的現場資料，這個版本只用公開資料集。
- 公開資料集的標注沒有 No-event，所以這裡只分 Eating 和 Rumination 兩類，原程式分三類。

## 流程
 
1. 切分：把錄音切成 5 秒不重疊的區塊，依類別分層隨機抽樣成 80/10/10（`random_state=42`）。同一區塊的窗口只會出現在同一個集合。
2. 視窗：區塊內以 1 秒窗、0.5 秒步長切片，用窗口中心點的標注決定標籤。
3. 特徵：Mel 頻譜（`n_fft=1024`、`hop=256`、64 個 Mel 頻帶、20–2000 Hz），後接 PCEN（`gain=0.25`、`bias=2`、`power=0.5`、`time_constant=0.4`）。特徵形狀是 64 × 63，標準化參數只用訓練集計算。
4. 增強：訓練集注入白噪音（SNR 10 ± 5 dB），驗證集和測試集不加。
5. 模型：`Resize(64×64) → BN → Conv32 → MaxPool → BN → Conv64 → MaxPool → BN → GlobalAvgPool → Dense128 → Dropout0.4 → Dense2`，共 28,168 個可訓練參數。
6. 訓練：Adam（學習率 1e-3、batch 64、最多 50 epoch）、class weight 設為 balanced、Dense(128) 加 L2(1e-4)、EarlyStopping（patience 5，還原最佳權重）、ReduceLROnPlateau（factor 0.5、patience 3）。
7. 評估：seed 0–9 各訓練一次，報告測試集指標的平均與標準差。

與原程式的不同：
- 切分和評估都只在這兩段錄音內進行，沒有跨牛隻、跨場域的評估。
- 噪音增強用白噪音，原程式的雜訊取自現場錄音。
- 只有訓練部分，不含 TFLite 量化與 ESP32 部署。

## 執行方式
 
```bash
pip install -r requirements.txt
jupyter notebook PCEN_CNN_PyTorch.ipynb
```
 
在 Google Colab 上直接上傳 notebook、從頭執行即可（需另外 `pip install librosa`）。執行後會產生：
 
- `pcen_features.npz`：標準化後的特徵
- `results_per_seed.csv`：每個 seed 的測試集指標
- `cnn_pcen_seed0.pt`：seed 0 的模型權重
