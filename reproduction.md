# Reproduction

Following a review of the training data and pipeline, a potential risk of data contamination was noticed. Since its impact on the original checkpoint could not be conclusively established, we conducted an additional training run in Aug 2026 using data that had been re-checked for overlap with the evaluation benchmarks.

The retrained checkpoint achieves results broadly comparable to those of the original checkpoint, with modest variations in both directions that fall within the expected range of run-to-run variability. To be transparent about this uncertainty and to support community research and benchmarking, we have released the [training code](training/) and the resulting [vggt_omega_1b_416_reproduce.pt](https://huggingface.co/facebook/VGGT-Omega/blob/main/vggt_omega_1b_416_reproduce.pt). Although the performance is quite similar, please use this checkpoint instead of the original checkpoint or its derivatives for benchmarking.

The results reported below are based on the retrained checkpoint, which supersedes the original checkpoint as the reference for future comparisons on these benchmarks. The remainder of this page presents the evaluation results and describes the training setup. It is worth noting, however, that the retrained checkpoint was produced on an accelerated schedule for the purpose of quickly investigating and reproducing the benchmark results, using a training setup that involved several compromises relative to the original recipe, as detailed below. While its benchmark performance is broadly comparable to that of the original checkpoint, it does not consistently match the original checkpoint’s qualitative performance on in-the-wild videos.



For installation, data preparation, launch commands, and configuration details, see the [training README](training/README.md).



## Training details

To complete the training within the available budget, the retraining setup differed from the original pipeline in the following ways:

1. **Compute and distributed setup.** We used DDP and approximately 30–40 images per GPU per optimizer step (around 35–50% of the original per-GPU image count), and trained on 256 GPUs instead of the 128 GPUs used in the paper. The resulting global batch size was comparable to or smaller than the original one, while training was approximately 1.8× faster.
2. **Input, augmentation, and frame schedule.** To reduce training time, we fixed the per-image pixel budget at 416 × 416 pixels (with the actual image dimensions determined by the aspect ratio) rather than randomly sampling the pixel budget between 416 × 416 and 512 × 512 pixels. We used a lighter augmentation policy to accelerate convergence. We first trained on sequences of 2–16 frames before fine-tuning on sequences of 2–32 frames.
3. **Initialization and training stages.** We initialized the aggregator from the public VGGT checkpoint for faster convergence and skipped the self-supervised training stage because, as reported in the paper, it does not affect the benchmark results considered here.


Therefore, when using the retrained checkpoint, preprocess the input images with `image_resolution=416` rather than `image_resolution=512`:

```python
images = load_and_preprocess_images(image_names, image_resolution=416).to("cuda")
```




## Results

Evaluations under three evaluation protocols (our primary benchmark suite, the MapAnything ETH3D protocol, and SpatialBench) indicate that the overall performance level of the public VGGT-Omega checkpoint is reproducible, with the additional training run yielding broadly comparable results across the evaluated benchmarks.

### Primary benchmarks

<!-- Each row has its own tbody to disable GitHub's alternating row background. -->

<table>
  <thead>
    <tr>
      <th scope="col">Dataset</th>
      <th scope="col">Metric</th>
      <th scope="col" align="right">VGGT</th>
      <th scope="col" align="right">PI3</th>
      <th scope="col" align="right">DA3</th>
      <th scope="col" align="right">Public VGGT-Omega</th>
      <th scope="col" align="right">Retrained VGGT-Omega</th>
    </tr>
  </thead>

  <tbody>
    <tr><th colspan="7" align="center">Camera estimation</th></tr>
  </tbody>
  <tbody>
    <tr><th scope="row">7 Scenes</th><td>AUC@3° ↑</td><td align="right">10.9</td><td align="right">13.3</td><td align="right">18.7</td><td align="right"><strong>29.6</strong></td><td align="right">27.4</td></tr>
  </tbody>
  <tbody>
    <tr><th scope="row">7 Scenes</th><td>AUC@30° ↑</td><td align="right">74.4</td><td align="right">77.0</td><td align="right">78.2</td><td align="right"><strong>83.1</strong></td><td align="right">82.4</td></tr>
  </tbody>
  <tbody>
    <tr><th scope="row">NRGBD</th><td>AUC@3° ↑</td><td align="right">81.7</td><td align="right">83.8</td><td align="right">86.4</td><td align="right">89.7</td><td align="right"><strong>90.1</strong></td></tr>
  </tbody>
  <tbody>
    <tr><th scope="row">NRGBD</th><td>AUC@30° ↑</td><td align="right">97.7</td><td align="right">98.2</td><td align="right">98.4</td><td align="right">98.8</td><td align="right"><strong>98.9</strong></td></tr>
  </tbody>
  <tbody>
    <tr><th scope="row">ETH3D</th><td>AUC@3° ↑</td><td align="right">18.8</td><td align="right">35.3</td><td align="right">46.1</td><td align="right">49.8</td><td align="right"><strong>51.5</strong></td></tr>
  </tbody>
  <tbody>
    <tr><th scope="row">ETH3D</th><td>AUC@30° ↑</td><td align="right">62.1</td><td align="right">79.6</td><td align="right">87.0</td><td align="right">88.5</td><td align="right"><strong>88.9</strong></td></tr>
  </tbody>
  <tbody>
    <tr><th scope="row">DyCheck</th><td>AUC@3° ↑</td><td align="right">21.0</td><td align="right">23.3</td><td align="right">32.1</td><td align="right"><strong>38.4</strong></td><td align="right">35.9</td></tr>
  </tbody>
  <tbody>
    <tr><th scope="row">DyCheck</th><td>AUC@30° ↑</td><td align="right">78.7</td><td align="right">81.0</td><td align="right">83.9</td><td align="right"><strong>87.3</strong></td><td align="right">86.5</td></tr>
  </tbody>
  <tbody>
    <tr><th scope="row">Sintel</th><td>AUC@3° ↑</td><td align="right">15.0</td><td align="right">14.8</td><td align="right">16.2</td><td align="right">35.3</td><td align="right"><strong>35.5</strong></td></tr>
  </tbody>
  <tbody>
    <tr><th scope="row">Sintel</th><td>AUC@30° ↑</td><td align="right">50.0</td><td align="right">53.5</td><td align="right">52.7</td><td align="right">73.0</td><td align="right"><strong>73.4</strong></td></tr>
  </tbody>
  <tbody>
    <tr><th scope="row">TUM-Dynamic</th><td>AUC@3° ↑</td><td align="right">16.6</td><td align="right">16.1</td><td align="right">20.8</td><td align="right"><strong>30.2</strong></td><td align="right">30.0</td></tr>
  </tbody>
  <tbody>
    <tr><th scope="row">TUM-Dynamic</th><td>AUC@30° ↑</td><td align="right">61.2</td><td align="right">59.2</td><td align="right">62.7</td><td align="right">82.3</td><td align="right"><strong>82.8</strong></td></tr>
  </tbody>
  <tbody>
    <tr><th colspan="7" align="center">Depth estimation</th></tr>
  </tbody>
  <tbody>
    <tr><th scope="row">7 Scenes</th><td>δ₁.₂₅ ↑</td><td align="right">91.9</td><td align="right">92.8</td><td align="right">93.0</td><td align="right"><strong>94.6</strong></td><td align="right">93.7</td></tr>
  </tbody>
  <tbody>
    <tr><th scope="row">7 Scenes</th><td>AbsRel ↓</td><td align="right">0.073</td><td align="right">0.068</td><td align="right">0.063</td><td align="right"><strong>0.058</strong></td><td align="right">0.060</td></tr>
  </tbody>
  <tbody>
    <tr><th scope="row">NRGBD</th><td>δ₁.₂₅ ↑</td><td align="right">99.1</td><td align="right">99.2</td><td align="right">99.5</td><td align="right"><strong>99.6</strong></td><td align="right"><strong>99.6</strong></td></tr>
  </tbody>
  <tbody>
    <tr><th scope="row">NRGBD</th><td>AbsRel ↓</td><td align="right">0.019</td><td align="right">0.011</td><td align="right">0.010</td><td align="right"><strong>0.010</strong></td><td align="right">0.011</td></tr>
  </tbody>
  <tbody>
    <tr><th scope="row">ETH3D</th><td>δ₁.₂₅ ↑</td><td align="right">97.4</td><td align="right">99.6</td><td align="right">99.6</td><td align="right"><strong>99.8</strong></td><td align="right"><strong>99.8</strong></td></tr>
  </tbody>
  <tbody>
    <tr><th scope="row">ETH3D</th><td>AbsRel ↓</td><td align="right">0.036</td><td align="right">0.016</td><td align="right">0.015</td><td align="right"><strong>0.012</strong></td><td align="right"><strong>0.012</strong></td></tr>
  </tbody>
  <tbody>
    <tr><th scope="row">DyCheck</th><td>δ₁.₂₅ ↑</td><td align="right">95.2</td><td align="right">97.4</td><td align="right">97.7</td><td align="right"><strong>98.4</strong></td><td align="right">97.9</td></tr>
  </tbody>
  <tbody>
    <tr><th scope="row">DyCheck</th><td>AbsRel ↓</td><td align="right">0.055</td><td align="right">0.041</td><td align="right">0.039</td><td align="right"><strong>0.038</strong></td><td align="right">0.039</td></tr>
  </tbody>
  <tbody>
    <tr><th scope="row">Sintel</th><td>δ₁.₂₅ ↑</td><td align="right">79.2</td><td align="right">82.5</td><td align="right">86.1</td><td align="right">89.5</td><td align="right"><strong>89.9</strong></td></tr>
  </tbody>
  <tbody>
    <tr><th scope="row">Sintel</th><td>AbsRel ↓</td><td align="right">0.189</td><td align="right">0.144</td><td align="right">0.118</td><td align="right">0.097</td><td align="right"><strong>0.094</strong></td></tr>
  </tbody>
  <tbody>
    <tr><th scope="row">TUM-Dynamic</th><td>δ₁.₂₅ ↑</td><td align="right">92.2</td><td align="right">95.5</td><td align="right">94.3</td><td align="right">97.4</td><td align="right"><strong>97.7</strong></td></tr>
  </tbody>
  <tbody>
    <tr><th scope="row">TUM-Dynamic</th><td>AbsRel ↓</td><td align="right">0.064</td><td align="right">0.046</td><td align="right">0.049</td><td align="right">0.041</td><td align="right"><strong>0.040</strong></td></tr>
  </tbody>
</table>

### MapAnything Dense-N-View on ETH3D

Following the MapAnything ETH3D protocol, the metrics below are averaged over evaluations using 2–100 input views.

<table>
  <thead>
    <tr>
      <th scope="col">Method</th>
      <th scope="col" align="right">AUC ↑</th>
      <th scope="col" align="right">ATE ↓</th>
      <th scope="col" align="right">Point Abs ↓</th>
      <th scope="col" align="right">Depth Abs ↓</th>
    </tr>
  </thead>

  <tbody>
    <tr><th scope="row">DA3</th><td align="right">72.362358</td><td align="right">0.018893113</td><td align="right">0.044251421</td><td align="right">0.030867289</td></tr>
  </tbody>
  <tbody>
    <tr><th scope="row">Public VGGT-Omega</th><td align="right">78.661116</td><td align="right"><strong>0.008457166</strong></td><td align="right"><strong>0.025827513</strong></td><td align="right"><strong>0.016451749</strong></td></tr>
  </tbody>
  <tbody>
    <tr><th scope="row">Retrained VGGT-Omega</th><td align="right"><strong>79.533665</strong></td><td align="right">0.009985055</td><td align="right">0.026264634</td><td align="right">0.020428510</td></tr>
  </tbody>
</table>

### SpatialBench

<table>
  <thead>
    <tr>
      <th scope="col">Split</th>
      <th scope="col">Method</th>
      <th scope="col" align="right">Depth<br><sub>RMSE ↓</sub></th>
      <th scope="col" align="right">Depth<br><sub>AbsRel ↓</sub></th>
      <th scope="col" align="right">Depth<br><sub>Inlier@1.03 ↑</sub></th>
      <th scope="col" align="right">Camera<br><sub>AUC@3° ↑</sub></th>
      <th scope="col" align="right">Camera<br><sub>AUC@30° ↑</sub></th>
    </tr>
  </thead>

  <tbody>
    <tr><th scope="row">Sparse</th><th scope="row">Public VGGT-Omega</th><td align="right">1.088388</td><td align="right"><strong>0.077394</strong></td><td align="right"><strong>0.559211</strong></td><td align="right"><strong>0.410944</strong></td><td align="right"><strong>0.802249</strong></td></tr>
  </tbody>
  <tbody>
    <tr><th scope="row">Sparse</th><th scope="row">Retrained VGGT-Omega</th><td align="right"><strong>0.666318</strong></td><td align="right">0.106014</td><td align="right">0.551625</td><td align="right">0.402408</td><td align="right">0.789016</td></tr>
  </tbody>
  <tbody>
    <tr><th scope="row">Medium</th><th scope="row">Public VGGT-Omega</th><td align="right">1.049209</td><td align="right"><strong>0.067020</strong></td><td align="right">0.590147</td><td align="right">0.416412</td><td align="right">0.794429</td></tr>
  </tbody>
  <tbody>
    <tr><th scope="row">Medium</th><th scope="row">Retrained VGGT-Omega</th><td align="right"><strong>0.620659</strong></td><td align="right">0.094004</td><td align="right"><strong>0.608652</strong></td><td align="right"><strong>0.463955</strong></td><td align="right"><strong>0.802315</strong></td></tr>
  </tbody>
</table>
