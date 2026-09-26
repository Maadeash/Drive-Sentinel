# DriveSentinel-FPGA: Methodology
### *IEEE Paper Section — Edge-AI Industrial Drive Health Monitor on PYNQ-Z2*

---

## III. Methodology

### A. Dataset and Experimental Setup

We evaluate our system on the Paderborn University (KAt) bearing dataset [CITE], which provides multi-modal sensor recordings from a spectrally-loaded motor-test-rig. Each recording captures seven simultaneous signal streams: phase currents $I_1$, $I_2$ at 64 kHz; vibration $v_1$ at 64 kHz; mechanical force $F$, torque $T$, and shaft speed $\omega$ at 4 kHz; and bearing-module temperature $\theta$ at 1 Hz. The dataset encompasses 32 distinct physical bearings across four fault categories: *healthy* (K001–K006, six bearings), *outer-race fault* (KA01–KA30, twelve bearings), *inner-race fault* (KI01–KI21, eleven bearings), and *combined inner+outer fault* (KB23, KB24, KB27, three bearings).

**Combined-class limitation.** The combined-fault class contains only three distinct physical bearings, which is a fundamental dataset constraint that limits the statistical reliability of combined-class metrics. In each cross-validation fold (Section III-B), at most two KB bearings are available for training and one for testing. Results for this class are reported separately and are explicitly excluded from the macro-averaged accuracy metric used for model selection.

### B. Leave-Bearing-Out Cross-Validation

The central challenge of bearing fault classification is *bearing identity leakage*: if training and test windows originate from the same physical bearing, the model can exploit individual bearing fingerprints (mounting asymmetries, surface micro-geometry, thermal equilibria) rather than learning general fault signatures. This inflates reported accuracy while providing no generalization guarantee to unseen bearings in deployment.

To eliminate this leakage, we implement **stratified leave-bearing-out $k$-fold cross-validation**. For each class $c$, its $N_c$ bearing codes are randomly partitioned into $k$ groups of size $\lceil N_c / k \rceil$. In fold $f$, the test set consists of every recording from bearings assigned to group $f$, for all classes simultaneously. Consequently, zero recordings from any test-set bearing are available to the model during training.

Formally, let $\mathcal{B}^{(c)}$ denote the set of bearing codes for class $c$ and $G^{(c)}_f \subset \mathcal{B}^{(c)}$ the test group for class $c$ in fold $f$. We enforce:

$$\mathcal{B}^{(c)}_{\text{train},f} = \mathcal{B}^{(c)} \setminus G^{(c)}_f, \quad \mathcal{B}^{(c)}_{\text{test},f} = G^{(c)}_f, \quad \forall c$$

$$\mathcal{B}^{(c)}_{\text{train},f} \cap \mathcal{B}^{(c)}_{\text{test},f} = \emptyset, \quad \forall c, f \quad \text{(verified by assertion)}$$

We set $k = 5$, yielding approximately 1–2 healthy, 2–3 outer-race, and 2 inner-race bearings per test fold. For the combined class, the three bearings produce at most three effective folds containing combined-class test data; the remaining folds use all three KB bearings for training. The reported metric is **mean ± standard deviation** of per-fold accuracy across all $k$ folds. This is the only number reported for model comparison.

### C. Feature Engineering

Each 0.25-second window of a high-rate channel (64 kHz or 4 kHz) yields 17 features: eight time-domain statistics (mean, standard deviation, RMS, peak, crest factor, kurtosis, skewness, peak-to-peak) and nine frequency-domain statistics (energies in eight equal-width spectral bands across $[0, f_s/2]$ plus dominant frequency). The 1 Hz temperature channel is summarized per file as a mean–standard deviation pair. Across six windowed channels, this produces $6 \times 17 + 2 = 104$ raw features per window.

To assess whether raw absolute features encode bearing identity rather than fault physics, we define three normalization variants:

**Raw:** Absolute feature values, as described above. These preserve amplitude and DC offset information, which may be driven by bearing-specific installation conditions rather than fault signatures.

**Norm:** Per-file z-score normalization. For each feature $x$ and source recording $r$:

$$x_{\text{norm}} = \frac{x - \mu_r}{\sigma_r + \epsilon}$$

where $\mu_r$ and $\sigma_r$ are the mean and standard deviation of that feature across all windows of recording $r$, and $\epsilon = 10^{-9}$ prevents division by zero. This removes recording-level DC offsets and amplitude scaling differences between bearings.

**Rank:** Within-file percentile rank, $x_{\text{rank}} = \hat{F}_r(x) \in [0, 1]$, where $\hat{F}_r$ is the empirical CDF of the feature within recording $r$. This strips absolute scale entirely, retaining only the relative shape of the within-recording distribution.

**Norm+Rank:** Concatenation of norm and rank variants ($2 \times 104 = 208$ features), allowing the classifier to exploit both relative distribution shape and z-score deviations jointly.

All four variants are evaluated on the identical folds, enabling controlled comparison.

### D. Classifier Comparison

We evaluate four classifiers spanning the parameter-count and inference-cost spectrum relevant to FPGA deployment:

1. **Random Forest (RF):** 300 trees, unlimited depth, majority-vote aggregation. Inference cost scales with tree count × average tree depth, not feature count.

2. **Logistic Regression (LR):** L2 regularization, hyperparameter $C$ selected by inner 3-fold CV within each training split. Linear decision boundaries; $n_\text{classes} \times n_\text{features}$ parameters, $n_\text{classes} \times n_\text{features}$ multiply-accumulate operations (MACs) per inference.

3. **Radial Basis Function SVM (RBF-SVM):** Hyperparameters $C$ and $\gamma$ tuned by inner 3-fold CV. Inference cost scales with support vector count, which can be large and is difficult to bound a priori.

4. **Multi-Layer Perceptron (MLP):** Two fully connected hidden layers of width 64 and 32 neurons (ReLU activations), yielding architecture $n_\text{feat} \rightarrow 64 \rightarrow 32 \rightarrow 4$. Total inference cost: $64 n_\text{feat} + 64{\cdot}32 + 32{\cdot}4$ MACs. This architecture is the primary FPGA target due to its regular dataflow, fixed topology, and suitability for fixed-point arithmetic.

For each of the $4 \times 4 = 16$ model–feature combinations, we report leave-bearing-out mean ± std accuracy, per-class F1 scores, summed confusion matrices across all folds, and estimated parameter count and inference MACs.

### E. Model Selection Criteria

The primary selection criterion is leave-bearing-out generalization accuracy (mean across folds). In cases of near-equal accuracy, preference is given to lower FPGA inference cost: specifically, the MLP is preferred over RF or SVM at equivalent accuracy because its fixed matrix-multiply dataflow maps efficiently onto DSP slices in Zynq-7020 (220 slices, 53K LUTs), and its parameter count is bounded and predictable. LR is preferred over MLP if accuracy is within 1% (lower cost, fully linear). SVM is deprioritized due to unbounded support-vector count.

### F. INT8 Quantization for FPGA Deployment

The selected MLP is retrained on all 32 bearings (no holdout, since deployment targets previously-unseen bearings in the field), then quantized to 8-bit fixed-point using symmetric per-tensor quantization. For each layer $l$ with float weights $\mathbf{W}_l \in \mathbb{R}^{n_\text{out} \times n_\text{in}}$ and calibrated activation statistics:

$$s_W^{(l)} = \frac{\max|\mathbf{W}_l|}{127}, \quad \mathbf{W}^{(l)}_{\text{int8}} = \text{clamp}\!\left(\left\lfloor\frac{\mathbf{W}_l}{s_W^{(l)}}\right\rceil,\, -128,\, 127\right)$$

$$s_x^{(l)} = \frac{\max|\mathbf{x}^{(l)}|}{127}, \quad \mathbf{x}^{(l)}_{\text{int8}} = \text{clamp}\!\left(\left\lfloor\frac{\mathbf{x}^{(l)}}{s_x^{(l)}}\right\rceil,\, -128,\, 127\right)$$

where $\lfloor\cdot\rceil$ denotes rounding to the nearest integer. The matrix–vector product is accumulated in 32-bit integers to prevent overflow:

$$\mathbf{y}^{(l)}_{\text{int32}} = \mathbf{W}^{(l)}_{\text{int8}} \cdot \mathbf{x}^{(l)}_{\text{int8}} + \mathbf{b}^{(l)}_{\text{int32}}$$

where the bias is pre-quantized to the accumulator scale: $\mathbf{b}^{(l)}_{\text{int32}} = \lfloor\mathbf{b}_l / (s_W^{(l)} \cdot s_x^{(l)})\rceil$. The result is dequantized to float for the next layer's activation scale computation:

$$\mathbf{y}^{(l)}_{\text{float}} = s_W^{(l)} \cdot s_x^{(l)} \cdot \mathbf{y}^{(l)}_{\text{int32}}$$

The quantized weights are exported as signed 8-bit hexadecimal `.mem` files compatible with Verilog's `$readmemh` directive, one value per line, row-major order. Scale factors are exported as a JSON metadata file. A standalone NumPy golden reference function implementing the identical arithmetic is generated alongside the weight files and used to verify RTL implementations: the golden reference must reproduce the same argmax class prediction as the float model on ≥98% of the calibration dataset.

### G. Assumptions and Limitations

The following assumptions and limitations apply to this work and should be considered when interpreting results:

1. **Single-dataset evaluation.** All results are on the Paderborn/KAt dataset. Generalization to other bearing geometries, test rigs, or operating envelope ranges is not demonstrated.

2. **Combined-class scarcity.** With only three KB bearings, combined-fault classification results have high variance and limited statistical significance. A confident combined-fault classifier would require at least 10–15 additional combined-fault specimens.

3. **Operating-condition variation.** The dataset spans multiple speed, load, and force conditions (N09/N15, M01/M07, F04/F10). Our fold assignment stratifies by bearing code, not by operating condition, so a test fold may underrepresent some conditions. Future work should stratify by operating condition as well.

4. **Per-tensor quantization.** We apply symmetric per-tensor (not per-channel) INT8 quantization for simplicity of hardware implementation. Per-channel quantization would improve accuracy at the cost of additional scale-factor storage and a more complex hardware multiplier chain.

5. **Temperature channel.** The `temp_2_bearing_module` channel samples at 1 Hz, yielding approximately one sample per 0.25-second window. Its contribution is limited to a per-file mean and standard deviation, which cannot capture intra-window thermal dynamics.

---

*End of Section III.*

---

> **Note to author:** Insert this section verbatim after Section II (Related Work) and before Section IV (Hardware Architecture). Update `[CITE]` with the Paderborn/KAt dataset citation: Lessmeier et al., "Condition Monitoring of Bearing Damage in Electromechanical Drive Systems by Using Motor Current Signals of Electric Motors," *PHME*, 2016. Replace placeholder accuracy values in Section IV's results table with actual numbers from `results_kfold.md` after running `ds_v1_kfold_comparison.py`.
