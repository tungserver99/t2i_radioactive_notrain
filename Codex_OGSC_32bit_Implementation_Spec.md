# CODEX IMPLEMENTATION SPEC — OGSC-32

**Tên phương pháp:** Oracle-Guided One-Step Semantic Control (OGSC-32)  
**Trạng thái:** Design/implementation specification — **chưa có kết quả GPU hoặc student transfer cho OGSC**  
**Mục tiêu:** Không train/động đến teacher weights; giữ nguyên **carrier, secret key và verifier `threshold_cfg_v1`** của `BEST_TEACHER_32bits_refine`; dùng một phép điều khiển latent thích nghi bằng **một VJP/backward duy nhất cho mỗi ảnh**; đánh giá **exact full-key recovery rate (FKR) theo nhóm ảnh**; tái sử dụng tối đa code oracle và weight-based hiện có.

> **Lệnh cho Codex:** Thực hiện bằng cách thêm module/runner/kiểm thử mới vào repo làm việc. **Không viết lại từ đầu** semantic carrier, VAE/UNet extractor, PNDM scheduler, scorer, prompt/seed/data pipeline, hoặc key metrics. **Không chạy GPU trên máy local**; chỉ viết code, test CPU/tiny model, tài liệu chạy cho server. Không tự ý chạy sweep, thêm trainable network, thay verifier hay tạo thêm phương pháp. Nếu thiếu một artifact/runtime script thì báo rõ blocker thay vì âm thầm thay thế.

---

## 0. ĐỌC CODE GỐC TRƯỚC KHI SỬA

Hai repository phải được coi là **hai nguồn độc lập**, có thể đặt cạnh nhau trong workspace:

- **Oracle / công cụ đường gradient:** `https://github.com/tungserver99/t2ifing_radioactive.git` — ưu tiên làm **working repo** để thêm `ogsc/`.
- **Bản weight-based 32-bit chính thức:** `https://github.com/ducthinhh1002/Malicious_quantization.git` — sử dụng `BEST_TEACHER_32bits_refine/` như một **read-only source** cho carrier, key, evaluation protocol, image generation và student pipeline cũ.

Đọc đúng các file sau trước khi code (đường dẫn tương đối theo từng repository):

| Nguồn | File | Tái sử dụng gì |
|---|---|---|
| Weight32 | `BEST_TEACHER_32bits_refine/harness/semantic_carrier.py` | **SemanticCarrier**, **SemanticExtractor**; `coordinates()` đúng phiên bản `threshold_cfg_v1` |
| Weight32 | `BEST_TEACHER_32bits_refine/harness/semantic_teacher.py` | `native_generate()`, `load_native()`, `extractor()`, `generate_set()`, `pixels()`, `score()`; prompt/seed/PNG protocol |
| Weight32 | `BEST_TEACHER_32bits_refine/harness/semantic_threshold.py` | `frozen_calibration()`, `CFG_SCALE=7.5`, `SHIFT_CAP=2.5`, `calibrated_shift_loss()` làm tham chiếu ngữ nghĩa |
| Weight32 | `BEST_TEACHER_32bits_refine/server21/files/var/lib/structure32_student12500/key_metrics.py` | **Dùng nguyên** `aggregate_metrics()`; BA/FKR group size 1/8/32/64/256 |
| Weight32 | `BEST_TEACHER_32bits_refine/server21/files/var/lib/structure32_student12500/campaign.py` | Thứ tự stages và cách dẫn `student12500` pipeline; **không chạy lại teacher training** |
| Weight32 | `BEST_TEACHER_32bits_refine/harness/semantic_report.py` | Scoring và paired teacher quality, giữ định nghĩa CLIP, LPIPS/FID/KID hiện có |
| Oracle | `Malicious_quantization/bitmark_lora/semantic_oracle_math.py` | **Tái sử dụng** `fake_quantize_rgb8_ste()`, `replay_controls()` (phần regression); không dùng per-image Adam |
| Oracle | `Malicious_quantization/bitmark_lora/semantic_oracle.py` | Tham khảo `FrozenOracleBackend.training_view()`: clean final latent → VAE → RGB8 STE → carrier |
| Oracle | `Malicious_quantization/bitmark_lora/semantic_trajectory.py` | Native PNDM trace/replay + kiểm tra precision, chỉ dùng làm control/QA, không chèn 4 residual vào OGSC runtime |
| Oracle | `Malicious_quantization/bitmark_lora/semantic_sign_train.py` | Tham khảo `decoded_image()` và `decoded_coordinates()` đã có gradient path/checkpointing |

**Hệ quả quan trọng:** Oracle repo `main` hiện có các nhánh RSSF sign (`semantic_sign_v1`, `semantic_sign_v2_trajectory`, 24/32 analytic detection). **Không lấy carrier/sign-verifier đó để thay phiên bản weight32**. Các hàm decoder/gradient/STE có thể tái sử dụng nhưng chủ thể verification phải là carrier/key/coordinates của **weight32**.

**Kiểm tra artifact cần thiết trước khi chạy:**

- `runs/structure32/carrier/{complete.json,key.json,pca.pt,stability.json,provenance.json}` (hoặc đúng artifact đã đóng băng từ run thành công).
- `runs/structure32/teacher_fp/calibration_target.json`, `result/threshold_teacher.json`, frozen clean calibration scores, clean teacher weights và prompt manifests.
- Các đường dẫn nêu trên có thể nằm trong `checkpoint_creation_documents/metadata/` hoặc `original_runs_metadata/`; **tuyệt đối không tự fit lại PCA/chọn lại key từ source khác**. Nếu thiếu bytes thật (ví dụ source code có metadata nhưng thiếu model artifacts/runtime dataset), dừng với thông báo đường dẫn thiếu và hash kỳ vọng.
- Git repo có thể **không lưu trọn** `student12500/` runtime như server ban đầu. Xác minh hiện trạng; nếu thiếu thì hoàn tất teacher/eval OGSC và xuất dữ liệu đúng schema, ghi blocker cho student, **không dựng một student trainer khác rồi gọi là protocol cũ**.

---

## 1. RÀ SOÁT CUỐI CÙNG: CÁC ĐIỂM SỬA SO VỚI OGSC ĐỀ XUẤT TRƯỚC

1. **Đúng mục tiêu FKR, không chỉ tổng score.** Công thức cũ dùng một scalar deficit `d(z)`, có thể đẩy trung bình điểm tăng trong khi một số bit vẫn âm. Thay bằng **vector 32 chiều deficit theo từng bit**. Đây là sửa đổi **phương pháp**, không phải thay verifier.
2. **Đúng preconditioner.** Không dùng `G = E[d JJ^T]` rồi nhân `d` nữa. Dùng `G = E[J J^T]` và giải hệ với vector deficit của **ảnh hiện tại**. Điều này loại bỏ một scalar weight trùng lặp, và chỉ cần một VJP khi inference.
3. **Không dùng oracle's Top-24 objective để thay FKR.** Oracle 24/32 là detection; **full key** là **32/32 đúng** trong từng nhóm và **FKR là phần trăm các nhóm** đạt 32/32.
4. **Không nhận sai thành bảo đảm giải tích.** `G` là Gram Jacobian **trung bình**; không đồng nhất với `J(z)J(z)^T` của ảnh hiện tại. Một gradient không bảo đảm cùng lúc sửa tất cả 32 bit, không bảo đảm radioactivity hoặc chất lượng từng ảnh. Chỉ báo cáo các kết quả thực đo.
5. **Định vị chi phí đúng:** Không thêm denoising call, nhưng **có một differentiable forward và một backward qua VAE + frozen semantic extractor** mỗi ảnh. Không được quảng cáo chỉ là một tensor addition / zero-extra-compute.
6. **Định vị điều khiển đúng:** OGSC tác động vào **final latent**, không phải bốn noise predictions. Có thể đối chiếu với oracle bằng `final_latent_oracle - final_latent_clean`; không được tuyên bố đây là cùng cơ chế nhân quả hay đã được oracle chứng minh transfer.

---

## 2. THIẾT KẾ TOÁN HỌC CHỐT (KHÔNG OPTIMIZER)

Định nghĩa **chính xác** từ weight32:

- `z` = final latent sạch do **SD2.1 clean** sinh bằng PNDM 50 steps, CFG 7.5, 512×512; tensor shape `[B,4,64,64]`.
- `D(z)` = canonical VAE decode + `VaeImageProcessor`/RGB8 PNG path của generator, dùng surrogate differentiable RGB8 STE của oracle **chỉ trong backward**.
- `F(x,c)` = **clean SD2.1 SemanticExtractor** từ weight32: VAE encoder posterior **mean**, scaling_factor, UNet `mid_block` ở `t=0` với prompt `c`, GAP.
- `q(z,c) = SemanticCarrier.coordinates(F(D(z),c))`, carrier artifacts gốc weight32. Đối với `threshold_cfg_v1`: `q = ((F-mean)@basis)/(sigmas+1e-8)` **không trừ medians**.
- Secret signs `s ∈ {-1,+1}^{32}`. `r(z,c)=s ⊙ q(z,c)` là 32 signed coordinates; `r_j > 0` tương đương đúng dấu bit `j`.
- `J(z,c)=∂r(z,c)/∂vec(z)` có shape `[32,16384]` với ảnh 512×512.

### 2.1. Một calibration matrix (offline, không learning)

Sử dụng **clean calibration split độc lập với test**. Lấy một subset đã định trước gồm **256 ảnh** (reuse quy mô `oracle_n=256`, lựa chọn cố định theo seed gốc, lưu danh sách ID và hashes), không lấy tập teacher watermark/test và không chọn ảnh theo key score. Tất cả source models frozen.

Với từng ảnh sạch `i`:

1. Tạo hoặc đọc **exact canonical clean final latent** và paired prompt; không coi VAE-encoded clean PNG là cùng final latent sinh ảnh.
2. Tính `r_i` và Jacobian `J_i` theo **đúng differentiable view** sẽ dùng ở inference. Trong giai đoạn offline mới tính 32 gradient riêng để lập 32 hàng của `J_i`; không có vòng tối ưu weights/delta.
3. Cộng dồn `G_sum += J_i @ J_i.T` ở FP64, cuối cùng `G = sym(G_sum / 256)`.
4. `G_pinv = torch.linalg.pinv(G, hermitian=True)` ở FP64; lưu cả `G`, `G_pinv`, eigenvalues/rank/condition diagnostic.

\[
\boxed{G=\mathbb E_{\mathrm{clean}}[JJ^\top],\qquad P=G^{\dagger}.}
\]

Không thêm Ridge λ, Adam/LR, rank của injector, margin mới hay schedule. Dùng pseudo-inverse theo numerical policy chính thức; nếu matrix không đủ hạng, phải báo rank/residual, không lén chỉnh λ hay giả vờ inverse chính xác. Nếu không thể ước lượng gradient/Gram ổn định, **báo infeasible** thay vì tự ý thêm một phương pháp khác.

### 2.2. Target deficit được ràng buộc từ bản weight32

Đọc chính xác `teacher_fp/calibration_target.json` và các clean-only thresholds đã đóng băng:

\[
t = \tau_{\rm clean} + \kappa\sigma_{\rm clean},\qquad \kappa=1\quad\text{(như cấu hình cũ)}.
\]

Với từng ảnh, tính **một vector 32 chiều**:

\[
\boxed{b_j(z,c)=\operatorname{clip}(t-r_j(z,c),0,2.5).}
\]

- `2.5` chính là `SHIFT_CAP` hiện có, **không phải hyperparameter mới**.
- Đây là **per-bit generalization** của target shift; **không phải** gọi nguyên bản `calibrated_shift_loss()`. Ghi rõ trong paper/code; tuyệt đối không chú thích rằng phương trình này chính là `calibrated_shift_loss` cũ.
- Các tọa độ đã có `r_j≥t` sẽ không có deficit trực tiếp.
- Phải đọc target `t` từ **frozen calibration artifact**, không tự fit lại `τ,σ` trên train watermark / validation / test.

### 2.3. One-step control tại inference

\[
\begin{aligned}
 w(z,c)&=P\,b(z,c), \\[-1mm]
 \Delta z(z,c)&=J(z,c)^\top w(z,c)
 =\nabla_z\big[\operatorname{stopgrad}(w(z,c))^\top r(z,c)\big],\\[-1mm]
 z_{\rm wm}&=z+\gamma\,\Delta z.
\end{aligned}
\]

**Chỉ một VJP/reverse-mode gradient**, vì `w` được **detach** trước khi tạo scalar `phi = sum(w * r)`.

Xấp xỉ bậc nhất:

\[
r(z_{\rm wm},c)\approx r(z,c)+\gamma J(z,c)J(z,c)^\top G^\dagger b(z,c).
\]

Nếu local Gram gần `G`, biểu thức xấp xỉ có thể tiến gần `r+γb`. **Không khẳng định** `JJᵀG†=I` trên từng ảnh, không khẳng định 32/32 sau một bước, và không khẳng định tương đương per-image oracle sau 120 Adam updates.

### 2.4. Một tham số hiệu chuẩn chất lượng `gamma`

- **Chỉ 1 scalar `γ` được chọn trên validation**, không dùng bit accuracy/FKR để chọn nó.
- Khóa `γ∈[0,1]`; bắt đầu kiểm tra `γ=1` trên paired validation, seed/prompt giống clean.
- Nếu không đạt giới hạn chất lượng cũ: dùng **bisection cố định 8 vòng** trong `[0,1]` để tìm mức tối đa thỏa **mean paired LPIPS ≤ 0.15 và mean CLIP drop ≤ 0.01** trên validation. Giới hạn này là từ `teacher_lpips_max` và `teacher_clip_drop_max`/oracle; **bắt buộc kiểm tra minh bạch** dù `threshold_cfg_v1` cũ coi LPIPS là diagnostic, không phải hard gate.
- Lưu toàn bộ các lần kiểm tra `γ`, số liệu, và lựa chọn trong manifest; **không tuning γ theo detector/FKR/test**.
- Bisection sử dụng giả định chất lượng theo cường độ tương đối trơn/đơn điệu; log mọi bất thường, không tuyên bố có chứng minh monotonicity. Nếu không tìm thấy `γ>0` hữu ích hoặc LPIPS/CLIP không ổn định, báo không đạt thay vì thay ngưỡng.
- Với `γ=0`, đường sinh ảnh phải **bitwise/PNG identical** với clean baseline cùng seeds và dtype, không gọi surrogate graph trước khi decode nếu không cần.

**Quan trọng:** các quality gates ở trên là đánh giá mức trung bình validation, **không bảo đảm chất lượng riêng từng ảnh unseen**. Runtime không tính LPIPS/CLIP để tránh chi phí.

---

## 3. KIẾN TRÚC TỐI THIỂU: FILE MỚI, IMPORT CODE CŨ

Tạo thư mục mới `ogsc/` trong **working repo** `t2ifing_radioactive`; giữ hai repo hiện có **read-only** ngoài những file mới của working repo.

```text
ogsc/
  __init__.py
  paths.py                 # Resolve existing sources; fail on missing artifacts
  carrier_bridge.py        # import/load weight32 carrier + frozen extractor, no duplicated formulas
  control.py               # G calibration, one-step VJP, target deficit
  generation.py            # adapter around original native_generate() path
  calibrate.py             # offline G and gamma, manifests, resume by completed records
  evaluate.py              # existing score() and key_metrics.aggregate_metrics()
  run.py                   # subcommands: preflight, fit-gram, fit-quality, generate, score, report
  tests/
    test_carrier_identity.py
    test_control_math.py
    test_dtype_png.py
    test_oracle_equivalence.py
    test_full_key_metrics.py
    test_manifest_resume.py
README.md
```

**KHÔNG** copy nguyên file 15 KB `semantic_carrier.py` hay thay `SemanticCarrier` bằng implementation tự viết. `carrier_bridge.py` chỉ import có kiểm soát theo source path được cấu hình (và kiểm tra SHA256). Dùng original `semantic_oracle_math.fake_quantize_rgb8_ste`, không copy logic RGB8 mới với rounding khác.

`ogsc/paths.py` nhận **operational paths** qua CLI/environment (không phải scientific hyperparameter):

- `--oracle-repo /path/to/t2ifing_radioactive`
- `--weight32-repo /path/to/Malicious_quantization`
- `--source-run /path/to/original/runs/structure32`
- `--output-run /path/to/new/ogsc32_run`
- `--teacher-clean /path/to/pinned/clean_SD21`

Nếu source-run có thể không mount trên máy local, `--preflight --plan-only` vẫn in các check sẽ thực hiện, không âm thầm tải model từ mạng. Với missing artifact ở GPU server, fail fast và nêu đúng path. Source manifests phải đọc chứ không sửa.

### Reuse policy

1. **Carrier/scorer:** Dùng `weight32/harness/semantic_carrier.py` và `weight32/harness/semantic_teacher.py`. Không dùng RSSF `semantic_sign.py` để báo FKR trong thí nghiệm weight32.
2. **Gradient/STE:** Import `oracle/bitmark_lora/semantic_oracle_math.py` và tham khảo backend oracle; **không import side effects** `semantic_oracle.py` vào production runtime, không chạy `optimize_batch()`.
3. **Generation:** Có thể thêm một adapter quanh hàm `native_generate()` trong `ogsc/generation.py` nhưng phải giữ nguyên checkpoint clean, PNDM scheduler, CFG, seeds, height/width, output_type='latent', canonical VAE decode và PIL/PNG save code. **Không gọi oracle's tail replay tại inference.**
4. **FKR:** Import `key_metrics.aggregate_metrics()` **nguyên bản** từ weight32. Nếu package imports bất tiện, đưa file vào `ogsc/vendor_locked/key_metrics.py` bằng **copy byte-for-byte, kèm upstream SHA**, không viết lại và không chỉnh logic grouping; ưu tiên import.
5. **Students:** Reuse chính xác SD1.4 student `rank32 attention_conv`, 12,500 steps, batch4, cooldown/checkpoint selection của experiment cũ. Nếu runner gốc không nằm trong repository checkout, chỉ xuất ảnh/captions và ghi `STUDENT_PIPELINE_MISSING`; không thay bằng run LoRA rank4 hay trainer khác.

---

## 4. IMPLEMENTATION GUIDELINES CHI TIẾT

### 4.1. `carrier_bridge.py`: kiểm soát source và gradient path

- Load frozen carrier artifacts weight32; assert: `num_bits == 32`, `signs ∈ {-1,+1}`, `carrier.carrier_version is None` (historical threshold branch), `mean/basis/sigmas` match source hashes.
- Clean `SemanticExtractor` phải dùng **SD2.1 clean**, không phải watermarked teacher, không phải SD1.4, và không phải h-space output của denoiser khi sinh ảnh.
- `SemanticExtractor.__init__` bản cũ **có `.to(dtype=torch.float32)`**; **không truyền shared instance của production `pipe`** cho extractor, vì nó sẽ âm thầm đổi dtype của generation UNet/VAE. Dùng **một scorer instance riêng frozen FP32** hoặc strategy offload/phase separation được test và giữ nguyên weights của production clean FP16.
- Đường differentiation phải là:

```text
final z (requires_grad)
  -> frozen VAE decode (float differentiable reference)
  -> clamp + oracle fake_quantize_rgb8_ste (forward matches RGB8)
  -> weight32 SemanticExtractor.image_latent (VAE encoder mean)
  -> weight32 SemanticExtractor.encode_latent (clean UNet mid_block t=0 + GAP)
  -> weight32 SemanticCarrier.coordinates
  -> signed r = coordinates * fixed_key
```

- Khi tính `q` phục vụ hiệu chuẩn, forward/backward dùng **cùng** đường tương thích RGB8. Khi đánh giá chính thức, **phải lưu PNG bằng native FP16 VAE** rồi gọi nguyên scorer `score()` weight32 đọc từ PNG. Báo chênh lệch `q_surrogate` vs `q_png`; không được coi STE giá trị surrogate là ground truth.
- Frozen model params `requires_grad_(False)` và luôn verify `grad is None`; chỉ `z` mang `requires_grad=True`.

### 4.2. `control.py`: pseudo-code bắt buộc

```python
# all heavy model objects are reused from the existing source modules
# z_clean: native clean final latent, [B,4,64,64]; c = prompts
# P = precomputed torch.float64 [32,32]
# tau, sigma, cap=2.5 from frozen source artifacts

def one_step(z_clean, prompts, extractor, carrier, P, target, gamma):
    if gamma == 0:
        return z_clean  # exact native baseline, without scorer execution
    # IMPORTANT: create tensor OUTSIDE torch.inference_mode()
    z = z_clean.detach().to(torch.float32).clone().requires_grad_(True)
    x = differentiable_decode_with_existing_vae(z, extractor)
    x = oracle_fake_quantize_rgb8_ste(x)
    features = extractor.encode_image(x, prompts)
    q = carrier.coordinates(features)
    r = q * carrier.signs.view(1, 32)
    b = (target - r.detach()).clamp(min=0.0, max=2.5)
    w = (b.to(torch.float64) @ P.T).to(r.dtype).detach()  # P symmetric
    phi = (w * r).sum()
    dz = torch.autograd.grad(phi, z, create_graph=False)[0]
    assert torch.isfinite(dz).all()
    marked = z.detach() + gamma * dz.detach()
    # Cast to NATIVE production dtype only at decode boundary
    return marked.to(dtype=z_clean.dtype)
```

**Lưu ý batched VJP:** `phi.sum()` xử lý một batch độc lập nếu extractor không trộn ảnh ở cấp batch. Để tránh coupling hoặc memory pressure, triển khai **per-image graph** ban đầu; batch generation có thể vẫn chia theo seed/prompt gốc. `--batch-size` chỉ là operational memory knob; không thay đổi thuật toán hoặc random generators.

**Không được:**

- gọi `w` chưa `detach()` (sẽ tạo đạo hàm bậc cao và thêm phụ thuộc sai),
- nhầm `q` và `r`,
- dùng một universal `delta_z` cho mọi prompt/seed,
- chuẩn hóa `dz` bằng heuristic/clip norm mới không có trong phương pháp,
- dùng Top-24 detection để tự động chọn bit hoặc thay verifier,
- gọi `torch.inference_mode()` khi tạo/lan truyền gradient,
- sử dụng `z` từ VAE encode của clean PNG thay cho **final latent** nếu đang chạy generation chính thức.

### 4.3. `calibrate.py`: tính `G`/`P` offline

Pseudo-code (có thể dùng `torch.autograd.grad` với `retain_graph=True` trong offline calibration):

```python
G_sum = torch.zeros(32, 32, dtype=torch.float64)
for record in fixed_256_clean_calibration_records:
    z = record.final_latent.detach().float().clone().requires_grad_(True)
    r = signed_coordinates_through_oracle_STE_and_weight32_carrier(z, record.prompt)
    jac_rows = []
    for j in range(32):
        gj = torch.autograd.grad(r[0, j], z, retain_graph=(j < 31))[0]
        jac_rows.append(gj.detach().reshape(-1).to(torch.float64))
    J = torch.stack(jac_rows)                    # [32, 16384]
    G_sum += (J @ J.T).cpu()
G = (G_sum / 256)
G = 0.5 * (G + G.T)
P = torch.linalg.pinv(G, hermitian=True)
```

- Offline 256×32 VJPs **có thể rất tốn GPU**, nhưng chỉ chạy một lần và không nằm trên inference path. Log thời gian thực tế; không đưa ra runtime estimate không có benchmark.
- Đọc prompt IDs từ calibration split đã đóng băng, lựa chọn 256 IDs một lần bằng seed đã có, hash toàn bộ. Không có train/val/test leakage.
- Dữ liệu `G`/`P` ở FP64, w/delta ở FP32. Check finite, symmetry, smallest/largest eigenvalues, numerical rank, rank residual, SHA256 artifact.
- Nếu graph checkpoint làm `autograd.grad` lỗi, tận dụng đúng helper `_checkpoint` và `SemanticExtractor` cũ; **không bypass feature extraction hoặc thay output mid_block**.
- Có resume theo mỗi record hoàn chỉnh: lưu per-image 32×32 Gram contribution (+ hash source), hoàn tất thì cộng dồn deterministic theo ID; **không lưu per-image model gradients hoặc weights**.

### 4.4. `generation.py`: can thiệp sau final latent, trước VAE

Trong nguồn weight32, `native_generate()` hiện dùng:

```python
with torch.inference_mode():
    latent = pipe(prompts, generator=generators,
                  num_inference_steps=steps, guidance_scale=7.5,
                  height=512, width=512, output_type='latent').images
    image = pipe.vae.decode(latent / pipe.vae.config.scaling_factor).sample
    return pipe.image_processor.postprocess(image, output_type='pil')
```

Triển khai **adapter mới** giữ lệnh `pipe(...)` đúng như trên, nhưng:

1. Lấy `latent` trong `torch.inference_mode()`; **thoát context**.
2. Tạo `z_clean = latent.detach().clone()` ở ngoài context (inference tensors không được gắn grad trực tiếp trong normal autograd).
3. Dùng `ogsc.control.one_step(...)` với scorer frozen riêng và `P` đã đóng băng.
4. Gọi **chính `pipe.vae.decode(z_marked / scaling_factor)` + `pipe.image_processor.postprocess`** dưới `torch.no_grad()`/`inference_mode()`; lưu canonical PNG theo code hiện hành.
5. Mọi prompt, seed, timestep, scheduler history phải không đổi. Đầu ra `gamma=0` phải **giống từng pixel** với original native generation.
6. Bắt buộc test no `.grad` for all model params, no modified state_dict, no out-of-order RNG effect từ scorer; scorer không dùng sampling RNG.
7. Cache `dz` khi chạy gamma quality calibration (cùng prompt/seed) để không tính gradient lặp lại cho mỗi phép thử gamma; chỉ decode/tính utility mới.

**GPU memory:** Clean SD2.1 producer FP16 và frozen scorer FP32 là **hai inference objects khác nhau**. Trước tiên làm preflight bộ nhớ GPU, thử per-image, hỗ trợ offload producer giữa pha generation/scoring nếu cần, không lặng lẽ đổi precision hoặc model. Định lượng peak VRAM và throughput sau khi chạy trên GPU server.

### 4.5. `evaluate.py`: tuyệt đối đúng FKR bản weight32

Tái sử dụng `key_metrics.aggregate_metrics()` từ `BEST_TEACHER_32bits_refine`:

```text
for group size N in {1,8,32,64,256}:
  N=1: dùng đủ 1000 ảnh, mỗi ảnh là một group
  N>1: 100 permutations có seeded RNG; groups không trùng ảnh trong partition
  mean_q_j = mean(original normalised q_j(image) over group)
  recovered_j = +1 if mean_q_j >= 0 else -1
  group_full_key = all(recovered_j == owner_sign_j for all 32 bits)
  FKR(N) = average group_full_key across groups / partitions
  bit_accuracy(N) = average matching bit ratio
```

- Chính xác tie rule `>=0 => +1`; `np.float64` aggregation như code original. **Không** dùng `K>=24`, `analytic_detected`, `model_K_M` hay aggregate toàn 1.000 ảnh làm primary metric.
- Official q phải được chấm **từ PNG thật**, không phải gradient surrogate float output; lấy `teacher_clean_test`, `teacher_ogsc_test`, student outputs và baselines được chấm với **same source carrier/key**.
- Student exact FKR cùng nhóm ảnh 1/8/32/64/256, 100 partitions, same key; report teacher và student riêng.
- `all_images_aggregate_bit_accuracy` chỉ diagnostic.
- Không dùng test FKR/quality để chọn gamma, matrix, config, hoặc student checkpoint.

### 4.6. Student distillation: kế thừa protocol, không train lại teacher

Xuất `teacher_ogsc_train`, `teacher_ogsc_validation`, `teacher_ogsc_test` đúng định dạng (RGB8 PNG + metadata.jsonl file_name/text + generation.json, hashes, seeds). Cần có `teacher_clean_*` matched source và immutable manifests.

Dùng **chính trainer SD1.4** của experiment `structure32_student12500` khi script có thật:

- Fresh clean SD1.4 base, **không** warm-start từ teacher hoặc watermarked weights.
- Rank32 attention+conv LoRA; **12.500 updates** và cooldown như code gốc; **không** chuyển thành rank4/20k của một nhánh nghiên cứu khác.
- Student chỉ thấy cặp `(PNG, caption)`, **không** thấy owner key, verifier hay training loss watermark.
- Chọn checkpoint dựa trên held-out denoising MSE như protocol cũ.
- Student inference native thường, không gắn OGSC; official scoring+FKR bằng source carrier.
- Nếu phiên bản student có ở server nhưng không ở local clone, chuẩn bị `--student-runner` trỏ đến file hiện hữu và ghi hash trước khi chạy. **Không viết lại student pipeline** để lấp chỗ trống.

---

## 5. CLI VÀ OUTPUT MONG MUỐN

Cung cấp `python -m ogsc.run --help` với các subcommands sau. Các ví dụ là **template**, không giả định mọi artifact trên máy local đã tồn tại:

```bash
# Only inspect versions, carrier, key, source artifacts and source hashes.
python -m ogsc.run preflight \
  --weight32-repo /workspace/Malicious_quantization \
  --source-run /workspace/original/runs/structure32 \
  --oracle-repo /workspace/t2ifing_radioactive \
  --output-run /workspace/ogsc32_run --plan-only

# GPU server ONLY: offline one-time Jacobian Gram / pseudoinverse.
python -m ogsc.run fit-gram \
  --weight32-repo /workspace/Malicious_quantization \
  --source-run /workspace/original/runs/structure32 \
  --oracle-repo /workspace/t2ifing_radioactive \
  --output-run /workspace/ogsc32_run

# GPU server ONLY: paired quality-only gamma calibration.
python -m ogsc.run fit-quality --output-run /workspace/ogsc32_run

# GPU server ONLY: native SD2.1 clean + one-step final-latent injection.
python -m ogsc.run generate --output-run /workspace/ogsc32_run \
  --split train
python -m ogsc.run generate --output-run /workspace/ogsc32_run \
  --split validation
python -m ogsc.run generate --output-run /workspace/ogsc32_run \
  --split test

# Score actual PNG and evaluate FKR with original weight32 implementation.
python -m ogsc.run score --output-run /workspace/ogsc32_run
python -m ogsc.run report --output-run /workspace/ogsc32_run
```

`fit-quality`, `generate`, `score`, `report` đọc frozen path manifest từ output-run, không yêu cầu nhập lại các đường dẫn/dtype/seed. Đầu vào flags chỉ là paths, split, worker/batch operational knobs. Không có CLI `--learning-rate`, `--optimizer`, `--weights`, `--margin`, `--rank` hay `--num-optimization-steps`.

Output:

```text
ogsc32_run/
  source_manifest.json               # pinned source hashes, original repo commits
  calibration/
    selected_ids.json                # 256 clean IDs, seed/hash
    gram.pt                          # G, P in FP64
    gram_diagnostics.json            # rank, eigen spectrum, timing, quality parity
    quality_gamma.json               # fixed gamma, LPIPS/CLIP validation evidence
  images/
    teacher_ogsc_train/
    teacher_ogsc_validation/
    teacher_ogsc_test/
  scores/
    teacher_ogsc_validation.json     # from actual PNG scorer
    teacher_ogsc_test.json
  result/
    teacher_key_metrics.json         # EXACT original aggregate_metrics
    teacher_quality.json            # actual PNG quality
    student_key_metrics.json         # only after existing student job completes
    latency.json                     # native baseline vs OGSC, GPU peak VRAM
    summary.md                       # status, source hashes, metrics, caveats
  state/
    stages.json
  complete.json                      # only after verified stages, never on partial failure
```

Dùng source hash + immutable completed-stage markers như oracle; xử lý resume theo stage/batch, không overwrite kết quả gốc hoặc output cũ không tương thích. Không gọi `freeze()` nhầm khi chưa tạo valid image manifests.

---

## 6. KIỂM THỬ BẮT BUỘC (CODEX CHẠY LOCAL, KHÔNG CẦN GPU)

### A. Identity & compatibility

1. Load đúng source carrier weight32, `carrier_version is None`; hash `key.json`, `pca.pt`, `complete.json` và exact signs giữ nguyên.
2. Nếu chỉ có RSSF sign carrier hoặc medians: **fail**, không convert tự động.
3. Không đổi checksum của weight32/Oracle files khi chạy `ogsc`.
4. `gamma=0` trùng output native trên synthetic/tiny path; test pixel equality với deterministic seeds.
5. Frozen modules không nhận `.grad` và không thay state_dict; `z` nhận finite nonzero grad.

### B. Math unit tests (tiny differentiable mapping)

6. Với toy linear `r(z)=A z + c`, tạo `G=A A^T`, verify `dz=A.T @ pinv(G) @ b`, target signed dimension 32, VJP equivalence tới floating tolerance.
7. `b_j=clip(t-r_j,0,2.5)` đúng cho mixed signs, zero deficit, cap và boundary; test `w.detach()` ngăn second-order graph.
8. Rank-deficient `G` finite pseudoinverse, report rank/residual; no optimizer/λ fallback.
9. Gradient một lượt online: kiểm tra `autograd.grad` gọi đúng **1 lần**/ảnh (không tính offline fit-gram).
10. Numerical finite-difference test với toy VAE+extractor, gradient qua STE VAE encode / frozen UNet-like mid_block; compare actual PNG-score only as QA, không dùng surrogate làm official score.

### C. Oracle reference tests

11. Dùng fixture nhỏ để chứng minh `z_oracle - z_clean` được tạo từ original `replay_controls()` với frozen cached noise predictions, và direct-final-latent addition giải mã cùng ảnh trong sai số precision. **Không** gán một `delta` oracle từng ảnh cho mọi ảnh khác.
12. Native original-dtype zero replay và FP32 drift diagnostic theo oracle; không relax tolerance chỉ để pass.

### D. FKR regression

13. Tạo toy coordinates/key 32 bit và dùng trực tiếp `key_metrics.aggregate_metrics()`; test nhóm 1/8/32/64/256, seed/permutations, all-bit group equality. Kiểm tra `24/32` phải **fail full key**; `32/32` phải **pass**.
14. Đánh giá 1000 ảnh thật chỉ trên GPU server; kiểm tra hashes/seed/PNG/q, không sử dụng model-level `K_M` thay FKR.

### E. Operational/resume

15. Test interrupted `fit-gram` lưu record contributions, resume không nhân đôi cộng dồn.
16. Test source-run/cache changes gây SHA mismatch; output marker-last.
17. CPU import/test/compile thực sự chạy; ghi lệnh và summary. CUDA-specific runtime chưa test local phải để rõ `NOT_RUN_NO_GPU`.

---

## 7. GATES THỰC NGHIỆM — CHỈ MỘT PHƯƠNG PHÁP

**Phase 1 — Preflight / code acceptance:** CPU unit tests pass, frozen source and carrier identity verified, zero-control tests valid. Không có GPU local.

**Phase 2 — Offline calibration:** Tính được `G`, `P`, spectrum/rank; target `t=τ+σ` lấy từ source; gamma fixed bằng validation quality **không dùng FKR**.

**Phase 3 — Teacher experiment:** Sinh paired images và official PNG scores; báo `BA(N)`, `FKR(N)` cho `N∈{1,8,32,64,256}`, quality LPIPS/CLIP/FID/KID, per-bit signed mean/variance, số bit âm, distribution `||dz||`, timing latency/peak VRAM. Không sửa `γ` sau test.

**Phase 4 — Student experiment:** Dùng original 12,500-step student runner nếu khả dụng; report exact FKR nhóm ảnh với source carrier; teacher weights luôn clean.

**Interpretation**: Nếu quality tốt nhưng FKR thấp, hoặc teacher tốt mà student không transfer, báo đúng thất bại của OGSC. **Không âm thầm thêm gradient steps, LoRA, auxiliary optimizer, latent scaling heuristic, Top24 objective hay phương pháp khác**. Một single experiment hoàn chỉnh là mục tiêu, không chạy 20 phiên bản cứu từng bước.

**Performance:** wall-clock latency và peak memory là kết quả cần đo, **không có cam kết trước** rằng 1 VJP chỉ tốn bao nhiêu % của PNDM50. Thời gian offline fit-gram phải được báo riêng; không tính vào inference latency. Khi đo runtime, dùng cùng hardware, input prompts/seeds, batch policy, warmup, synchronize CUDA trước/sau, báo median và p95, so với clean generation baseline.

---

## 8. CÁC BLOCKER / FAIL-FAST MÀ CODEX PHẢI TRẢ RA

- Không có đúng original `pca.pt`, key, calibration target hoặc clean SD2.1: **không tạo artifact giả hoặc tự fit key mới**.
- Không có original student trainer: hoàn thành teacher/eval và báo `STUDENT_PIPELINE_MISSING` thay vì viết trainer mới.
- Original `SemanticExtractor` không hỗ trợ backward với dependency hiện có: làm CPU tiny repro và sửa **adapter/compatibility layer**; không thay feature definition.
- `G` không finite/rank quá thấp, surrogate và PNG q sai khác lớn, `gamma` chỉ chọn được 0, hoặc device OOM: báo số đo + nguyên nhân; không đổi model/precision/tolerance/quality gate âm thầm.
- Nếu ta thiếu test artifacts từ oracle đã chạy trên server: chỉ đối chiếu code và test fixtures sẵn có, **không tuyên bố tái lập kết quả oracle**.

---

## 9. YÊU CẦU BÀN GIAO CHO CODEX

1. Toàn bộ code dưới `ogsc/`, không sửa file nguồn gốc hai phương pháp; liệt kê rõ bất kỳ file thay đổi ngoài thư mục này và vì sao.
2. `README.md` của OGSC: dependency thực tế, mapping nguồn, lệnh chạy CPU/GPU, operational hardware requirements, `--plan-only`, ví dụ source paths.
3. Tests chạy CPU, `python -m compileall ogsc`, format/checks; log pass/fail; ghi rõ không thực thi GPU.
4. Manifest exact source SHAs và all stage hashes, deterministic offline calibration IDs.
5. Trên GPU server, sinh một kết quả teacher FKR/quality/latency hoàn chỉnh; nếu student original accessible thì chạy theo code gốc, nếu không báo blocker.
6. Scientific summary phân biệt **derived first-order prediction** và **measured RGB8-FKR/student transfer**, không suy diễn tính thành công từ oracle hoặc một gradient algebra.

---

## 10. IMPLEMENTATION PRIORITY / KHÔNG MỞ RỘNG SCOPE

**P0 (phải có):** carrier identity, 32-bit math, one-step VJP, frozen models, canonical PNG, original FKR, calibration path, CPU tests.

**P1 (phải có trước kết quả khoa học):** validation γ quality, official scoring/metrics, deterministic IDs, source hashes, GPU latency + VRAM, paired clean controls.

**P2 (sau khi P0/P1 hoàn thành):** kết nối lại **chính** student 12,500-step pipeline gốc khi đã tìm thấy đủ script/checkpoint/dataset format.

**Không làm:** vector latent cố định, mid-block hook CAHI, oracle per-image Adam120, tự thiết kế verifier mới, thay `threshold_cfg_v1` bằng sign threshold 24/32, đào tạo classifier/detector phụ, train LoRA teacher, chọn model bằng test FKR hoặc test quality.

**Acceptance cuối cùng:** một repo chứa module OGSC-32 mới, tái sử dụng đúng 2 nguồn, có thể chạy trên GPU server bằng các lệnh deterministic, và báo cáo FKR nhóm ảnh chính xác như code weight-based. **Chưa được phép tuyên bố OGSC thành công trước dữ liệu thực nghiệm.**
