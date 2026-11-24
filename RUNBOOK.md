# RUNBOOK: Tahapan Menjalankan Eksperimen Revisi

Semua perintah dijalankan dari folder root proyek (yang berisi `run_experiments.py`).
Total 240 run (~14,2 hari GPU di RTX 4060, asumsi 85 menit/run). Semua tahap bisa
di-Ctrl+C lalu dijalankan ulang: run yang sudah selesai otomatis di-skip.

## Ringkasan alur

| Tahap | Tujuan | Estimasi | Gerbang lolos jika |
|---|---|---|---|
| A. Setup | environment siap | 15 mnt | semua import berhasil |
| B. Fase 0a | cek TorchGeo, split, overlap | 10 mnt | scene & lokasi di >1 split = 0 |
| C. Fase 0b | index patch + statistik | 1-2 jam/dataset | file stats & index terbentuk |
| D. Uji cepat | semua jalur kode jalan | 30 mnt | "QUICK TEST SELESAI" |
| E. Fase 1 | factorial 2x2x2 (80 run) | ~4,7 hari | analyze menampilkan n=5 |
| F. Fase 2 | baseline lain + random non-empty (60 run) | ~3,5 hari | idem |
| G. Fase 4 | ResNet-18 SSL4EO-L + DeepLabV3+ (60 run) | ~3,5 hari | missing_keys hanya fc |
| H. Fase 3 | sensitivitas p/tau/jitter (40 run) | ~2,4 hari | idem |
| I. Analisis | tabel, uji, figure | 30 mnt | `analysis/report.md` ada |

Urutan prioritas jika waktu terbatas: A-D, lalu Fase 1, 2, 4, 3.

---

## A. Setup

Environment sudah dibuat di `D:\Reexperiment\envs\cloudshadow` (conda, Python 3.11),
dengan versi yang SAMA dengan eksperimen lama: torch 2.5.1+cu121, torchvision 0.20.1,
torchgeo 0.8.1, timm 1.0.28, rasterio 1.4.4, numpy 1.26.4, pandas 2.3.2.
Versi lengkap ada di `requirements-lock.txt`.

Aktifkan sebelum menjalankan apa pun (Anaconda Prompt / PowerShell):

```bash
conda activate D:\Reexperiment\envs\cloudshadow
cd D:\Reexperiment\cloudshadow_revision
```

Env ini memasang `PYTHONNOUSERSITE=1` saat diaktifkan, supaya paket di
`AppData\Roaming\Python\Python311` (dipakai bersama Python lain) tidak ikut terbaca.
Kalau memanggil python env secara langsung tanpa `conda activate`, set variabel itu sendiri.

Membuat ulang env di mesin lain:

```bash
conda create -p <path_env> --override-channels -c conda-forge python=3.11 pip -y
conda env config vars set PYTHONNOUSERSITE=1 -p <path_env>
conda activate <path_env>
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-lock.txt
```

Dataset mentah ada di `dataset/l7irish/` (206 scene) dan `dataset/l8biome/` (96 scene),
sudah terekstrak. Artefak fase 0 disimpan terpisah di `data/`.

## B. Fase 0a: cek setup dan split

```bash
python scripts/phase0_check_setup.py --dataset l7irish l8biome
```

Periksa output:
- Tipe bbox terbaca dan "Baca bbox tergeser: OK". Kalau gagal, sesuaikan `src/geo_compat.py`.
- Jumlah scene train/val/test terisi (bukan 0). Kalau 0, daftar scene gagal dibaca dan
  bootstrap per scene tidak akan tersedia.
- `Scene yang muncul di >1 split: 0` (wajib).
- `Split mode: location` dan `Lokasi path/row yang muncul di >1 split: 0` (wajib). Split dibuat
  per lokasi WRS-2, jadi akuisisi berbeda dari lokasi yang sama tidak terpecah ke train dan test.
- `Pasangan footprint test-train yang beririsan`: sisa sidelap path/row bertetangga (lintang tinggi).
  Laporkan jumlahnya di paper.

Hasil saat ini (seed split 123):

| | Scene train/val/test | Lokasi train/val/test | Footprint test-train ≥50% |
|---|---|---|---|
| L7 Irish | 124 / 42 / 40 | 108 / 36 / 40 | 5 pasangan (sidelap) |
| L8 Biome | 58 / 20 / 18 | 58 / 20 / 18 | 0 |

Output: `data/{ds}_split.json`. Split menentukan isi index Fase 0b, jadi kalau `SPLIT_MODE`
atau `SPLIT_SEED` diubah, ulangi Fase 0a dan 0b, lalu hapus `runs/`.

## C. Fase 0b: scan scene training

```bash
python scripts/phase0_scan_train.py --dataset l7irish l8biome --extra_splits val test
```

Tips: uji dulu dengan `--max_patches 200` untuk memastikan tidak error, lalu hapus opsi itu.
(Jangan pakai hasil `--max_patches` untuk eksperimen; jalankan ulang penuh.)

Output yang dipakai paper (`data/{ds}_train_stats.json`):
- `pct_of_all_pixels` vs `pct_of_valid_pixels` kelas shadow: menjelaskan 0,7% vs 0,033%.
- `shadow_pool_by_tau`: definisi patch ber-shadow.
- `random_sampler.pi_shadow`: pengganti asumsi 0,01 di Eq. 7 (klaim "60x" perlu dihitung ulang).
- `scan_minutes`: biaya preprocessing.

Hasil scan (split lokasi, seed 123), untuk paper:

| | L7 Irish | L8 Biome |
|---|---|---|
| Shadow, % piksel valid (train) | 0,731% | 0,848% |
| Shadow, % semua piksel termasuk Fill | 0,095% | 0,121% |
| Patch grid train non-kosong | 134.209 dari 968.665 (13,9%) | 18.150 dari 108.845 (16,7%) |
| Patch RandomGeoSampler berisi data | 14,8% | 18,0% |
| π_shadow, RandomGeoSampler | 0,90% | 1,94% |
| π_shadow, patch non-kosong acak | 6,1% | 10,8% |
| Pool shadow τ=0 | 7.939 patch, 47 scene | 1.950 patch, 40 scene |
| Shadow, % piksel valid val / test | 0,63% / 0,12% | 0,22% / 4,77% |
| Waktu scan train | 53 menit | 8 menit |

Dengan p=0,6, peluang patch ber-shadow ≈ 0,6 + 0,4 × π: sekitar 60,4% (L7) dan 60,8% (L8),
yaitu ~67× dan ~31× dibanding RandomGeoSampler, atau ~10× dan ~6× dibanding sampling
acak dari patch non-kosong.

## D. Uji cepat (wajib sebelum menghabiskan hari GPU)

```bash
python run_experiments.py --phase 1 2 3 4 --quick_test
```

Ini menjalankan satu run per kombinasi faktor selama 1 epoch, hasil tidak disimpan.
Kalau ada error di sampler, loss, atau arsitektur, ketahuan di sini.

Lihat daftar dan estimasi tiap fase:

```bash
python run_experiments.py --phase 1 --dry_run
```

## E. Fase 1: factorial

```bash
python run_experiments.py --phase 1
```

Setelah beberapa run selesai, cek bertahap tanpa menunggu semuanya:

```bash
python analyze.py --n_boot 0
```

Sanity check awal: log tiap epoch menampilkan "patch ber-shadow". Untuk run oversample
harus jauh lebih tinggi (puluhan persen) dibanding random (di bawah ~1-3%). Kalau sama,
sampler tidak bekerja.

Strategi hemat waktu: mulai dengan `--dataset l7irish --seeds 42 123 2024`, lalu tambah
seed 7 dan 2025 serta L8 Biome belakangan.

## F. Fase 2: baseline imbalance lain

```bash
python run_experiments.py --phase 2
```

Log "[Loss] class-balanced weights" mencetak bobot CB. Salin nilainya ke paper.
Log "[Sampler] nonempty: pool=..." mencetak jumlah patch non-kosong; catat untuk paper.

Cara membaca hasil Fase 2 bersama Fase 1 (U-Net, WCE):
- random → non-empty : efek menghindari patch Fill
- non-empty → OS     : efek memprioritaskan shadow (inti klaim paper)

## G. Fase 4: backbone pretrained (ResNet-18 SSL4EO-L, DeepLabV3+ ImageNet)

```bash
python run_experiments.py --phase 4 --dataset l7irish --seeds 42
```

Perintah di atas menjalankan 6 run (4 ResNet-18, 2 DeepLabV3+) untuk seed 42. Buka JSON
run ResNet-18 SSL di `runs/l7irish/`:
- `model_load_info.missing_keys` hanya `fc` atau kosong.
- `conv1_adapted` = `false`.

Untuk DeepLabV3+, `model_load_info` kosong (bobot ImageNet dimuat oleh smp; conv1 diadaptasi
otomatis ke 9/11 band). Run pertama mengunduh bobot ResNet-50 ImageNet (~100 MB), jadi
butuh internet.

Kalau lolos, jalankan penuh:

```bash
python run_experiments.py --phase 4
```

Catatan normalisasi: default `zscore`. Kalau dokumentasi TorchGeo SSL4EO-L meminta skala
lain, ubah **default `norm` di `RunConfig`** (`src/experiment.py`) sebelum menjalankan
Fase 4. Jangan memakai `--set norm=div255` untuk produksi: nama konfigurasi akan berakhiran
`norm-div255` dan `analyze.py` (yang mencari kunci default) tidak akan menemukannya.

## H. Fase 3: sensitivitas p, tau, jitter (L7 Irish)

```bash
python run_experiments.py --phase 3
```

Konfigurasi p=0,6 sama dengan Fase 1, jadi tidak dijalankan ulang.

## H2. Fase 6: pilot partial label (1 seed, 12 run)

```bash
python scripts/make_partial_masks.py --dataset l8biome l7irish --mode geom all
python run_experiments.py --phase 6 --quick_test --dataset l8biome
python run_experiments.py --phase 6
```

Membandingkan mask asli, partial label semua-Clear (Francis 2024), dan partial label dibatasi
geometri bayangan, masing-masing dengan sampling non-empty dan oversample. Run `none` sama
namanya dengan run Fase 1/2, jadi tidak dijalankan dua kali. Penjelasan lengkap di README.

Yang dicek setelah pilot: shadow IoU/precision/recall di scene test berlabel shadow, dan apakah
`geom` > `all` > `none`. Kalau ya, fase utama disusun ulang di sekitar faktor `partial`.

## H3. Koreksi prior (setelah run oversample selesai)

```bash
python scripts/prior_correction.py --pattern "*oversample*"
```

Hanya inferensi pada checkpoint yang ada; bisa paralel dengan training. Lihat README.

## H4. Validasi eksternal SPARCS (setelah ada run L8 Biome)

```bash
python scripts/prepare_sparcs.py   # sekali (butuh dataset/sparcs/sparcs_data_L8.zip)
python scripts/eval_sparcs.py      # run yang sudah dievaluasi di-skip; --force untuk ulang
```

Hasil di `runs/l8biome/sparcs/`. Hanya inferensi; bisa paralel dengan training. Lihat README.

## I. Analisis dan figure

```bash
python analyze.py                        # report.md + CSV (bootstrap 2000x)
python make_figures.py --cm --curves
python make_figures.py --qualitative --dataset l7irish --seed 42
```

Hasil di `analysis/` dan `figures/`.

---

## Aturan agar hasil tidak tercampur

- Jangan ubah `SELECT_METRIC`, `TRAIN_DEFAULTS`, atau kode loss/sampler di tengah
  eksperimen. Nama run tidak mencatat `SELECT_METRIC` dan beberapa pengaturan lain, jadi hasil
  lama dan baru akan bercampur diam-diam. Kalau terpaksa mengubah, hapus `runs/` atau
  pindahkan ke folder lain.
- Untuk menjalankan ulang satu run: tambahkan `--force`.
- Run yang gagal dicatat di `runs/failures_phase*.json`. Jalankan ulang perintah yang sama.
- Checkpoint ada di `runs/{ds}/checkpoints/`, log per run di `runs/{ds}/logs/`.

## Troubleshooting singkat

| Gejala | Penyebab umum | Tindakan |
|---|---|---|
| Error proj.db di Windows | konflik PROJ | sudah ditangani `data.py`; pastikan rasterio/pyproj terpasang |
| CUDA out of memory | batch 8 x 256 x 256 terlalu besar | tutup aplikasi GPU lain; turunkan `--num_workers` tidak membantu, turunkan `batch_size` lewat `--set batch_size=4` (mengubah nama run) |
| DataLoader macet di Windows | worker multiprocessing | `--num_workers 0` |
| "Pool shadow kosong" | tau terlalu tinggi | turunkan tau |
| Bootstrap/ANOVA tidak muncul | scene kosong atau factorial belum lengkap | cek Fase 0a dan kelengkapan Fase 1 |
| `ModuleNotFoundError: tabulate` | dependensi | `pip install tabulate` |

## Checklist sebelum menulis ulang paper

- [ ] `*_split.json`: overlap footprint test-train dilaporkan
- [ ] `*_train_stats.json`: prevalensi shadow dengan dan tanpa Fill, pi_shadow empiris
- [ ] Fase 1 lengkap n=5 di kedua dataset, ANOVA terisi
- [ ] Fase 2 dan 4 lengkap, termasuk SSL tanpa oversampling
- [ ] Tabel melaporkan recall, precision, F1, IoU, serta thin dan cloud
- [ ] Bobot WCE hand-picked [0, 5, 1, 1.5, 1], gamma, alpha/beta Tversky (alpha FP = 0,3; beta FN = 0,7), dan beta CB disebut di paper
- [ ] Kriteria pemilihan checkpoint (val mIoU, hanya dari epoch ≥5) dan training minimal 20 epoch sebelum early stopping disebut di paper
- [ ] Klaim judul dan kesimpulan dibatasi pada modul dan loss yang diuji
