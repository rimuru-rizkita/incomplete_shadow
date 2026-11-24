"""
run_experiments.py
==================
Runner untuk fase training (1-4). Run yang sudah selesai otomatis di-skip,
jadi aman dihentikan (Ctrl+C) dan dijalankan ulang kapan saja.

Contoh:
    python run_experiments.py --phase 1 --dry_run              # lihat daftar run + estimasi waktu
    python run_experiments.py --phase 1 --quick_test           # uji semua jalur kode (1 epoch)
    python run_experiments.py --phase 1                        # jalankan fase 1 lengkap
    python run_experiments.py --phase 1 --dataset l7irish --seeds 42 123
    python run_experiments.py --phase 4 --set norm=div255      # override field RunConfig
"""

import argparse
import time
import traceback

from src import config as C
from src.experiment import apply_overrides
from src.phases import PHASES, get_runs
from src.utils import Tee, save_json

AVG_MINUTES_PER_RUN = 85  # dari pengalaman versi lama (60-110 menit di RTX 4060)


def keep_system_awake():
    """
    Windows: minta sistem tidak idle-sleep selama proses ini hidup
    (SetThreadExecutionState, otomatis lepas saat proses selesai; pengaturan
    daya tidak diubah). Menutup lid / Sleep manual tetap membuat laptop tidur.
    """
    import sys
    if sys.platform != "win32":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    if ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED):
        print("[Power] sistem dijaga tidak sleep selama training berjalan.")


def unique(runs):
    seen, out = set(), []
    for r in runs:
        if r.name() not in seen:
            seen.add(r.name())
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, nargs="+", required=True, choices=list(PHASES))
    ap.add_argument("--dataset", nargs="+", default=None, choices=list(C.DATASETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="override field, mis. lr=0.003")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--quick_test", action="store_true")
    ap.add_argument("--force", action="store_true", help="jalankan ulang walau hasil sudah ada")
    args = ap.parse_args()

    runs = []
    for ph in args.phase:
        runs += get_runs(ph, args.dataset, args.seeds)
    runs = unique([apply_overrides(r, args.set) for r in runs])
    todo = [r for r in runs if args.force or not r.result_path().exists()]

    print(f"Fase {args.phase}: {len(runs)} run unik, {len(runs) - len(todo)} sudah selesai, "
          f"{len(todo)} tersisa (estimasi ~{len(todo) * AVG_MINUTES_PER_RUN / 60:.0f} jam GPU)")
    if args.dry_run:
        for r in todo:
            print("  ", r.name())
        return

    if args.quick_test:
        from src.train import train_one_run
        # Satu run per kombinasi faktor unik (seed pertama saja), 1 epoch, 16 patch,
        # test 32 patch. Hasil TIDAK disimpan. Tujuannya: memastikan semua jalur kode
        # (sampler, loss, arsitektur) jalan sebelum menghabiskan hari-hari GPU.
        combos, picked = set(), []
        for r in runs:
            c = (r.dataset, r.sampling, r.p, r.tau, r.jitter, r.arch, r.cbam, r.loss,
                 r.pretrained, r.freeze_encoder, r.partial)
            if c not in combos:
                combos.add(c)
                picked.append(r)
        print(f">>> QUICK TEST: {len(picked)} kombinasi, num_workers=0, hasil tidak disimpan.")
        for r in picked:
            train_one_run(r, num_workers=0, quick_test=True)
        print("\nQUICK TEST SELESAI: semua jalur kode berjalan.")
        return

    from src.train import train_one_run
    keep_system_awake()
    failures = []
    t0 = time.time()
    for i, cfg in enumerate(todo, 1):
        print(f"\n##### [{i}/{len(todo)}] {cfg.name()}")
        try:
            with Tee(cfg.log_path()):
                train_one_run(cfg, num_workers=args.num_workers, force=args.force)
        except KeyboardInterrupt:
            print("\nDihentikan manual. Jalankan ulang perintah yang sama untuk melanjutkan.")
            break
        except Exception as e:  # catat, lanjut ke run berikutnya
            traceback.print_exc()
            failures.append({"run": cfg.name(), "error": repr(e)})
    hours = (time.time() - t0) / 3600
    print(f"\nSelesai dalam {hours:.1f} jam. Gagal: {len(failures)}")
    if failures:
        save_json(failures, C.RUNS_DIR / f"failures_phase{'-'.join(map(str, args.phase))}.json")
        for f in failures:
            print("  GAGAL:", f["run"], f["error"])


if __name__ == "__main__":
    main()
