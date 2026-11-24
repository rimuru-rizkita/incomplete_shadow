"""
experiment.py
=============
RunConfig: SATU sumber kebenaran untuk setiap run. Semua faktor eksperimen
disimpan di JSON hasil (versi lama: sebagian faktor hanya ditebak dari nama
file, sehingga run --no_pretrained / lr / augment berbeda bisa tercampur
dalam satu baris tabel).

Nama run dibentuk deterministik dari faktor-faktornya, sehingga konfigurasi
yang sama di fase berbeda (mis. p=0.6 di fase 1 dan fase 3) otomatis
memakai hasil yang sama dan tidak dijalankan dua kali.
"""

from dataclasses import asdict, dataclass, fields, replace

from . import config as C

D = C.TRAIN_DEFAULTS


@dataclass(frozen=True)
class RunConfig:
    dataset: str
    seed: int
    sampling: str = "random"      # random | oversample | ratio_weighted | class_aware | nonempty
    p: float = 0.6                # peluang ambil dari pool shadow
    tau: float = 0.0              # ambang fraksi shadow untuk masuk pool (0 = >=1 piksel)
    jitter: float = 0.0           # geser patch pool maks jitter x lebar patch
    arch: str = "unet"            # unet | resnet18 | deeplabv3plus | segformer
    cbam: bool = False            # hanya unet & resnet18
    loss: str = "wce"             # ce | wce | cb | ohem | ohem_w | hybrid
    pretrained: bool = False      # resnet18: SSL4EO-L; deeplabv3plus & segformer: ImageNet
    freeze_encoder: bool = False  # hanya untuk resnet18
    augment: bool = True
    norm: str = "zscore"          # zscore | div255 | none
    partial: str = "none"         # none | all | geom (partial label scene tanpa anotasi shadow)
    lr: float = D["lr"]
    epochs: int = D["epochs"]
    batch_size: int = D["batch_size"]
    patch_size: int = D["patch_size"]
    patches_per_epoch: int = D["patches_per_epoch"]
    val_patches: int = D["val_patches"]

    # ------------------------------------------------------------------
    def uses_pool(self) -> bool:
        return self.sampling in ("oversample", "ratio_weighted", "class_aware", "nonempty")

    def key(self) -> str:
        """Nama konfigurasi TANPA seed (dipakai untuk agregasi antar seed)."""
        parts = [self.dataset, f"samp-{self.sampling}"]
        if self.sampling in ("oversample", "ratio_weighted"):
            parts.append(f"p{self.p:g}_tau{self.tau:g}")
        if self.uses_pool() and self.jitter > 0:
            parts.append(f"jit{self.jitter:g}")
        arch = self.arch
        if self.arch == "resnet18":
            arch += "-ssl" if self.pretrained else "-scratch"
            if self.freeze_encoder:
                arch += "-frozen"
        elif self.arch in ("deeplabv3plus", "segformer"):
            arch += "-in1k" if self.pretrained else "-scratch"
        parts.append(f"arch-{arch}{'-cbam' if self.cbam else ''}")
        parts.append(f"loss-{self.loss}")
        if self.partial != "none":
            parts.append(f"pl-{self.partial}")
        if not self.augment:
            parts.append("noaug")
        if self.norm != "zscore":
            parts.append(f"norm-{self.norm}")
        for f, default in [("lr", D["lr"]), ("epochs", D["epochs"]),
                           ("batch_size", D["batch_size"]), ("patch_size", D["patch_size"]),
                           ("patches_per_epoch", D["patches_per_epoch"])]:
            if getattr(self, f) != default:
                parts.append(f"{f}{getattr(self, f):g}")
        return "__".join(parts)

    def name(self) -> str:
        return f"{self.key()}__s{self.seed}"

    def result_path(self):
        return C.RUNS_DIR / self.dataset / f"{self.name()}.json"

    def checkpoint_path(self):
        return C.RUNS_DIR / self.dataset / "checkpoints" / f"{self.name()}.pt"

    def log_path(self):
        return C.RUNS_DIR / self.dataset / "logs" / f"{self.name()}.log"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "RunConfig":
        names = {f.name for f in fields(RunConfig)}
        return RunConfig(**{k: v for k, v in d.items() if k in names})


def apply_overrides(cfg: RunConfig, overrides: list[str]) -> RunConfig:
    """Override field dari CLI, format key=value (mis. norm=div255 lr=0.003)."""
    if not overrides:
        return cfg
    types = {f.name: f.type for f in fields(RunConfig)}
    changes = {}
    for item in overrides:
        k, v = item.split("=", 1)
        if k not in types:
            raise KeyError(f"Field tidak dikenal: {k}")
        t = types[k] if not isinstance(types[k], str) else eval(types[k])  # noqa: S307
        changes[k] = (v.lower() in ("1", "true", "yes")) if t is bool else t(v)
    return replace(cfg, **changes)
