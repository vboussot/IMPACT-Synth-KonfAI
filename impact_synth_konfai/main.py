import argparse
import importlib.metadata
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import SimpleITK as sitk  # noqa: N813
from konfai.utils.utils import get_available_models_on_hf_repo

SUPPORTED_EXTENSIONS = [
    "mha",
    "mhd",  # MetaImage
    "nii",
    "nii.gz",  # NIfTI
    "nrrd",
    "nrrd.gz",  # NRRD
    "gipl",
    "gipl.gz",  # GIPL
]

IMPACT_SYNTH_KONFAI_REPO = "VBoussot/ImpactSynth"


def save_sct(sct_path: Path, output_sct_path: Path):
    if not sct_path.exists():
        print(f"❌ Prediction not found at: {sct_path}\n   Check KonfAI logs for details.", file=sys.stderr)
        sys.exit(1)

    try:
        sct = sitk.ReadImage(str(sct_path))
        sitk.WriteImage(sct, str(output_sct_path))
    except Exception as e:
        print(
            f"❌ Error saving output synthetic CT :\n   from: {sct_path}\n   to  : {output_sct_path}\n   detail: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="ImpactSynth", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-i",
        "--input",
        metavar="filepath",
        dest="input",
        help="Input image path.",
        type=lambda p: Path(p).absolute(),
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output",
        metavar="filepath",
        dest="output",
        help="Output synthetic CT path.",
        type=lambda p: Path(p).absolute(),
        required=False,
        default=Path("sct.mha").absolute(),
    )

    parser.add_argument(
        "-m",
        "--model",
        choices=list(get_available_models_on_hf_repo(IMPACT_SYNTH_KONFAI_REPO)),
        help="Select which model to use. This determines what is predicted.",
        default="MR",
    )

    parser.add_argument(
        "--tta", type=int, default=2, help="Number of test-time augmentations (TTA) to apply during inference."
    )

    parser.add_argument("--ensemble", type=int, default=5, help="Number of models to ensemble for prediction.")

    parser.add_argument(
        "--mc_dropout", type=int, default=0, help="Number of Monte Carlo dropout samples for uncertainty estimation."
    )

    parser.add_argument("-uncertainty", action="store_true", help="Save uncertainty output.")

    parser.add_argument("-quiet", action="store_true", help="Suppress console output.")

    parser.add_argument(
        "-g",
        "--gpu",
        type=str,
        default=(os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ else ""),
        help="GPU list (e.g. '0' or '0,1'). Leave empty for CPU.",
    )

    parser.add_argument("--cpu", type=int, default=1, help="Number of CPU cores to use when --gpu is empty.")

    parser.add_argument("--version", action="version", version=importlib.metadata.version("IMPACT-Synth-KonfAI"))

    args = parser.parse_args()
    # --- Input checks ---
    if not args.input.exists():
        print(f"❌ Input file does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not any(str(args.input).endswith(ext) for ext in SUPPORTED_EXTENSIONS):
        print(f"❌ Unsupported input extension: {args.input.name}", file=sys.stderr)
        print(f"   Supported: {', '.join(SUPPORTED_EXTENSIONS)}", file=sys.stderr)
        sys.exit(1)

    # --- Output checks ---
    out_parent = args.output.parent
    try:
        out_parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"❌ Cannot create output directory {out_parent}: {e}", file=sys.stderr)
        sys.exit(1)

    if not any(str(args.output).endswith(ext) for ext in SUPPORTED_EXTENSIONS):
        print(f"❌ Unsupported output extension: {args.output.name}", file=sys.stderr)
        print(f"   Supported: {', '.join(SUPPORTED_EXTENSIONS)}", file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        dataset_p = tmpdir / "Dataset" / "P001"
        dataset_p.mkdir(parents=True, exist_ok=True)

        # Convert input to expected NIfTI
        vol_out = dataset_p / "Volume.nii.gz"
        try:
            img = sitk.ReadImage(str(args.input))
            sitk.WriteImage(img, str(vol_out))
        except Exception as e:
            print(
                f"❌ Error reading/writing image with SimpleITK:\n   in : {args.input}\n",
                f"out: {vol_out}\n   detail: {e}",
                file=sys.stderr,
            )
            sys.exit(1)

        cmd = [
            "konfai",
            "PREDICTION_HF",
            "-y",
            "--MODEL",
            str(args.ensemble),
            "--tta",
            str(args.tta),
            "--config",
            f"{IMPACT_SYNTH_KONFAI_REPO}:{args.model}",
        ]

        if args.gpu:
            cmd += ["--gpu", args.gpu]
        else:
            cmd += ["--cpu", str(args.cpu)]
        if args.quiet:
            cmd += ["-quiet"]

        try:
            subprocess.run(cmd, cwd=tmpdir, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ 'konfai PREDICTION' failed with exit code {e.returncode}.", file=sys.stderr)
            sys.exit(e.returncode)
        except FileNotFoundError:
            print("❌ 'konfai' executable not found. Ensure it is installed and on PATH.", file=sys.stderr)
            sys.exit(1)

        save_sct(tmpdir / "Predictions" / "ImpactSynth" / "Dataset" / "P001" / "sCT.mha", args.output)

        suffix = "".join(args.output.suffixes)
        base = args.output.stem
        i = 0
        if args.uncertainty:
            uncertainty_path = args.output.parent / f"{base}_Uncertainty"
            uncertainty_path.mkdir(parents=True, exist_ok=True)
            for ensemble in range(args.tta):
                for tta in range(args.ensemble):
                    save_sct(
                        tmpdir / "Predictions" / "ImpactSynth" / "Dataset" / "P001" / f"sCT_{i}.mha",
                        uncertainty_path / f"{base}_{ensemble:02d}_{tta:02d}{suffix}",
                    )
                    i += 1

            save_sct(
                tmpdir / "Predictions" / "ImpactSynth" / "Dataset" / "P001" / "sCT_var.mha",
                uncertainty_path / f"{base}_var{suffix}",
            )
    if not args.quiet:
        print(f"✅ Done. Synthetic CT saved to: {args.output}")
