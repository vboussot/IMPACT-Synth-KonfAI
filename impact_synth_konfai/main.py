import argparse
from pathlib import Path
import os
import importlib.metadata
import tempfile
import shutil
import sys
import subprocess
from ruamel.yaml import YAML

import SimpleITK as sitk # noqa: N813
from huggingface_hub import hf_hub_download, HfApi

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
HF_REPOSITORIES = ["VBoussot/ImpactSynth"]


def ensure_konfai_available() -> None:
    from shutil import which

    if which("konfai") is None:
        print("❌ 'konfai' CLI not found in PATH. Install/activate KonfAI.", file=sys.stderr)
        sys.exit(1)

def _get_available_models(default_hf_repositories: list[str]) -> dict[str, str]:
    api = HfApi()
    default_models_dir = {}
    for default_hf_repository in default_hf_repositories:
        tree = api.list_repo_tree(repo_id=default_hf_repository)
        for entry in tree:
            if entry.path not in ["README.md", ".gitattributes"]:
                default_models_dir[entry.path] = default_hf_repository
    return default_models_dir

def get_models_name(model: str, number_of_models: int) -> list[str]:
    return [f"{model}/CV_{i}.pt" for i in range(number_of_models)]

def download_models(base_model: str, models_name: list[str], repo_id: str) -> tuple[list[str], str, str]:
    models_path = []
    for model_name in models_name:
        models_path.append(hf_hub_download(
            repo_id=repo_id,
            filename=model_name,
            repo_type="model",
            revision="main"
        ))

    model_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{base_model}/Model.py",
            repo_type="model",
            revision="main"
        )
    
    inference_file_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{base_model}/Prediction.yml",
            repo_type="model",
            revision="main"
        )
    
    requirements_file_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{base_model}/requirements.txt",
            repo_type="model",
            revision="main"
        )
    
    with open(requirements_file_path, "r", encoding="utf-8") as f:
        required_packages = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    installed = {dist.metadata["Name"].lower() for dist in importlib.metadata.distributions()}
    missing = []
    for req in required_packages:
        if req not in installed:
            missing.append(req)
    if missing:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            print("✅ Installation completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Installation failed with exit code {e.returncode}.")
            sys.exit(e.returncode)

    return models_path, inference_file_path, model_path

def set_augmentation_nb(inference_file_path, new_value):
    yaml = YAML()
    with open(inference_file_path) as f:
        data = yaml.load(f)

    tmp = data["Predictor"]["Dataset"]["augmentations"]
    if "DataAugmentation_0" in tmp:
        tmp["DataAugmentation_0"]["nb"] = new_value
    with open(inference_file_path, "w") as f:
        yaml.dump(data, f)

def save_sCT(sCT_path: Path, output_sCT_path: Path):
    if not sCT_path.exists():
        print(f"❌ Prediction not found at: {sCT_path}\n   Check KonfAI logs for details.", file=sys.stderr)
        sys.exit(1)

    try:
        sCT = sitk.ReadImage(str(sCT_path))
        sitk.WriteImage(sCT, str(output_sCT_path))
    except Exception as e:
        print(
            f"❌ Error saving output synthetic CT :\n   from: {sCT_path}\n   to  : {output_sCT_path}\n   detail: {e}",
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
        default=Path("sCT.mha").absolute(),
    )

    available_models = _get_available_models(HF_REPOSITORIES)
    parser.add_argument("-m", "--model", choices=list(available_models.keys()),
                        help="Select which model to use. This determines what is predicted.",
                        default="MR")
    
    parser.add_argument(
        "--tta",
        type=int,
        default=2,
        help="Number of test-time augmentations (TTA) to apply during inference."
    )

    parser.add_argument(
        "--ensemble",
        type=int,
        default=5,
        help="Number of models to ensemble for prediction."
    )

    parser.add_argument(
        "--mc_dropout",
        type=int,
        default=1,
        help="Number of Monte Carlo dropout samples for uncertainty estimation."
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


    parser.add_argument('--version', action='version', version=importlib.metadata.version("ImpactSynth"))
        
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
    

    ensure_konfai_available()
    models_name = get_models_name(args.model, args.ensemble)
    models_path, inference_file_path, model_path = download_models(args.model, models_name, available_models[args.model])
    set_augmentation_nb(inference_file_path, args.tta)
    
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        dataset_p = tmpdir / "Dataset" / "P001"
        dataset_p.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copy2(model_path, tmpdir / "Model.py")
        except Exception as e:
            print(f"❌ Cannot copy Model.py into temp dir: {e}", file=sys.stderr)
            sys.exit(1)
        
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
            "PREDICTION",
            "-y",
            "--MODEL",
            ":".join(models_path),
            "--config",
            inference_file_path,
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

        save_sCT(tmpdir / "Predictions" / "ImpactSynth" / "Dataset" / "P001" / "sCT.mha", args.output)
        
        suffix = "".join(args.output.suffixes)
        base = args.output.stem
        i = 0
        if args.uncertainty:
            uncertainty_path = args.output.parent / f"{base}_Uncertainty"
            uncertainty_path.mkdir(parents=True, exist_ok=True)
            for e in range(args.tta):
                for t in range(len(models_name)):
                    save_sCT(tmpdir / "Predictions" / "ImpactSynth" / "Dataset" / "P001" / f"sCT_{i}.mha", uncertainty_path / f"{base}_{e:02d}_{t:02d}{suffix}")
                    i+=1
            
            save_sCT(tmpdir / "Predictions" / "ImpactSynth" / "Dataset" / "P001" / f"sCT_var.mha", uncertainty_path / f"{base}_var{suffix}")
    if not args.quiet:
        print(f"✅ Done. Synthetic CT saved to: {args.output}")
