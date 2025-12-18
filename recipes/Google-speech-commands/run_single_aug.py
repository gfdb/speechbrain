#!/usr/bin/env python3
"""
Run isolated augmentation experiments for wav2aug vs SpeechBrain comparison.

This script makes it easy to run experiments with individual augmentations
to identify performance differences between wav2aug and SpeechBrain.

Usage:
    # Run wav2aug drop_chunk experiment:
    python run_single_aug.py --aug drop_chunk --backend wav2aug --data_folder /path/to/GSC

    # Run SpeechBrain drop_chunk experiment:
    python run_single_aug.py --aug drop_chunk --backend speechbrain --data_folder /path/to/GSC
    
    # Run baseline (no augmentation):
    python run_single_aug.py --aug none --data_folder /path/to/GSC
    
    # Run all augmentations sequentially for both backends:
    python run_single_aug.py --all --backend both --data_folder /path/to/GSC

Available augmentations:
    add_noise, speed_perturb, drop_freq, drop_chunk, clipping,
    rand_amp, sign_flip, chunk_swap, babble_noise, none
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

AUGMENTATIONS = [
    "add_noise",
    "speed_perturb", 
    "drop_freq",
    "drop_chunk",
    "clipping",
    "rand_amp",
    "sign_flip",
    "chunk_swap",
    "babble_noise",
]

# Mapping from augmentation name to SpeechBrain YAML reference
SB_AUG_REFS = {
    "add_noise": "sb_add_noise",
    "speed_perturb": "sb_speed_perturb",
    "drop_freq": "sb_drop_freq",
    "drop_chunk": "sb_drop_chunk",
    "clipping": "sb_clipping",
    "rand_amp": "sb_rand_amp",
    "sign_flip": "sb_sign_flip",
    "chunk_swap": "sb_chunk_swap",
    "babble_noise": "sb_babble_noise",
}


def run_experiment(
    aug_name: str,
    backend: str,
    data_folder: str,
    extra_args: list[str] | None = None,
) -> int:
    """Run a single augmentation experiment."""
    script_dir = Path(__file__).parent
    
    if backend == "wav2aug":
        hparams_file = script_dir / "hparams" / "single_aug.yaml"
        cmd = [
            sys.executable,
            str(script_dir / "train.py"),
            str(hparams_file),
            f"--data_folder={data_folder}",
            f"--augmentation_name={aug_name}",
        ]
    else:
        # For SpeechBrain, we need to modify the augmentations list in the YAML
        # We do this via command-line override
        hparams_file = script_dir / "hparams" / "single_aug_sb.yaml"
        
        if aug_name == "none":
            # No augmentation - override wav_augment to None
            override = "wav_augment=null"
        else:
            sb_ref = SB_AUG_REFS.get(aug_name)
            if not sb_ref:
                print(f"Unknown augmentation for SpeechBrain: {aug_name}")
                return 1
            # Override the augmentations list to use the selected one
            override = f"wav_augment=!new:speechbrain.augment.augmenter.Augmenter\n  parallel_augment: False\n  concat_original: False\n  repeat_augment: 1\n  shuffle_augmentations: False\n  min_augmentations: 1\n  max_augmentations: 1\n  augmentations: [!ref <{sb_ref}>]"
        
        cmd = [
            sys.executable,
            str(script_dir / "train.py"),
            str(hparams_file),
            f"--data_folder={data_folder}",
            f"--augmentation_name={aug_name}",
            f"--output_folder=results/single_aug_sb_{aug_name}/1986",
        ]
        
        # For SpeechBrain, we'll use a simpler approach: generate a temp YAML
        if aug_name != "none":
            temp_yaml = generate_sb_yaml(aug_name, script_dir)
            cmd[2] = str(temp_yaml)
    
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"\n{'='*60}")
    print(f"Running: {backend} with augmentation={aug_name}")
    print(f"Command: {' '.join(cmd[:5])}...")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd)
    return result.returncode


def generate_sb_yaml(aug_name: str, script_dir: Path) -> Path:
    """Generate a temporary YAML file for SpeechBrain single augmentation."""
    base_yaml = script_dir / "hparams" / "single_aug_sb.yaml"
    content = base_yaml.read_text()
    
    sb_ref = SB_AUG_REFS[aug_name]
    
    # Replace the augmentations line
    old_line = "augmentations: [!ref <sb_drop_chunk>]"
    new_line = f"augmentations: [!ref <{sb_ref}>]"
    content = content.replace(old_line, new_line)
    
    # Also update output folder
    content = content.replace(
        "output_folder: !ref results/single_aug_sb_<augmentation_name>/<seed>",
        f"output_folder: !ref results/single_aug_sb_{aug_name}/<seed>"
    )
    
    # Write to temp file in hparams directory (so relative refs work)
    temp_yaml = script_dir / "hparams" / f"_temp_single_aug_sb_{aug_name}.yaml"
    temp_yaml.write_text(content)
    return temp_yaml


def main():
    parser = argparse.ArgumentParser(
        description="Run isolated augmentation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--aug",
        type=str,
        default="none",
        help=f"Augmentation to test. Options: {', '.join(AUGMENTATIONS + ['none'])}",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["wav2aug", "speechbrain", "both"],
        default="wav2aug",
        help="Which augmentation backend to use",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        required=True,
        help="Path to Google Speech Commands dataset",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all augmentations sequentially",
    )
    
    args, extra = parser.parse_known_args()
    
    if args.all:
        augmentations = AUGMENTATIONS + ["none"]
    else:
        augmentations = [args.aug]
    
    backends = ["wav2aug", "speechbrain"] if args.backend == "both" else [args.backend]
    
    results = {}
    for backend in backends:
        for aug in augmentations:
            key = f"{backend}/{aug}"
            ret = run_experiment(aug, backend, args.data_folder, extra)
            results[key] = "SUCCESS" if ret == 0 else f"FAILED (code {ret})"
    
    print(f"\n{'='*60}")
    print("Experiment Summary")
    print(f"{'='*60}")
    for key, status in results.items():
        print(f"  {key}: {status}")


if __name__ == "__main__":
    main()
