"""Extract agent training data from a trained Belief World Model checkpoint.

Runs the frozen world model over the dataset and saves per-slot features
(beliefs, priors, canonical logits, oracle emit labels, distortions) as
shard files that can be loaded by AgentDataset during GRPO training.

Usage:
    python extract_agent_data.py \\
        --checkpoint runs/world_model_v3/best.pt \\
        --features-dir /data/wm_features \\
        --metadata-dir /data/metadata \\
        --output-dir /data/agent_data
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from huperwm.data.vocab import Vocabulary
from huperwm.data.world_model import BeliefWMCollator, BeliefWMDataset
from huperwm.data.agent import extract_agent_features
from huperwm.model.world_model import BeliefWorldModel, WorldModelConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract agent training features from the world model.")
    p.add_argument("--checkpoint", required=True, help="Path to world model best.pt")
    p.add_argument("--features-dir", required=True, help="Pre-computed feature directory")
    p.add_argument("--metadata-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "validation"])
    p.add_argument("--evidence-type", choices=["logits", "hidden"], default="hidden")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phone_vocab = Vocabulary.from_file(Path(args.metadata_dir) / "phone_vocab.json")
    text_vocab = Vocabulary.from_file(Path(args.metadata_dir) / "text_vocab.json")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config: WorldModelConfig = ckpt["config"]
    model = BeliefWorldModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded model from {args.checkpoint}", flush=True)

    for split in args.splits:
        print(f"Extracting {split}...", flush=True)
        ds = BeliefWMDataset(
            args.features_dir, split, args.metadata_dir,
            phone_vocab, text_vocab,
            evidence_type=args.evidence_type,
            max_examples=args.max_examples,
        )
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=BeliefWMCollator(),
            pin_memory=True,
        )
        count = extract_agent_features(model, loader, phone_vocab, device, output_dir, split)
        print(f"  {split}: {count} utterances → {output_dir / split}", flush=True)

    print(f"Extraction complete → {output_dir}")


if __name__ == "__main__":
    main()
