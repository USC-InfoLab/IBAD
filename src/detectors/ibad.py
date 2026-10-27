"""IBAD as a detector: train the forecaster, score each day by KL(observed ‖ forecast)."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.detectors.base import Detector
from src.features.visits import VisitSequences
from src.models.ibad import IBAD
from src.training import loop
from src.training.dataset import AgentSequenceDataset, collate, to_device


class IBADDetector(Detector):
    name = "ibad"
    display_name = "IBAD"
    config_path = "configs/models/ibad.yaml"
    checkpoint_ext = ".pt"

    def __init__(self, cfg, cohort, seed, device):
        super().__init__(cfg, cohort, seed, device)
        sequences = cohort.sequences
        self.model = IBAD(
            n_bundles=sequences.n_bundles,
            n_templates=sequences.n_templates,
            n_agents=sequences.n_agents,
            **cfg["model"],
        ).to(device)
        print(f"  IBAD: {sum(p.numel() for p in self.model.parameters()):,} parameters")

    def fit(self) -> dict:
        training = self.cfg["training"]
        dataset = AgentSequenceDataset(self.cohort.sequences, self.cfg["inputs"])
        common = dict(batch_size=training["batch_size"], collate_fn=collate,
                      num_workers=2, pin_memory=True)
        fit = loop.train(
            self.model,
            DataLoader(dataset, shuffle=True, **common),
            DataLoader(dataset, shuffle=False, **common),
            self.device,
            epochs=training["epochs"], lr=training["lr"],
            weight_decay=training["weight_decay"],
            early_stop=training["early_stop"], patience=training["patience"],
        )
        return {"epochs_ran": fit["epochs_ran"], "best_val_kl": fit["best_val"]}

    @torch.no_grad()
    def score(self, sequences: VisitSequences) -> np.ndarray:
        """KL(observed mixture ‖ forecast) for every day with a day before it."""
        self.model.eval()
        batch_size = self.cfg["training"]["batch_size"]
        dataset = AgentSequenceDataset(sequences, self.cfg["inputs"])
        grid = np.full((sequences.n_agents, sequences.n_days - 1), np.nan, dtype=np.float32)
        for start in range(0, sequences.n_agents, batch_size):
            rows = range(start, min(start + batch_size, sequences.n_agents))
            batch = to_device(collate([dataset[row] for row in rows]), self.device)
            kl = self.model.anomaly_scores(
                batch["arrival"], batch["departure"], batch["bundle_id"],
                batch["visit_mask"], batch["day_ts"], batch["theta"], batch["agent_idx"],
                batch["dow"],
            ).cpu().numpy()
            lengths = batch["day_mask"].sum(dim=1).cpu().numpy().astype(int)
            for offset, length in enumerate(lengths):
                if length >= 2:
                    grid[start + offset, :length - 1] = kl[offset, :length - 1]
        return grid

    def save(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: Path) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
