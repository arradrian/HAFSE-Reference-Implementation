"""Model construction demo: prints the layer-wise transfer table (Table X)
for every value of N, showing that each N yields a distinct configuration."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import build_source_model, build_target_model, transfer_summary, unfrozen_layers
from src.reproducibility import set_global_seed

set_global_seed(0)
T = 32
source = build_source_model(T, n_features=52, n_classes=25)      # e.g. TE: d = 52
print(f"Source model parameters: {source.count_params()}")
for N in range(1, 6):
    target = build_target_model(source, T, target_features=6, target_classes=8,
                                n_unfrozen=N, n_new_layers=3)   # e.g. TD1: d = 6
    trainable = sum(int(w.shape.num_elements()) for w in target.trainable_weights)
    print(f"N={N}: unfrozen transferred layers = {list(unfrozen_layers(N))}; "
          f"trainable parameters = {trainable}")
print("\nTable X for N=2, M=3 (TD1-like dimensions):")
for row in transfer_summary(build_target_model(source, T, 6, 8, 2, 3)):
    print(f"  {row['layer']:<22} {str(row['weight_shapes'][0]):<16} params={row['params']:>7}  "
          f"trainable={row['trainable']!s:<5} {row['status']}")
