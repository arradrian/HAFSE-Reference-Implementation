"""CNN-LSTM backbone and HAFSE target-model construction.

The architecture follows Fig. 4 and Table X of the manuscript:

    Conv1D block 1 : Conv1D(64, k=7)  + BN + MaxPooling1D(2) + Dropout(0.2)
    Conv1D block 2 : Conv1D(128, k=5) + BN + MaxPooling1D(2) + Dropout(0.2)
    Conv1D block 3 : Conv1D(256, k=3) + BN + MaxPooling1D(2) + Dropout(0.2)
    LSTM block 1   : LSTM(128, return_sequences=True) + BN + Dropout(0.2)
    LSTM block 2   : LSTM(64,  return_sequences=True) + BN + Dropout(0.2)
    LSTM block 3   : LSTM(32,  return_sequences=False) + BN + Dropout(0.2)
    Head           : Dense(128, ReLU) + BN + Dropout(0.5) + Dense(C, softmax)

Transfer to a target domain (Sec. 4.1.1, Table X):

* Conv1D block 1 is the target-specific input adapter. Its kernel has shape
  (7, d, 64) and depends on the number of input variables d, so it is always
  built for the target dimensionality, randomly initialized and trainable.
* Conv1D blocks 2-3 and LSTM blocks 1-3 do not depend on d. They are
  initialized with the pretrained source parameters and form the transferred
  representation backbone.
* N (1..5) selects how many transferred parametric layers are unfrozen,
  counted backwards from the output:
      N=1: LSTM-3
      N=2: LSTM-2, LSTM-3
      N=3: LSTM-1, LSTM-2, LSTM-3
      N=4: + Conv1D-3
      N=5: + Conv1D-2
  Each batch-normalization layer follows the status of the layer before it;
  pooling and dropout layers have no parameters and are not counted.
* M (0..5) target-specific expansion blocks Dense(expansion_units, ReLU) are
  inserted between LSTM-3 and the classification head (M=0 is the baseline
  configuration M0 used while N is selected).
* The source classification head (Dense-128 and softmax) is discarded; a new
  target-specific head is randomly initialized and trained.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers

CONV_FILTERS = (64, 128, 256)
CONV_KERNELS = (7, 5, 3)
LSTM_UNITS = (128, 64, 32)
DENSE_UNITS = 128

# Transferred parametric layers, ordered from the output backwards (defines N).
TRANSFER_ORDER = ("rep_lstm3", "rep_lstm2", "rep_lstm1", "rep_conv3", "rep_conv2")
# Batch-normalization layer that follows each transferred layer.
BN_OF = {
    "rep_conv2": "rep_bn_conv2",
    "rep_conv3": "rep_bn_conv3",
    "rep_lstm1": "rep_bn_lstm1",
    "rep_lstm2": "rep_bn_lstm2",
    "rep_lstm3": "rep_bn_lstm3",
}
# Every layer whose weights are copied from the source model.
TRANSFERRED_LAYERS = tuple(TRANSFER_ORDER) + tuple(BN_OF.values())
# Target-specific input adapter (never transferred, always trainable).
ADAPTER_LAYERS = ("adapter_conv1", "adapter_bn_conv1")


def _backbone(x, conv1_prefix: str):
    """Shared CNN-LSTM backbone. Returns the last LSTM block output."""
    for i, (f, k) in enumerate(zip(CONV_FILTERS, CONV_KERNELS), start=1):
        if i == 1:
            conv_name, bn_name = f"{conv1_prefix}_conv1", f"{conv1_prefix}_bn_conv1"
        else:
            conv_name, bn_name = f"rep_conv{i}", f"rep_bn_conv{i}"
        x = layers.Conv1D(f, k, padding="same", activation="relu", name=conv_name)(x)
        x = layers.BatchNormalization(name=bn_name)(x)
        x = layers.MaxPooling1D(2, padding="same", name=f"pool{i}")(x)
        x = layers.Dropout(0.2, name=f"drop_conv{i}")(x)
    for i, u in enumerate(LSTM_UNITS, start=1):
        x = layers.LSTM(u, return_sequences=(i < len(LSTM_UNITS)), name=f"rep_lstm{i}")(x)
        x = layers.BatchNormalization(name=f"rep_bn_lstm{i}")(x)
        x = layers.Dropout(0.2, name=f"drop_lstm{i}")(x)
    return x


def _head(x, n_classes: int, prefix: str):
    x = layers.Dense(DENSE_UNITS, activation="relu", name=f"{prefix}_dense128")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn_dense128")(x)
    x = layers.Dropout(0.5, name=f"{prefix}_drop_dense128")(x)
    return layers.Dense(n_classes, activation="softmax", name=f"{prefix}_classifier")(x)


def build_source_model(timesteps: int, n_features: int, n_classes: int) -> Model:
    """Source CNN-LSTM model trained from scratch on the source domain."""
    inp = layers.Input((timesteps, n_features), name="source_input")
    x = _backbone(inp, conv1_prefix="source")
    out = _head(x, n_classes, prefix="source")
    return Model(inp, out, name="source_cnn_lstm")


def unfrozen_layers(n_unfrozen: int) -> tuple[str, ...]:
    """Names of the transferred parametric layers unfrozen for a given N."""
    if not 0 <= n_unfrozen <= len(TRANSFER_ORDER):
        raise ValueError(f"N must be in [0, {len(TRANSFER_ORDER)}], got {n_unfrozen}.")
    return TRANSFER_ORDER[:n_unfrozen]


def build_target_model(
    source_model: Model,
    timesteps: int,
    target_features: int,
    target_classes: int,
    n_unfrozen: int,
    n_new_layers: int,
    expansion_units: int = 64,
) -> Model:
    """Build the HAFSE target model for a structural configuration (N, M)."""
    inp = layers.Input((timesteps, target_features), name="target_input")
    x = _backbone(inp, conv1_prefix="adapter")
    for i in range(n_new_layers):
        x = layers.Dense(expansion_units, activation="relu", name=f"expansion_{i + 1}")(x)
    out = _head(x, target_classes, prefix="target")
    model = Model(inp, out, name=f"target_N{n_unfrozen}_M{n_new_layers}")

    # Copy pretrained parameters into the transferred backbone layers.
    for name in TRANSFERRED_LAYERS:
        src_w = source_model.get_layer(name).get_weights()
        tgt = model.get_layer(name)
        if [w.shape for w in src_w] != [w.shape for w in tgt.get_weights()]:
            raise ValueError(f"Shape mismatch in transferred layer {name}.")
        tgt.set_weights(src_w)

    # Freeze / unfreeze transferred layers according to N.
    unfrozen = set(unfrozen_layers(n_unfrozen))
    for name in TRANSFER_ORDER:
        trainable = name in unfrozen
        model.get_layer(name).trainable = trainable
        model.get_layer(BN_OF[name]).trainable = trainable

    # The input adapter, expansion blocks and head are always trainable.
    return model


def transfer_summary(model: Model) -> list[dict]:
    """Layer-wise summary (shapes, parameter counts, status) used for Table X."""
    rows = []
    for layer in model.layers:
        if not layer.weights:
            continue
        if layer.name in TRANSFERRED_LAYERS:
            status = "transferred"
        elif layer.name in ADAPTER_LAYERS:
            status = "target-specific input adapter (random init)"
        else:
            status = "new (random init)"
        rows.append({
            "layer": layer.name,
            "weight_shapes": [tuple(w.shape) for w in layer.weights],
            "params": int(sum(int(tf.size(w)) for w in layer.weights)),
            "trainable": bool(layer.trainable),
            "status": status,
        })
    return rows
