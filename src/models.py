from __future__ import annotations

from tensorflow.keras.layers import Dense, Input, LSTM, Masking, Multiply
from tensorflow.keras.models import Model


def build_multi_output_lstm_gated(time_steps: int, n_features: int, n_classes_typ: int, n_classes_kdr: int) -> Model:
    inp = Input(shape=(time_steps, n_features), name="input")
    h = Masking(mask_value=0.0)(inp)

    h_shared = LSTM(64, activation="tanh", return_sequences=True, name="shared_lstm_64")(h)

    out_next = Dense(1, activation="sigmoid", name="delirium_next_day")(h_shared)
    h_gated = Multiply(name="soft_gate")([h_shared, out_next])

    out_typ = Dense(n_classes_typ, activation="softmax", name="delirium_typ")(h_gated)
    out_kdr = Dense(n_classes_kdr, activation="softmax", name="K_DRS_R_98")(h_gated)

    model = Model(inputs=inp, outputs=[out_next, out_typ, out_kdr])
    model.compile(
        optimizer="adam",
        loss={
            "delirium_next_day": "binary_crossentropy",
            "delirium_typ": "sparse_categorical_crossentropy",
            "K_DRS_R_98": "sparse_categorical_crossentropy",
        },
        metrics={"delirium_next_day": "AUC"},
    )
    return model
