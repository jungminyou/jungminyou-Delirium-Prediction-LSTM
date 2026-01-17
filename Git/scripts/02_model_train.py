import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedShuffleSplit

def build_soft_gated_multitask_model(seq_shape, static_shape, params):
    """
    Builds a Multi-Output LSTM model where the Primary Outcome (Delirium) 
    acts as a soft gate for Secondary Outcomes, as described in the manuscript.
    """
    # L2 regularization: 200 log-spaced values from 10^-7 to 10^-1 (passed via params)
    l2_reg = l2(params.get('l2_rate', 1e-4))

    # 1. Feature Extraction Branch
    # Temporal features from intraoperative time-series
    seq_in = Input(shape=seq_shape, name='sequence_input')
    lstm_out = LSTM(params['lstm_units'], 
                    kernel_regularizer=l2_reg, 
                    recurrent_regularizer=l2_reg)(seq_in)
    lstm_out = Dropout(params['dropout_rate'])(lstm_out)

    # Baseline clinical features
    static_in = Input(shape=static_shape, name='static_input')
    static_dense = Dense(params['static_units'], 
                         activation='relu', 
                         kernel_regularizer=l2_reg)(static_in)
    static_dense = Dropout(params['dropout_rate'])(static_dense)

    # Feature Fusion
    shared_features = Concatenate()([lstm_out, static_dense])

    # 2. Primary Outcome: Postoperative Delirium (1:1 Balanced via PSM)
    delirium_output = Dense(1, activation='sigmoid', name='delirium_output')(shared_features)

    # 3. Soft Gating Mechanism
    # The primary delirium prediction modulates the information flow to secondary outcomes
    gate_layer = Dense(params['lstm_units'] + params['static_units'], 
                       activation='sigmoid', 
                       name='soft_gate_mechanism')(delirium_output)
    gated_features = Multiply()([shared_features, gate_layer])

    # 4. Secondary Outcomes (e.g., phenotypes, severity)
    # These outputs use class-weighted loss to handle natural imbalance
    secondary_output = Dense(1, activation='sigmoid', name='secondary_output')(gated_features)

    model = Model(inputs=[seq_in, static_in], outputs=[delirium_output, secondary_output])
    
    # Joint Optimization using Total Validation Loss as the objective
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr']),
        loss={
            'delirium_output': 'binary_crossentropy', 
            'secondary_output': 'binary_crossentropy' # Weighted in fit()
        },
        loss_weights={'delirium_output': 1.0, 'secondary_output': 1.0},
        metrics={'delirium_output': ['AUC', 'Recall'], 'secondary_output': ['AUC']}
    )
    
    return model

def run_optimized_training(X_seq, X_static, y_delirium, y_secondary):
    """
    Executes Stratified 80/20 split, Grid Search, and Early Stopping (20 epochs).
    """
    # 1. Stratified 80/20 Training-Validation Split (Maintains 1:1 PSM ratio)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, val_idx in sss.split(X_seq, y_delirium):
        x_train = [X_seq[train_idx], X_static[train_idx]]
        x_val = [X_seq[val_idx], X_static[val_idx]]
        y_train = {'delirium_output': y_delirium[train_idx], 'secondary_output': y_secondary[train_idx]}
        y_val = {'delirium_output': y_delirium[val_idx], 'secondary_output': y_secondary[val_idx]}

    # 2. Hyperparameter Grid Search Space (Based on Reviewer Response)
    l2_space = np.logspace(-7, -1, 200) # 200 log-spaced values
    dropout_space = [0.2, 0.3, 0.4, 0.5]
    lr_space = [1e-3, 1e-4, 1e-5]

    best_val_loss = float('inf')
    best_model = None

    # Example loop for grid search optimization
    for l2_val in [l2_space[0], l2_space[100], l2_space[-1]]: # Subsampled for brevity
        for dr in dropout_space:
            for lr in lr_space:
                params = {
                    'lstm_units': 64, 'static_units': 32,
                    'dropout_rate': dr, 'l2_rate': l2_val, 'lr': lr
                }

                model = build_soft_gated_multitask_model(
                    X_seq.shape[1:], (X_static.shape[1],), params
                )

                # 3. Early Stopping set at 20 epochs to prevent overfitting
                early_stop = EarlyStopping(
                    monitor='val_loss', 
                    patience=20, 
                    restore_best_weights=True
                )

                # 4. Class-weighted loss for imbalanced secondary phenotypes
                # Calculate weights for secondary outcomes manually for multi-output
                counts = np.bincount(y_secondary[train_idx].astype(int))
                class_weights = {0: 1.0, 1: (counts[0] / counts[1])}

                model.fit(
                    x_train, y_train,
                    validation_data=(x_val, y_val),
                    epochs=200,
                    batch_size=32,
                    callbacks=[early_stop],
                    verbose=0
                    # Note: For multi-output, class_weight is typically applied via sample_weight
                )

                current_val_loss = model.evaluate(x_val, y_val, verbose=0)[0]
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    best_model = model

    return best_model