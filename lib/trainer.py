from typing import List, Dict, Optional
import tensorflow as tf


class Trainer:
    def __init__(self,
                 epoch: int,
                 model: tf.keras.Model,
                 train_datset: tf.data.Dataset,
                 optimizer: tf.keras.optimizers.Optimizer,
                 validation_dataset: Optional[tf.data.Dataset] = None) -> None:
        self.epoch = epoch
        self.model = model
        self.train_dataset = train_datset
        self.validation_dataset = validation_dataset
        self.optimizer = optimizer

    def run(self):
        for _ in range(self.epoch):
            sum_loss = 0.
            for batch in self.train_dataset:
                with tf.GradientTape() as tape:
                    info = self.training_step(batch)
                loss = info['loss']
                var = self.model.trainable_weights
                grad = tape.gradient(loss, var)
                self.optimizer.apply_gradients(zip(grad, var))
                sum_loss += loss.numpy().tolist()
            print(f'Training loss: {sum_loss}')
            if self.validation_dataset:
                outputs = []
                for batch in self.validation_dataset:
                    outputs.append(self.validation_step(batch))
                info = self.validation_end(outputs)

    def training_step(self, batch: List[tf.Tensor]) -> Dict[str, tf.Tensor]:
        loss = self.model(*batch)
        return {'loss': loss}

    def validation_step(self, batch: List[tf.Tensor]) -> Dict[str, tf.Tensor]:
        y = self.model.inference(*batch)
        return {'val_loss': y}

    def validation_end(self, outputs: List[Dict[str, tf.Tensor]]) -> Dict[str, tf.Tensor]:
        ...
