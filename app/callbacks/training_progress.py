from transformers import TrainerCallback
import time
from datetime import datetime
from utils.logger import logger


class TrainingProgressCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None
        self.total_steps = None
        self.current_step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        self.best_eval_loss = float("inf")
        self.training_started = False

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.start_time = time.time()
            self.total_steps = state.max_steps
            logger.info(f"Starting training for {self.total_steps} total steps")
            logger.info(f"Training will run for {args.num_train_epochs} epochs")
            self.training_started = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.training_started or not state.is_world_process_zero or logs is None:
            return

        self.current_step = state.global_step
        current_loss = logs.get("loss")

        # This is a training log if it has a 'loss' and a 'learning_rate'
        if current_loss is not None and "learning_rate" in logs:
            self.best_loss = min(self.best_loss, current_loss)
            current_time = time.time()
            elapsed = current_time - self.start_time
            steps_per_second = self.current_step / elapsed if elapsed > 0 else 0
            remaining_steps = self.total_steps - self.current_step
            eta_seconds = (
                remaining_steps / steps_per_second if steps_per_second > 0 else 0
            )
            progress = (self.current_step / self.total_steps) * 100
            eta_str = datetime.fromtimestamp(time.time() + eta_seconds).strftime(
                "%H:%M:%S"
            )

            logger.info(
                f"Step {self.current_step}/{self.total_steps} ({progress:.1f}%) | "
                f"Loss: {current_loss:.4f} | Best Loss: {self.best_loss:.4f} | "
                f"LR: {logs['learning_rate']:.2e} | Speed: {steps_per_second:.1f} steps/s | "
                f"ETA: {eta_str}"
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not state.is_world_process_zero or metrics is None:
            return

        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None:
            self.best_eval_loss = min(self.best_eval_loss, eval_loss)
            logger.info(
                f"--- Evaluation --- Step {state.global_step} | "
                f"Eval Loss: {eval_loss:.4f} | "
                f"Best Eval Loss: {self.best_eval_loss:.4f}"
            )

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.epoch += 1
            logger.info(f"Completed epoch {self.epoch}/{args.num_train_epochs}")

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            total_time = time.time() - self.start_time
            logger.info(f"Training completed in {total_time / 3600:.2f} hours")
            logger.info(f"Best training loss: {self.best_loss:.4f}")
            if self.best_eval_loss != float("inf"):
                logger.info(f"Best evaluation loss: {self.best_eval_loss:.4f}")
