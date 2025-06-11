from transformers import TrainerCallback
import time
from datetime import datetime
from utils.logger import get_logger as logger


class TrainingProgressCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None
        self.last_log_time = None
        self.total_steps = None
        self.current_step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        self.training_started = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.total_steps = state.max_steps
        logger.info(f"Starting training for {self.total_steps} total steps")
        logger.info(f"Training will run for {args.num_train_epochs} epochs")
        self.training_started = True

    def on_step_end(self, args, state, control, **kwargs):
        self.current_step = state.global_step

        # Log every 10 steps or if 30 seconds have passed
        current_time = time.time()
        if (self.current_step % 10 == 0) or (current_time - self.last_log_time > 30):
            elapsed = current_time - self.start_time
            steps_per_second = self.current_step / elapsed
            remaining_steps = self.total_steps - self.current_step
            eta = remaining_steps / steps_per_second if steps_per_second > 0 else 0

            # Calculate progress percentage
            progress = (self.current_step / self.total_steps) * 100

            # Get current learning rate
            current_lr = kwargs.get("optimizer").param_groups[0]["lr"]

            # Get current loss if available
            current_loss = (
                state.log_history[-1].get("loss", "N/A") if state.log_history else "N/A"
            )

            # Format loss string based on type
            if isinstance(current_loss, float):
                self.best_loss = min(self.best_loss, current_loss)
                loss_str = f"{current_loss:.4f}"
                best_loss_str = f"{self.best_loss:.4f}"
            else:
                loss_str = str(current_loss)
                best_loss_str = f"{self.best_loss:.4f}"

            logger.info(
                f"Step {self.current_step}/{self.total_steps} ({progress:.1f}%) | "
                f"Loss: {loss_str} | Best Loss: {best_loss_str} | "
                f"LR: {current_lr:.2e} | Speed: {steps_per_second:.1f} steps/s | "
                f"ETA: {datetime.fromtimestamp(current_time + eta).strftime('%H:%M:%S')}"
            )
            self.last_log_time = current_time

    def on_epoch_end(self, args, state, control, **kwargs):
        self.epoch += 1
        logger.info(f"Completed epoch {self.epoch}/{args.num_train_epochs}")

        # Log epoch statistics
        if state.log_history:
            epoch_loss = state.log_history[-1].get("loss", "N/A")
            logger.info(f"Epoch {self.epoch} final loss: {epoch_loss}")

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        logger.info(f"Training completed in {total_time / 3600:.2f} hours")
        logger.info(
            f"Final loss: {state.log_history[-1].get('loss', 'N/A') if state.log_history else 'N/A'}"
        )
        logger.info(f"Best loss achieved: {self.best_loss:.4f}")
