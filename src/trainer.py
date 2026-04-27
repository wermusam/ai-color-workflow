"""Trainer class for the color grading model."""

from pathlib import Path

import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import PairImageDataset
from src.model import ColorGradeNet


class Trainer:
    """Trains a ColorGradeNet on ungraded/graded image pairs."""

    def __init__(
        self,
        ungraded_dir: str,
        graded_dir: str,
        output_path: str,
        loss_plot_path: str = "models/loss_curve.html",
        epochs: int = 100,
        batch_size: int = 4,
        learning_rate: float = 1e-3,
    ) -> None:
        self.ungraded_dir = Path(ungraded_dir)
        self.graded_dir = Path(graded_dir)
        self.output_path = Path(output_path)
        self.loss_plot_path = Path(loss_plot_path)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_history: list[float] = []

    def run(self) -> float:
        """Train the model and save weights. Returns final average loss."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        dataset = PairImageDataset(
            ungraded_dir=self.ungraded_dir,
            graded_dir=self.graded_dir,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        model = ColorGradeNet()
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            total_loss = 0.0
            num_batches = 0

            for ungraded_batch, graded_batch in dataloader:
                prediction = model(ungraded_batch)
                loss = loss_fn(prediction, graded_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            self.loss_history.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}  loss={avg_loss:.6f}")

        torch.save(model.state_dict(), self.output_path)
        print(f"Saved model to {self.output_path}")

        self._save_loss_plot()

        return self.loss_history[-1]

    def _save_loss_plot(self) -> None:
        """Save a Plotly chart of the training loss curve."""
        epochs = list(range(1, len(self.loss_history) + 1))
        starting_loss = self.loss_history[0]
        final_loss = self.loss_history[-1]
        improvement = starting_loss / final_loss

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=self.loss_history,
                mode="lines",
                name="Training loss",
                line=dict(color="#5dade2", width=2),
                hovertemplate="Epoch %{x}<br>Loss: %{y:.6f}<extra></extra>",
            )
        )
        fig.update_layout(
            title=dict(
                text=(
                    "<b>Training Loss Over Time</b><br>"
                    "<sup>Lower is better. The model learns to predict graded images "
                    "from ungraded inputs.</sup>"
                ),
                x=0.5,
                xanchor="center",
            ),
            xaxis_title="Epoch (one full pass through all training pairs)",
            yaxis_title="Mean Squared Error (per-pixel error, 0 = perfect)",
            template="plotly_dark",
            annotations=[
                dict(
                    text=(
                        f"Started at {starting_loss:.4f}, "
                        f"ended at {final_loss:.6f} "
                        f"({improvement:.0f}× improvement)"
                    ),
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.20,
                    showarrow=False,
                    font=dict(size=12, color="#a0a0a0"),
                )
            ],
            margin=dict(b=100),
        )

        self.loss_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(self.loss_plot_path)
        print(f"Saved loss plot to {self.loss_plot_path}")