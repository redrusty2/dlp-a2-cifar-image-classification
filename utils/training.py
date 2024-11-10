from typing import Any, Callable
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
from tqdm.notebook import tqdm
from utils.dataset import mixup_criterion, mixup_data
from utils.early_stopping import EarlyStopping
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_model(model, loader: DataLoader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print("Accuracy: {:.2f} %".format(accuracy))


def validate_model(model, criterion, loader: DataLoader):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        val_acc = 0
        total = 0
        correct = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_acc = correct / total

    return val_loss, val_acc


class Trainer:
    def __init__(
        self,
        name: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        early_stop: EarlyStopping | None,
        scheduler: Callable[[Any], None] | None = None,
        mixup: bool = False,
    ):
        self.name = name
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.early_stop = early_stop
        self.scheduler = scheduler
        self.mixup = mixup

        self.epochs = 0
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []

    def fit(self, num_epochs: int):
        for epoch in tqdm(range(self.epochs, self.epochs + num_epochs)):
            epoch_loss = 0
            epoch_acc = 0
            num_batches = len(self.train_loader)

            self.model.train()
            for _, (images, labels) in enumerate(self.train_loader):
                images = images.to(device)
                labels = labels.to(device)

                if self.mixup:
                    images, labels_a, labels_b, lam = mixup_data(images, labels)
                    outputs = self.model(images)
                    loss = mixup_criterion(
                        self.criterion, outputs, labels_a, labels_b, lam
                    )
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                epoch_loss += loss.item()
                epoch_acc += (outputs.argmax(1) == labels).float().mean().item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            avg_loss = epoch_loss / num_batches
            avg_acc = epoch_acc / num_batches
            self.train_losses.append(avg_loss)
            self.train_accs.append(avg_acc)

            val_loss, val_acc = validate_model(
                self.model, self.criterion, self.val_loader
            )
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            if self.early_stop is not None:
                self.early_stop(val_loss, self.model)

            lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch [{epoch+1}/{num_epochs}] {"M" if self.mixup else ""} LR: {lr:.5f}  T_Loss: {avg_loss:.4f}   "
                f"V_Loss: {val_loss:.4f}   T_Acc: {avg_acc:.2f}   V_Acc: {val_acc:.2f}"
            )

            if self.scheduler is not None:
                self.scheduler(val_loss)

            if self.early_stop is not None and self.early_stop.should_stop:
                print("Early stopping")
                break

    def plot(self):
        plt.title(f"{self.name} - Loss")
        plt.plot(self.train_losses, label="Training")
        plt.plot(self.val_losses, label="Validation")
        plt.legend()
        plt.xlabel(r"Epochs")
        plt.ylabel(r"Loss")
        plt.savefig(f"docs/plots/{self.name}-loss.pdf", format="pdf", dpi=300)
        plt.show()

        plt.title(f"{self.name} - Accuracy")
        plt.plot(self.train_accs, label="Training")
        plt.plot(self.val_accs, label="Validation")
        plt.legend()
        plt.xlabel(r"Epochs")
        plt.ylabel(r"Accuracy %")
        plt.savefig(f"docs/plots/{self.name}-acc.pdf", format="pdf", dpi=300)
        plt.show()

    def validate_best_checkpoint(self):
        best_checkpoint = torch.load(self.early_stop.path)
        val_loss, val_acc = validate_model(
            best_checkpoint, self.criterion, self.val_loader
        )
        print(f"Best checkpoint - Loss: {val_loss:.4f}   Accuracy: {val_acc:.2f}")
        return val_loss, val_acc

    def final_test(self, test_loader: DataLoader):
        best_checkpoint = torch.load(self.early_stop.path)
        test_model(best_checkpoint, test_loader)
