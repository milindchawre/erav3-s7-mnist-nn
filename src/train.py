import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from model import FirstModel, SecondModel, ThirdModel
from utils import load_data
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau

losses_train = []
losses_test = []
accuracy_train = []
accuracy_test = []

def execute_training(model, device, data_loader_train, optimizer, epoch_num):
    model.train()
    progress_bar = tqdm(data_loader_train)
    correct_predictions = 0
    total_processed = 0
    for batch_index, (inputs, labels) in enumerate(progress_bar):
        # Move data to device
        inputs, labels = inputs.to(device), labels.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(inputs)

        # Compute loss
        loss_value = F.nll_loss(predictions, labels)
        losses_train.append(loss_value)

        # Backward pass
        loss_value.backward()
        optimizer.step()

        # Update progress bar
        predicted_classes = predictions.argmax(dim=1, keepdim=True)
        correct_predictions += predicted_classes.eq(labels.view_as(predicted_classes)).sum().item()
        total_processed += len(inputs)

        progress_bar.set_description(desc=f'Loss={loss_value.item()} Batch_id={batch_index} Accuracy={100 * correct_predictions / total_processed:0.2f}')
        accuracy_train.append(100 * correct_predictions / total_processed)

    return 100 * correct_predictions / total_processed  # Return final training accuracy

def execute_testing(model, device, data_loader_test):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for inputs, labels in data_loader_test:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            total_loss += F.nll_loss(output, labels, reduction='sum').item()  # Sum up batch loss
            predicted_classes = output.argmax(dim=1, keepdim=True)
            correct_predictions += predicted_classes.eq(labels.view_as(predicted_classes)).sum().item()

    average_loss = total_loss / len(data_loader_test.dataset)
    losses_test.append(average_loss)

    accuracy = 100. * correct_predictions / len(data_loader_test.dataset)
    accuracy_test.append(accuracy)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        average_loss, correct_predictions, len(data_loader_test.dataset), accuracy))
    
    return average_loss, accuracy  # Return both test loss and accuracy

def main_execution(model_choice):
    # Check for device availability
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Use MPS if available
    elif torch.cuda.is_available():
        device = torch.device("cuda")  # Use CUDA if available
    else:
        device = torch.device("cpu")  # Fallback to CPU

    print(f"Using device: {device}")

    # Select model based on command line argument
    if model_choice == '1':
        model = FirstModel().to(device)
    elif model_choice == '2':
        model = SecondModel().to(device)
    elif model_choice == '3':
        model = ThirdModel().to(device)
    else:
        raise ValueError("Invalid model choice. Please select '1', '2', or '3'.")

    summary(model, input_size=(1, 28, 28))
    batch_size = 128

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Initialize the learning rate scheduler for ThirdModel
    scheduler = None
    if model_choice == '3':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    data_loader_train, data_loader_test = load_data(batch_size)

    best_accuracy_train = 0
    best_accuracy_test = 0
    best_epoch_test = 0
    
    total_epochs = 15
    for epoch in range(1, total_epochs + 1):
        print("Epoch:", epoch)
        train_accuracy = execute_training(model, device, data_loader_train, optimizer, epoch)
        test_loss, test_accuracy = execute_testing(model, device, data_loader_test)
        
        # Update best accuracies
        best_accuracy_train = max(best_accuracy_train, train_accuracy)
        if test_accuracy > best_accuracy_test:
            best_accuracy_test = test_accuracy
            best_epoch_test = epoch

        # Step the scheduler if using ThirdModel
        if scheduler:
            scheduler.step(test_loss)

    # Print final summary
    total_parameters = sum(p.numel() for p in model.parameters())
    print("\n===================")
    print("Results:")
    print(f"Parameters: {total_parameters / 1000:.1f}k")
    print(f"Best Train Accuracy: {best_accuracy_train:.2f}")
    print(f"Best Test Accuracy: {best_accuracy_test:.2f} ({best_epoch_test}th Epoch)")
    print("===================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a neural network model on the MNIST dataset.')
    parser.add_argument('--model', type=str, required=True, choices=['1', '2', '3'],
                        help='Select the model to train: 1 for FirstModel, 2 for SecondModel, 3 for ThirdModel')
    args = parser.parse_args()
    main_execution(args.model)