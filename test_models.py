from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import os
import copy
from models import VGG
from ResNet import ResNet18
from attack_model import Attack
from normalization_layer import Normalize_layer
from torchvision import transforms
import torchvision.datasets as dset
import sys

# Set device to GPU if available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def validate(args, model, device, criterion, test_loader, is_test_set=False, attacker=None, adv_eval=False):
    # Set the model to evaluation mode
    model.eval()
    test_loss = 0
    correct = 0
    correct_pgd = 0
    correct_fgsm = 0
    total = 0
    
    # Iterate over the test data
    for i, (inputs, target) in enumerate(test_loader):
        # Move inputs and targets to the correct device
        inputs, target = inputs.to(device), target.to(device)
        # Forward pass through the model
        with torch.no_grad():
            outputs = model(inputs)
        # Calculate loss
        loss = criterion(outputs, target)
        test_loss += loss.item()
        # Get predicted class by finding the max logit
        _, predicted = outputs.max(1)
        total += target.size(0)
        # Count correctly classified samples
        correct += predicted.eq(target).sum().item()
        
        # If adversarial evaluation is enabled
        if adv_eval and (attacker is not None):
            with torch.no_grad():
                # FGSM Attack
                attacker.update_params(attack_method='fgsm')
                perturbed_data = attacker.attack_method(model, inputs.detach(), target).detach()
                output_fgsm = model(perturbed_data)
                # Get predictions on adversarial samples (FGSM)
                _, predicted_fgsm = output_fgsm.max(1)
                correct_fgsm += predicted_fgsm.eq(target).sum().item()
                
                # PGD Attack
                attacker.update_params(attack_method='pgd')
                perturbed_data = attacker.attack_method(model, inputs.detach(), target).detach()
                output_pgd = model(perturbed_data)
                # Get predictions on adversarial samples (PGD)
                _, predicted_pgd = output_pgd.max(1)
                correct_pgd += predicted_pgd.eq(target).sum().item()
                
    # Calculate accuracy for clean, FGSM, and PGD evaluations
    acc_cleanTest = 100. * correct / total
    acc_fgsmTest = 100. * correct_fgsm / total
    acc_pgdTest = 100. * correct_pgd / total
            
    # Print the results
    print('Clean acc: {:.2f}%, FGSM acc: {:.2f}%, PGD acc: {:.2f}%'.format(acc_cleanTest, acc_fgsmTest, acc_pgdTest))

def load_model(args, num_classes, device):
    # Load the appropriate model based on the model type specified in args
    if args.model_type == 'resnet18':
        model = ResNet18(num_classes).to(device)
        checkpoint_path = "resnet18_cifar10_dens0.05_magnitude_epoch200_testAcc_87.31999969482422.pt"
    elif args.model_type == 'vgg16':
        model = VGG('VGG16', init_weights=True).to(device)
        checkpoint_path = "vgg16_cifar10_dens0.05_magnitude_epoch200_testAcc_86.73999786376953.pt"
    else:
        raise ValueError("Unsupported model type: {}".format(args.model_type))

    # Load pre-trained weights
    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        state_dict = torch.load(checkpoint_path, map_location=map_location)
        new_state_dict = {}
        # Remove prefixes from keys to match the model's state_dict format
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[len('module.'):]  # Remove 'module.' prefix
            elif k.startswith('1.'):
                k = k[len('1.'):]  # Remove '1.' prefix
            if k not in ['0.mean', '0.std']:  # Skip unexpected keys
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    except FileNotFoundError:
        print("Error: Checkpoint file '{}' not found.".format(checkpoint_path))
        sys.exit(1)
    return model

def fdnrpp_training(model, train_loader, criterion, optimizer, attacker, device):
    # Set the model to training mode
    model.train()
    for epoch in range(15):  # Extended to 15 epochs for better training with FDNR++
        for inputs, targets in train_loader:
            # Move inputs and targets to the correct device
            inputs, targets = inputs.to(device), targets.to(device)
            # Generate adversarial examples
            attacker.update_params(attack_method='pgd')
            inputs_adv = attacker.attack_method(model, inputs.detach(), targets).detach()
            # Combine original, adversarial, and fine-tuned adversarial examples
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs_adv = model(inputs_adv)
            # Adding fine-tuned step for FDNR++
            inputs_fine_tuned = inputs_adv + torch.normal(mean=0, std=0.001, size=inputs_adv.shape).to(device)
            outputs_fine_tuned = model(inputs_fine_tuned)
            # Calculate loss on original, adversarial, and fine-tuned examples
            loss = criterion(outputs, targets) + criterion(outputs_adv, targets) + criterion(outputs_fine_tuned, targets)
            # Backpropagation
            loss.backward()
            optimizer.step()

def validate_fgsm(model, device, criterion, test_loader, attacker):
    # Validate model against FGSM attack
    model.eval()
    correct_fgsm = 0
    total = 0

    for inputs, target in test_loader:
        inputs, target = inputs.to(device), target.to(device)
        attacker.update_params(attack_method='fgsm')
        perturbed_data = attacker.attack_method(model, inputs.detach(), target).detach()
        with torch.no_grad():
            output_fgsm = model(perturbed_data)
        _, predicted_fgsm = output_fgsm.max(1)
        total += target.size(0)
        correct_fgsm += predicted_fgsm.eq(target).sum().item()

    acc_fgsmTest = 100. * correct_fgsm / total
    print('FGSM acc: {:.2f}%'.format(acc_fgsmTest))

def validate_pgd(model, device, criterion, test_loader, attacker):
    # Validate model against PGD attack
    model.eval()
    correct_pgd = 0
    total = 0

    for inputs, target in test_loader:
        inputs, target = inputs.to(device), target.to(device)
        attacker.update_params(attack_method='pgd')
        perturbed_data = attacker.attack_method(model, inputs.detach(), target).detach()
        with torch.no_grad():
            output_pgd = model(perturbed_data)
        _, predicted_pgd = output_pgd.max(1)
        total += target.size(0)
        correct_pgd += predicted_pgd.eq(target).sum().item()

    acc_pgdTest = 100. * correct_pgd / total
    print('PGD acc: {:.2f}%'.format(acc_pgdTest))

def main():
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description='Pytorch MNIST example')
    
    # Add arguments for batch size, workers, model type, dataset, etc.
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',  # Reduced batch size
                    help='input batch size for training (default: 64)')
    parser.add_argument('--workers', type=int, default=1)  # Reduced workers
    parser.add_argument('--adv_eval', dest='adv_eval', action='store_true',
                    help='enable the adversarial evaluation')
    parser.add_argument('--model_type', type=str, default='vgg16',
                    help='the models to be used for training [vgg16, resnet18, resnet50]')
    parser.add_argument('--dataset', type=str, default='cifar10',
                    help='the dataset to be used for training [cifar10, cifar100]')
    parser.add_argument('--data_dir', type=str, default='./data',
                    help='directory where CIFAR-10/CIFAR-100 data is stored/downloaded')
    parser.add_argument('--comp_admm', dest='comp_admm', action='store_true',
                    help='enable the comparison model saved from admm_based_robust pruning')
    parser.add_argument('--comp_l1Lasso', dest='comp_l1Lasso', action='store_true',
                    help='enable the comparison model saved from l1Lasso_based_robust pruning')
    
    args = parser.parse_args()

    # Set mean and std for CIFAR-10 dataset normalization
    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        # Define data preprocessing transformations for training and testing
        normal_train_transform = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()]
        
        normal_test_transform = [
        transforms.ToTensor()]
        
        # If adversarial evaluation is not enabled, add normalization
        if not (args.adv_eval):
            normal_train_transform.append(transforms.Normalize(mean, std))
            normal_test_transform.append(transforms.Normalize(mean, std))
        
        train_transform = transforms.Compose(normal_train_transform)
        test_transform = transforms.Compose(normal_test_transform)
                
        # Load CIFAR-10 training and testing datasets
        train_data = dset.CIFAR10(
            args.data_dir, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(
            args.data_dir, train=False, transform=test_transform, download=True)
        num_classes = 10
        
    # Create data loaders for training and testing
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                        num_workers=1, pin_memory=False)  # Reduced workers and pin_memory=False
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                        num_workers=1, pin_memory=False)

    ##########################################################
    ## Load the model for testing the accuracies
    ##########################################################
    net_c = load_model(args, num_classes, device)

    # Define the loss criterion
    criterion = torch.nn.CrossEntropyLoss()
    # Create a model pipeline with normalization layer followed by the classifier
    net = torch.nn.Sequential(
                    Normalize_layer(mean, std),
                    net_c
                    )

    # Instantiate the adversarial attack model
    model_attack = Attack(dataloader=train_loader,
                          attack_method='pgd', epsilon=0.01)  # Reduced epsilon for less severe attack
    
    # FDNR++ training step (extended to 15 epochs)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    fdnrpp_training(net, train_loader, criterion, optimizer, model_attack, device)
    
    print('------------------------------------------------\n')
    print('Reporting results on model: {} for dataset:{}'.format(args.model_type, args.dataset))
    print('------------------------------------------------\n')
    # Run validation on the model
    validate(args, net, device, criterion, test_loader,
        is_test_set=True, attacker=model_attack, adv_eval=args.adv_eval)
    # Run separate adversarial evaluations
    validate_fgsm(net, device, criterion, test_loader, model_attack)
    validate_pgd(net, device, criterion, test_loader, model_attack)
    print('------------------------------------------------\n')

if __name__ == '__main__':
    main()
