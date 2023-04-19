#Import all the required Libraries
#Ensure to activate your environment it is conda activate crack_detection

import streamlit as st#loading streamlit
import pandas as pd#pandas for any visualisation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import torchvision
import numpy as np
from pathlib import Path

cwd = Path.cwd()
from PIL import Image
import time
import copy
import random
import cv2
import re
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

st.title(
    'Detection of cracks in Slabs,Roads and Images using Machine Learning,Logistics Regression,CNN and Comparision of the models result'
)
st.text(f'This application will analyze if there is a crack in slab')
st.markdown("We will be analyzing the image based on our trained model")
st.markdown(
    "The current training is done through series of 100 cracked concrete slabs images and 100 uncracked images of slabs"
)

mean_nums = [0.485, 0.456, 0.406]  # mean
std_nums = [0.229, 0.224, 0.225]  #standard deviation
st.text(f'Checking the Current working directory for any directory update')
st.write(f"This is the current working directory:{cwd}")

# till is good so now we will be training the model for 100 images
st.markdown("Lets roll We load images as crack_images from our database ")
crack_images = os.listdir(
    'Positive')  # training for positive data which images has cracks
# we upload all the images with cracks in positive
st.markdown("Lets check the number of crack images ")
st.write("Number of Crack Images: ", len(crack_images))

## Visualize Random images with cracks
random_indices_c = np.random.randint(0, len(crack_images), size=4)
st.write("Random Images with Cracks")
random_images_c = np.array(crack_images)[random_indices_c.astype(int)]

f, axarr = plt.subplots(2, 2)
axarr[0,
      0].imshow(mpimg.imread(os.path.join(cwd, 'Positive',
                                          random_images_c[0])))
axarr[0,
      1].imshow(mpimg.imread(os.path.join(cwd, 'Positive',
                                          random_images_c[1])))
axarr[1,
      0].imshow(mpimg.imread(os.path.join(cwd, 'Positive',
                                          random_images_c[2])))
axarr[1,
      1].imshow(mpimg.imread(os.path.join(cwd, 'Positive',
                                          random_images_c[3])))

st.write("Now we are starting to plot here below")

# Display the plot in Streamlit
st.pyplot(f)
st.markdown("Lets check the number of crack images ")
no_crack_images = os.listdir('Negative')
st.write("Number of No Crack Images: ", len(no_crack_images))
random_indices_u = np.random.randint(0, len(no_crack_images), size=4)
st.write("Random Images with No Cracks")
random_images_u = np.array(no_crack_images)[random_indices_u.astype(int)]

f, axarr = plt.subplots(2, 2)
axarr[0,
      0].imshow(mpimg.imread(os.path.join(cwd, 'Negative',
                                          random_images_u[0])))
axarr[0,
      1].imshow(mpimg.imread(os.path.join(cwd, 'Negative',
                                          random_images_u[1])))
axarr[1,
      0].imshow(mpimg.imread(os.path.join(cwd, 'Negative',
                                          random_images_u[2])))
axarr[1,
      1].imshow(mpimg.imread(os.path.join(cwd, 'Negative',
                                          random_images_u[3])))
st.write("Now we are starting to plot here below")

# Display the plot in Streamlit
st.pyplot(f)

# working fine
#Create training folder

import os
import streamlit as st

base_dir = '/Users/vishalkumar/Desktop/ML_Project_Final/Streamlit_Project_Crack_Detection'  #setting base directory
files = (base_dir)  #os.listdir(base_dir)

positive_train = '/Users/vishalkumar/Desktop/ML_Project_Final/Streamlit_Project_Crack_Detection/train/Positive/'  # adding positive train
positive_val = base_dir + "/val/Positive/"  # adding data for validation for positives
negative_train = base_dir + "/train/Negative/"  #adding negative train dataset
negative_val = base_dir + "/val/Negative/"  #adding negative validation

positive_files = (positive_train)  #os.listdir(positive_train)
negative_files = (negative_train)  # os.listdir(negative_train)

st.write(f"Number of Positive training files: {len(positive_files)}")
st.write(f"Number of Negative training files: {len(negative_files)}")  #  display the number of files in the training dataset for a crack detection project.

device = torch.device("cuda" if torch.cuda.is_available() else
                      "cpu")  # trying to use gpu if possible
st.write("Is it GPU or CPU ? : ", device)

mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]  # normalization or standardization. # mean_nums and std_nums appear a list of means and standard deviations for three channels in an image. 

## Define data augmentation and transforms
chosen_transforms = {  # augmentation and transformation operations to be applied to the training and validation datasets
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=227), # randomly crops a square patch from the input image and resizes it to the specified size (227x227 in this case)
        transforms.RandomRotation(degrees=10), # RandomRotation: randomly rotates the image by a specified number of degrees (10 in this case)
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),# converts the image to a PyTorch tensor
        transforms.Normalize(mean_nums, std_nums)
    ]),
    'val':
    transforms.Compose([ # Resize: resizes the image to the specified size (227x227 in this case)
        transforms.Resize(227),
        transforms.CenterCrop(227), 
        transforms.ToTensor(),
        transforms.Normalize(mean_nums, std_nums)
    ]),
}


def load_dataset(format, batch_size):
    data_path = os.path.join(cwd, format)
    dataset = datasets.ImageFolder(root=data_path,
                                   transform=chosen_transforms[format])
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             num_workers=0,
                             shuffle=True)
    return data_loader, len(dataset), dataset.classes


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device) updated at the top

train_loader, train_size, class_names = load_dataset('train', 3)
st.write("Train Data Set size is: ", train_size)
st.write("Class Names are: ", class_names)

try: # load a batch of data from the training dataset using the PyTorch DataLoader class.
    inputs, classes = next(iter(train_loader))
    st.write("Input shape:", inputs.shape)
    st.write("Classes shape:", classes.shape)
except Exception as e:
    st.write("Error loading data:", str(e))

import io


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([mean_nums])
    std = np.array([std_nums])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    st.image(buf, use_column_width=True)
    plt.clf()
    plt.cla()
    plt.close()

    # Grab some of the training data to visualize


import matplotlib.pyplot as plt
import numpy as np
import torchvision

# Get some of the training data to visualize
# Grab some of the training data to visualize
inputs, classes = next(iter(train_loader))
# class_names = chosen_datasets['train'].classes
# Now we construct a grid from batch
out = torchvision.utils.make_grid(inputs)

idx_to_class = {0: 'Negative', 1: 'Positive'}
plt.figure(figsize=(20, 10))
imshow(out, title=[x.data.numpy() for x in classes])

# create a class for Logi regrression
class LogisticRegression(nn.Module):
    def __init__(self, in_features, out_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        out = self.linear(x)
        return out
    
# Define the logistic regression model
logistic_model = LogisticRegression(in_features=227*227*3, out_features=2)
logistic_model = logistic_model.to(device)

## Load pretrained model
resnet50 = models.resnet50(pretrained=True)

# Freeze model parameters
for param in resnet50.parameters():
    param.requires_grad = False

## Change the final layer of the resnet model
# Change the final layer of ResNet50 Model for Transfer Learning
fc_inputs = resnet50.fc.in_features

resnet50.fc = nn.Sequential(nn.Linear(fc_inputs, 128), nn.ReLU(),
                            nn.Dropout(0.4), nn.Linear(128, 2))

#  but i dont have gpu will use for gpu
resnet50 = resnet50.to(device)

from torchsummary import summary

st.write(summary(resnet50, (3, 227, 227)))

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(resnet50.fc.parameters(), lr=0.001, momentum=0.9) #
#optimizer = optim.SGD(logistic_model.parameters(), lr=0.001, momentum=0.9) # SGD try with logistic regression model and Adam Can be tried 

optimizer = optim.Adam(logistic_model.parameters(), lr=0.001) # adam  update April 18 6:57 am revert to SDG if results are bad
# results are same not much dif
#update the learning rate from 0.001 to 0.01 date 13 April 2023s
# bad results does not identify cracks revert to 0.001 13 april 2023
#multiple mainly Stochastic gradient descent instead of adam as its better than adam
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

dataloaders = {}
dataset_sizes = {}
batch_size = 10
dataloaders['train'], dataset_sizes['train'], class_names = load_dataset(
    'train', batch_size)
dataloaders['val'], dataset_sizes['val'], _ = load_dataset('val', batch_size)
idx_to_class = {0: 'Negative', 1: 'Positive'}

# train model
#original

def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        st.write('Epoch {}/{}'.format(epoch, num_epochs - 1))
        st.write('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            current_loss = 0.0
            current_corrects = 0

            # Here's where the training happens
            st.write('Iterating through data...')

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # We need to zero the gradients, don't forget it
                optimizer.zero_grad()

                # Time to carry out the forward training poss
                # We only need to log the loss stats if we are in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # We want variables to hold the loss statistics
                current_loss += loss.item() * inputs.size(0)
                current_corrects += torch.sum(preds == labels.data)

            epoch_loss = current_loss / dataset_sizes[phase]
            epoch_acc = current_corrects.double() / dataset_sizes[phase]

            st.write('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # Make a copy of the model if the accuracy on the validation set has improved
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        st.write()

    time_since = time.time() - since
    st.write('Training complete in {:.0f}m {:.0f}s'.format(
        time_since // 60, time_since % 60))
    st.write('Best val Acc: {:4f}'.format(best_acc))

    # Now we'll load in the best model weights and return it
    model.load_state_dict(best_model_wts)
    return model


# visualise

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_handeled = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_handeled += 1
                ax = plt.subplot(num_images // 2, 2, images_handeled)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_handeled == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


# visualise

base_model = train_model(resnet50,
                         criterion,
                         optimizer,
                         exp_lr_scheduler,
                         num_epochs=5)

fig = visualize_model(base_model)

st.pyplot(fig)

#inference


def predict(model, test_image, print_class=False): #prediction function

    transform = chosen_transforms['val']

    test_image_tensor = transform(test_image)

    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 227, 227).cuda()# if i get gpu will use that
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 227, 227) # cpu will be good 

    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(test_image_tensor)
        ps = torch.exp(out)
        topk, topclass = ps.topk(1, dim=1)
        class_name = idx_to_class[topclass.cpu().numpy()[0][0]]
        if print_class:
            st.write("Output class :  ", class_name)
    return class_name


import tempfile  # added temp file for user to add files for analysis


#creating a function to analysis cracks in the slabs as per trained model
def predict_on_crops(input_image, height=227, width=227, save_crops=False):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(input_image.read())
        input_image_path = tmp.name

    im = cv2.imread(input_image_path)
    imgheight, imgwidth, channels = im.shape
    k = 0
    output_image = np.zeros_like(im)
    for i in range(0, imgheight, height):
        for j in range(0, imgwidth, width):
            a = im[i:i + height, j:j + width]
            ## discard image cropss that are not full size
            predicted_class = predict(base_model, Image.fromarray(a))
            ## save image
            file, ext = os.path.splitext(input_image_path)
            image_name = file.split('/')[-1]
            folder_name = 'out_' + image_name
            ## Put predicted class on the image
            if predicted_class == 'Positive':
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            cv2.putText(a, predicted_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 1, cv2.LINE_AA)
            b = np.zeros_like(a, dtype=np.uint8)
            b[:] = color
            add_img = cv2.addWeighted(a, 0.9, b, 0.1, 0)
            ## Save crops
            if save_crops:
                if not os.path.exists(os.path.join('uploads', folder_name)):
                    os.makedirs(os.path.join('uploads', folder_name))
                filename = os.path.join('uploads', folder_name,
                                        'img_{}.png'.format(k))
                cv2.imwrite(filename, add_img)
            output_image[i:i + height, j:j + width, :] = add_img
            k += 1
    ## Save output image
    cv2.imwrite(os.path.join('uploads', 'predictions', folder_name + '.jpg'),
                output_image)
    return output_image


uploaded_file = st.file_uploader("Choose an image...",
                                 key="file_uploader_1",
                                 type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file to disk
    with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    #get image details 

    st.write(f"Image format: {image.format}")
    st.write(f"Image size: {image.size}")
    st.write(f"Image mode: {image.mode}")

    # Perform crack detection analysis on saved file
    output_image = predict_on_crops(
        open(os.path.join('uploads', uploaded_file.name), "rb"), 128, 128)

    # Save output image to disk
    output_file_path = os.path.join("results",
                                    f"{uploaded_file.name}_output.jpg")
    cv2.imwrite(output_file_path, output_image)

    # Display output image
    st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB),
             use_column_width=True)

    # Display link to download output image
    st.markdown(
        f"Download output image: [output_image.jpg]({output_file_path})")
else:
    st.write("Please upload an image.")

    # Ask user if they want to upload a new image
upload_new_image = st.checkbox("Upload new image")

if upload_new_image:
    # Display file uploader
    uploaded_file = st.file_uploader("Choose an image...",
                                     key="file_uploader_new_image_1",
                                     type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save uploaded file to disk
        with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
            predict_on_crops(os.path.join('uploads', uploaded_file.name), 128,
                             128)

        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        st.write(f"Image format: {image.format}")
        st.write(f"Image size: {image.size}")
        st.write(f"Image mode: {image.mode}")

        # Perform crack detection analysis on saved file
        output_image = predict_on_crops(
            os.path.join('uploads', uploaded_file.name), 128, 128)

        # Save output image to disk
        output_file_path = os.path.join("results",
                                        f"{uploaded_file.name}_output.jpg")
        cv2.imwrite(output_file_path,
                    cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))

        # Display output image
        st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB),
                 use_column_width=True)

        # Display link to download output image
        st.markdown(
            f"Download output image: [output_image.jpg]({output_file_path})")
    else:
        st.write("Please upload an image.")
else:
    st.write("Please upload a new image.")

# Add a button to delete all files from uploads folder
if st.button("Delete all files from uploads folder"):
    for filename in os.listdir("uploads"):
        file_path = os.path.join("uploads", filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            st.error(f"Error deleting file: {e}")
    st.success("All files deleted from uploads folder!")

    #add logistics regresssion model below this
