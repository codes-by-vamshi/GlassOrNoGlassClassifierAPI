# Image Classification Algorithm for Glasses Detection

This project involves an algorithm that classifies an image of a face to determine whether the person is wearing glasses or not. The algorithm uses a pre-trained MobileNetV2 model fine-tuned for this specific task.

## GitHub Repositories

The project is divided into two main components, each hosted in its own GitHub repository:

1. **API Repository**: This repository contains the code for the Flask API that serves the model for inference.
   - **Repository Link**: [API Repository](https://github.com/codes-by-vamshi/GlassOrNoGlassClassifierAPI)
   
2. **Streamlit Application Repository**: This repository includes the code for the Streamlit application, which provides a user-friendly interface to interact with the model.
   - **Repository Link**: [Streamlit Application Repository](https://github.com/codes-by-vamshi/GlassOrNoGlassClassifierStreamlit)

## Hosted Application

The entire application is hosted on Render, allowing easy access and interaction with the model.

- **Hosted Link**: [Render Application](https://glassornoglassclassifierstreamlit.onrender.com/)

## Demonstration Video

[![YouTube Video](https://i9.ytimg.com/vi_webp/2530iGwlwro/mq1.webp?sqp=CIjJq7IG&rs=AOn4CLAYUN7tClpNR-mh8EieaDVBStrygA)](https://www.youtube.com/watch?v=2530iGwlwro)

## How It Works

1. **Input**: The user uploads an image of a face.
2. **Processing**: The image is processed and passed through the MobileNetV2 model.
3. **Output**: The model classifies the image and returns whether the person is wearing glasses or not.

## Usage

To use the application, follow these steps:

1. Visit the [Streamlit Application](https://glassornoglassclassifierstreamlit.onrender.com/).
2. Upload an image of a face using the provided interface.
3. Wait for the model to process the image.
4. View the classification result indicating whether the person in the image is with glasses or without glasses.

## Conclusion

This project demonstrates the use of a deep learning model to classify images of faces based on the presence of glasses. The combination of a Flask API and a Streamlit front-end makes it accessible and easy to use. Explore the repositories and the hosted application to learn more and see the model in action.
