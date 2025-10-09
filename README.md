# NYP_project

## 💻 How to Set Up the Environment

This section details the necessary steps to configure your computer for working on the project.

### 1. GPU Setup
A **GPU (Graphics Processing Unit)** is a specialized hardware component essential for deep learning. It accelerates the complex mathematical computations required for training and running models far more efficiently than a standard CPU.

* **Why it's crucial**: The project likely employs a sophisticated model, such as YOLO from Ultralytics (mentioned in the references), which demands significant computational power. A GPU drastically reduces the time needed for both training the model and running detections.
* **What to do**: You must install the appropriate drivers and software libraries for your specific GPU. For NVIDIA GPUs, this typically involves installing the **CUDA Toolkit** and the **cuDNN library**. These tools create a bridge that allows programming frameworks like PyTorch to leverage the GPU's processing power. 

### 2. Virtual Environment Setup
A **virtual environment** is a self-contained directory tree that includes a specific Python installation and a number of additional packages.

* **Why it's important**: It isolates the project's dependencies, preventing conflicts with other projects on your system. For instance, if this project requires version 1.0 of a library but another project needs version 2.0, virtual environments allow both to coexist without issues.
* **What to do**: The command `pip install -r requirements.txt` is the standard way to set this up. It reads a text file named `requirements.txt` (which should be included in the project) and installs all the listed libraries into your active virtual environment. If you encounter errors, you may need to install certain packages manually.

### 3. Git Environment and Cloning
**Git** is a distributed version control system used for tracking changes in source code during software development.

* **Why it's used**: It helps manage the project's codebase, track revisions, and collaborate with others effectively.
* **What to do**: After installing Git on your computer, you use the command `git clone <repository_url>` to download a complete copy of the project's code from a remote host like GitHub. The note about copying from a professor's USB is a simpler, offline alternative for getting the source code.

---

## 🎨 How to Use Roboflow (Data Preparation)

**Roboflow** is a comprehensive online platform for building computer vision datasets. This guide walks through the process of creating the custom dataset needed to train your model.

1.  **Get Started & Create Account**: Standard sign-up process on the Roboflow website.
2.  **Create a New Project**: Define the parameters of your project.
    * **Project Name**: A descriptive name, e.g., "High-Rise Safety Detection".
    * **Annotation Group**: The categories of objects you intend to detect. For this project, these would be `person` and `text`.
    * **Project Type**: The documentation recommends **Instance Segmentation**. This advanced annotation method involves drawing a precise, pixel-level mask around each object, providing the model with richer data than a simple rectangular bounding box.
3.  **Convert Video to Images**: Machine learning models are trained on individual images, not video files. The provided `video_to_images.py` script is a utility to extract still frames from video footage.
4.  **Upload and Annotate**: Upload the extracted image frames to your Roboflow project. This is where you will perform the manual task of **annotation** (also called labeling or segmenting) by carefully drawing masks around every person and piece of text in your images. This is the most crucial and labor-intensive step in building an accurate custom model.
5.  **Generate a Dataset Version**: Once your images are annotated, you generate a "version" of the dataset. This step allows you to apply:
    * **Preprocessing**: Standardizing actions, such as resizing all images to a uniform resolution (e.g., $640 \times 640$ pixels).
    * **Augmentation**: Techniques to artificially expand your dataset's size and diversity. This involves creating modified copies of your images with variations like rotation, changes in brightness, or added noise. Augmentation is key to training a robust model that can perform well on new, unseen data.
6.  **Download Dataset**: After generating a version, you can export it for training. Roboflow supports various formats, including the "YOLO format," which is compatible with the Ultralytics training framework.



---

## ▶️ How to Run the Project

The command `python detect_person_and_text.py` is the entry point to run the project's core functionality.

* **Prerequisites**: Before running, you must **activate your virtual environment** in the command line interface (CLI). This ensures that the script can access the specific libraries you installed for the project.
* **What it does**: This script loads the custom-trained model and performs **inference**. It takes a new input (such as an image, a video file, or a live webcam feed), processes it through the model, and displays the output—the original input overlaid with segmentation masks highlighting the detected people and text.

---

## 📚 Referred Documents

The provided links point to the documentation for the key technologies that power this project.

* **Roboflow Blog**: This "Getting Started" guide is an excellent resource for learning the fundamentals of the Roboflow platform and the data preparation workflow.
* **Ultralytics Docs**: This link directs to the documentation for **YOLO (You Only Look Once)**, a state-of-the-art family of real-time object detection models. The reference to "train mode" specifically indicates that the project uses the Ultralytics framework to train the custom model on the dataset prepared with Roboflow.


