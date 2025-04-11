# acml-hw-3
For the purpose of this assignment, the app will be a simple application which takes in an image as input and ouputs the predicted class. The classifier is trained on the commonly-used CIFAR-10 dataset and classifies from: `airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`.


# Workflow
When a user logs in, they are greeted with a simple UI
![alt text](https://github.com/kentjliu/acml-hw-3/blob/main/etc/start-screen.png "Logo Title Text 1")

They can then upload an image. In this case, it is a dancing cat.
![alt text](https://github.com/kentjliu/acml-hw-3/blob/main/etc/image_upload.png "Logo Title Text 1")


The app returns a classification with a confidence score (from softmax layer)!
![alt text](https://github.com/kentjliu/acml-hw-3/blob/main/etc/get-result.png "Logo Title Text 1")

# Steps
## Get a Cluster
Personally, the hardest part of this project was getting my hands on a Kubernetes cluster with a GPU. Initially, I was running this command in the Google Cloud Terminal and hoping it worked

```
gcloud container clusters create acml-hw-3-gpu  --zone=australia-southeast1-c --machine-type=n1-standard-4  --num-nodes=1  --accelerator=type=nvidia-tesla-t4,count=1 
```

Unfortunately, it did not immediately work. After running it and changing the zone (and praying) it still did not work. Taking this approach was arduous: it takes about 20 minutes each time only for it to output that it was not available. 

To account for this issue, I decided to write a script which automates the search process. Even then, it took a long time so I decided to paralellize it and eventually was able to get my hands on two GPU clusters with 1 node. This code can be found in `create_kub_cluster.ipynb`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kentjliu/acml-hw-3/blob/main/create_kub_cluster.ipynb)




## Training and Inference Components
As per the description, I have a simple training program for a CNN, a docker file, and a training job yaml file.

## Frontend and Application
With the help of AI, I was able to create a simple frontend which uses the served model for simple image classification.

# Steps to Build and Run Application
(These are the params I used)
```
gcloud container clusters get-credentials acml-hw-3-gpu --zone asia-northeast1-a --project hpml-435803
```
Push the training image
```
cd training
docker build -t gcr.io/$PROJECT_ID/training-image:latest .
docker push gcr.io/$PROJECT_ID/training-image:latest
```
Push the inference image
```
cd ../inference
docker build -t gcr.io/$PROJECT_ID/inference-image:latest .
docker push gcr.io/$PROJECT_ID/inference-image:latest
```
Apply PVC and deploy both training and inference jobs
```
cd ../yaml
kubectl apply -f pvc.yaml
kubectl apply -f training.yaml
kubectl apply -f inference.yaml
kubectl get service model-infer-service
```
Finally, we can go to our favorite browser and go to https://[INSERT EXTERNAL IP] and classify images!
