# acml-hw-3

## Get a Cluster
Personally, the hardest part of this project was getting my hands on a Kubernetes cluster with a GPU. Initially, I was running this command in the Google Cloud Terminal and hoping it worked

```
gcloud container clusters create acml-hw-3-gpu  --zone=australia-southeast1-c --machine-type=n1-standard-4  --num-nodes=1  --accelerator=type=nvidia-tesla-t4,count=1 
```

Unfortunately, it did not immediately work. After running it and changing the zone (and praying) it still did not work. Taking this approach was arduous: it takes about 20 minutes each time only for it to output that it was not available. 

To account for this issue, I decided to write a script which automates the search process. Even then, it took a long time so I decided to paralellize it and eventually was able to get my hands on two GPU clusters with 1 node. This code can be found in `create_kub_cluster.ipynb`


## Training and Inference Components
As per the description, I have a simple training program for a CNN, a docker file, and a training job yaml file.

## Frontend and Application
With the help of AI, I was able to create a simple frontend which uses the served model.
