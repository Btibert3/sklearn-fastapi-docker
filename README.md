# About

This repo aims to provide a basic example, but one that highlights how to build and deploy a Scikit-Learn Model with FastAPI and Docker.  Depending on your use-case, this infrastructure can go a long way in production and let's you focus on testing and optimizing models.  Unless deep-learning is necessary, the sklearn ecosystem provides a rich set of tooling for both unsupervised and supervised tasks with a simple interface and limited complexity for deployment.


# Process

1.  `train.py` will take the data from select `20 newsgroups` posts and use a sklearn pipeline to fit a text classifier.  
2.   `app.py` will load the model from `model/` and serve the inference API.  This will take a single piece of text and return a single answer.
3.  Build the docker image via `docker build -t textcat .` where `textcat` can be anything you like for a tag
4.  Run the docker image with `docker run -p 80:80 textcat`.

Now that you have dockerized your ML app, you can access the docs at `http://0.0.0.0/docs` locally.  

> When viewing the docs, it's really important to highlight that __you can try out the API__ within each endpoint!


# Next Steps

- The model constructed was intended to highlight scikit's awesome `Pipeline` object as well as how a model can be written to disk and loaded when the API server loads.   
- Add code to allow bulk processing endpoints.  For example, pass a list of 1000 texts, and get 1000 predictions back, as opposed to one at a time.
- Add logging and map volumes to save the files.
- Add authentication
