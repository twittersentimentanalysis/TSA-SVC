# SVC Model API

API for applying sentiment analysis using SVC (Support Vector Classifier) model.

## Requirements
- python >= 3.9 [(download it here)](https://www.python.org/downloads/)

## How to run 
### Local
1. Clone this project to a local folder and go to root folder

   `git clone https://github.com/twittersentimentanalysis/TSA-SVC.git`

2. Install required libraries with `pip`

    `pip install -r requirements.txt`
    
3. Run the project

    `py RestAPI.py`

4. Make API requests

    Base URL: `http://localhost:6232/svc/v1`


## Endpoints
### Localhost
http://localhost:6232/svc/v1/emotion

### GESSI Server
http://gessi-sw.essi.upc.edu:6232/svc/v1/emotion