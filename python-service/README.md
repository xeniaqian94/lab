
## New README
`autograb-server.py` and variants with timestamp are scripts to start an aws service in EC2 instance as a localhost. 

It calls `util.py` which has an `inference()` function that calls the `reproducibility_classifier_train.py` for a pre-trained model.


## Old README 

In this directory, _`hello.py`_ (NO!!! new one is `python autograb-server.py `) is the Flask python service which src/io.ts calls from to auto-grab details from parsed PDF.

i.e. the TypeScript end-point invokes the API from http://localhost:5000/autograb/pdfdata

It is suggested to call with `source activate snorkel` for local debug of the service.

Our AWS instance runs with allennlp==0.8.1 and some version of spacy. Enter the instance to learn more about its platform. The autograb-service-aws.py is the current working script.




