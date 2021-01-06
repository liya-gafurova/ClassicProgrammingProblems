replace *pretrained* directory to other place, because during *docker build* it will copy all files from current directory.  
If model file will be big, it will take a lot of time to build an image.  

AllenNLP:
1) dependency model: can be downloaded and loaded from archive: 

*archive:Archive = load_archive('/root/.cache/torch/transformers/pretrained/biaffine-dependency-parser-ptb-2020.04.06.tar.gz')  
predictor: Predictor = Predictor.from_archive(archive)*

2) sentiment analysis model  

can be downoaded with archive, BUT during loading downloads additionally  Roberta Nodel form HuggingFace (*from_pretrained*).  
models from HuggingFace are downloaded to .cache directory (in container it is */root/.cache/torch/transformers*).  
Once it is in cache, it can be used from it and not downloaded.  

### -----------

I have placed downloaded models into directory with cached roBERTA files in:  
/root/.cache/torch/transformers/  
/root/.cache/torch/transformers/pretrained (*here are downloaded models*)  

### -----------

for more information see Dockerfile(nltk and spacy models are downloaded there)

### -----------

My image is 3.89 Gb.  
I think it is quite OK, because it is size of libraries, that are installed (torch, for example).  
For example, NLP for medicine backend image is approximately 6 Gb.

