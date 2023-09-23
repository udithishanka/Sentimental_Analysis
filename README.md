# Sentimental_Analysis

Harnessing the Power of Natural Language Processing

In today's data-driven world, understanding the sentiment and emotional tone behind textual data is crucial for making informed decisions and gaining valuable insights. This Sentiment Analysis Model, built on state-of-the-art natural language processing technology, empowers you to delve deeper into the sentiments expressed in text. Leveraging powerful tools such as the Transformers library, PyTorch, web scraping with BeautifulSoup, and regex for data preprocessing, this model goes beyond mere sentiment classification. It enables you to explore and analyze sentiment dynamics, providing valuable perspectives on customer feedback, social media trends, and text-based data in various domains. I have published an article with this model in [Spark Insights](https://eesoc.lk/magazine) Magazine by the Electrical Engineering Society, University of Moratuwa, Sri lanka, named "Propelling Intelligent Automation in Customer Service with NLP" on Page 48. 

## Content Overview
- Install and Import Dependencies
- Instantiate Model
- Encode and Calculate Sentiment
- Collect Reviews
- Load Reviews into the Dataframe and Score

### Install and Import Dependancies

```python
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
!pip install transformers requests beautifulsoup4 pandas numpy

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re
```

### Instantiate Model
```python
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
```

### Encode and Calculate Sentiment

```python
tokens = tokenizer.encode('It was good but couldve been better. Great', return_tensors='pt')
result = model(tokens)

```

### Collect Reviews

```python
r = requests.get('https://www.yelp.com/biz/social-brew-cafe-pyrmont')
soup = BeautifulSoup(r.text, 'html.parser')
regex = re.compile('.*comment.*')
results = soup.find_all('p', {'class':regex})
reviews = [result.text for result in results]

```


### Load Reviews into the Dataframe and Score

```python
import numpy as np
import pandas as pd

df = pd.DataFrame(np.array(reviews), columns=['review'])

def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1

```


![image](https://github.com/udithishanka/Sentimental_Analysis/assets/107479890/8ee9e381-c079-4f11-8ade-039da4eece41)
