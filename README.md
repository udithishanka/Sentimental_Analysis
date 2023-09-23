# Sentimental_Analysis

Harnessing the Power of Natural Language Processing

In today's data-driven world, understanding the sentiment and emotional tone behind textual data is crucial for making informed decisions and gaining valuable insights. This Sentiment Analysis Model, built on state-of-the-art natural language processing technology, empowers you to delve deeper into the sentiments expressed in text. Leveraging powerful tools such as the Transformers library, PyTorch, web scraping with BeautifulSoup, and regex for data preprocessing, this model goes beyond mere sentiment classification. It enables you to explore and analyze sentiment dynamics, providing valuable perspectives on customer feedback, social media trends, and text-based data in various domains. I have published an article with this model in [Spark Insights](https://eesoc.lk/magazine) Magazine, named "Propelling Intelligent Automation in Customer Service with NLP" on Page 48. 

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
```
