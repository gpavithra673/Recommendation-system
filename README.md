# Recommendation-system
## AIM:
### To develop a recommendation system for BOOKS using python.
## PROCEDURE:
### STEP 1:
### Import the dataset using pandas.

### STEP 2:
### Apply data-preprocessing techniques to remove the unnecessary contents from the dataset.
In this case, we can either create a user-defined function or we can use built-in functions to clean the data.
Here I have used user-defined function and anonymous function based upon the dataset.
### Summing up this process, we will end up with dataset in which all the uppercase letter are converted to lowercase, null values were removed, each row is applied for cleaning and converting the datas to string.

### STEP 3:
### After preprocessing we can apply COUNT VECTORIZER. The Count Vectorizer converts text documents into a matrix where each row represents a document, and each column represents a unique word in the entire collection of documents. The matrix's entries (cells) contain counts of how many times each word appears in each document.
![image](https://github.com/gpavithra673/Recommendation-system/assets/93427264/f372a6d1-ea19-4439-9cbe-b1d2b1d0d95c)

### STEP 4:
### Fit the dataset and transform it by applying fit_transform. The fit_transform() function is used on the training data so that we can scale the training data and also learn the scaling parameters of that data.

### STEP 5:
### Now we can calculate the similarity between the strings using cosine similarity. Cosine similarity is a metric, helpful in determining, how similar the data objects are irrespective of their size. We can measure the similarity between two sentences in Python using Cosine Similarity
![image](https://github.com/gpavithra673/Recommendation-system/assets/93427264/59135931-0f85-4950-b48c-9bad90d8af4f)
![image](https://github.com/gpavithra673/Recommendation-system/assets/93427264/fb677cbe-d29f-4cad-8ede-b10c22a47ebc)
We can also use other similarity measuring algorithms like EUCLIDEAN DISTANCE, JACCARD SIMILARITY etc.

### STEP 6:
### Print the similarity matrix to visuvalize the result and if needed we can reset the index for better handling of the dataset.

### STEP 7:
### Now test the model by using a sample input and print the result.

## PROGRAM:
```
import pandas as pd
df = pd.read_csv('BX-Books.csv',error_bad_lines=False,encoding='latin-1',sep=';')
df.head()
df.duplicated(subset='Book-Title').sum()
df = df.drop_duplicates(subset='Book-Title')
sample_size = 15000
df = df.sample(n=sample_size, replace=False, random_state=490)
df = df.reset_index()
df = df.drop('index',axis=1)
df.head()
def clean_text(author):
    result = str(author).lower()
    return(result.replace(' ',''))
df['Book-Author'] = df['Book-Author'].apply(clean_text)
df['Book-Title'] = df['Book-Title'].str.lower()
df['Publisher'] = df['Publisher'].str.lower()
df2 = df.drop(['ISBN','Image-URL-S','Image-URL-M','Image-URL-L','Year-Of-Publication'],axis=1)
df2['data'] = df2[df2.columns[1:]].apply(
    lambda x: ' '.join(x.dropna().astype(str)),
    axis=1
)
print(df2['data'].head())
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorized = vectorizer.fit_transform(df2['data'])
from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(vectorized)
print(similarities)
df = pd.DataFrame(similarities, columns=df['Book-Title'], index=df['Book-Title']).reset_index()
df.head()
input_book = 'far beyond the stars (star trek deep space nine)'
recommendations = pd.DataFrame(df.nlargest(11,input_book)['Book-Title'])
recommendations = recommendations[recommendations['Book-Title']!=input_book]
print(recommendations)
```
## OUTPUT: 
### DATASET DETAIL:
![image](https://github.com/gpavithra673/Recommendation-system/assets/93427264/047399e2-646f-4864-8e16-3f907607e0fd)

### VECTORIZED DATASET:
![image](https://github.com/gpavithra673/Recommendation-system/assets/93427264/52e1229b-b2d0-4686-9b01-0b6a6d017a02)

### COSINE SIMILARITY MATRIX:
![image](https://github.com/gpavithra673/Recommendation-system/assets/93427264/dd495d52-0725-47b3-ba1f-d72e23bf84b8)

### SAMPLE INPUT:
![image](https://github.com/gpavithra673/Recommendation-system/assets/93427264/76ca832d-a68c-4a62-846c-dddd65d06f81)

## RESULT:
### Thus we have successfully created a recommender for recommending books using the above mentioned code.
