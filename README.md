### MFML (Matrix Factorization MovieLens in torch)
This repository is implementation about [Matrix Factorization Techniques for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)

### Dataset

The MovieLens Dataset is used as a bench mark dataset in journals, conference papers, and State-of-the-Art papers in the Recommendation System field. The MovieLens dataset consists of columns including userId, movieId, rating, timestamp, tag, title, genres, etc. Among the columns, the ratings.csv file consisting of userId, movieId, rating, and timestamp is used. Based on a 100k dataset, the dataset consists of 610 users, 9742 movies, and 100836 ratings. MovieLens data is basically a user-based dataset, including only users who have evaluated at least 20 movies for each user. The file size ranges from 100k, 1M to a maximum of 20M.

### Model explanation

<p align="center">
<div class="center">
  <figure>
    <a href="/images/matrix.png"><img src="images/matrix.png" width="600" ></a>
  </figure>
</div>
</p>

- A Matrix Factorization model was implemented based on the Matrix Factorization Techniques for Recommender Systems paper.
- The Matrix Factorization model completes the matrix for the target by inner product (dot product) of latent factors for user-item interaction. MF can be implemented considering both implicit feedback and explicit feedback, and the model was trained in consideration of only explicit feedback in the task.

- In this task, in addition to dot product of features, terms such as bias and confidence score described in the paper were added to the prediction.
   - The bias term was added to prevent the bias of ratings for specific users or specific items from being included in the model.
   - The confidence score described in the paper is an indicator of how reliable the user is, and it can also be judged by the frequency of ratings for each user.

   This formula is a cost function with the addition of bias and confidence score terms. RSME (Root Squared Mean Error) was used as the cost function.

### Matrix Factorization model directory tree

```python
.
├── dataset
├── evaluation.py
├── main.py
├── model
│   └── MF.py
├── optimize.py
├── train.py
└── utils.py
```

### development enviroment

- OS: Max OS X
- IDE: pycharm, gpu sever computer vim
- GPU: NVIDIA RTX A6000

### Quick start

```python
python main.py -e 30 -b 32 -f 30 -lr 0.001 -down True
```

## Matrix Factorization Result

| MovieLens 100K | average cost(RMSE) | epoch | # latent factor |
| --- | --- | --- | --- |
| Matrix Factorization | 0.9108 | 30 | 30 |
| Matrix Factorization with confidence score | 0.9242 | 30 | 30 |
| Matrix Factorization with bias | 0.9668 | 30 | 30 |
| Matrix Factorization with confidence score and bias | 1.4973 | 30 | 30 |



The number of factors showed the best performance at 19.

<p align="center">
<div class="center">
  <figure>
    <a href="/images/optimize_graph.png"><img src="images/optimize_graph.png" width="600" ></a>
  </figure>
</div>
</p>


epoch =30, batch=32, learning rater = 0.001, best factor =19일 때의 epoch별 RMSELoss.

<p align="center">
<div class="center">
  <figure>
    <a href="/images/loss_curve.png"><img src="images/loss_curve.png" width="600" ></a>
  </figure>
</div>
</p>


reference : [Matrix Factorization Techniques for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)
review written in korean : [Review](https://changhyeonnam.github.io/2021/12/21/Matrix_Factorization.html)

Matrix Factorization with MovieLens in torch
