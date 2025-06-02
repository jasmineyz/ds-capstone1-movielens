# ds-capstone1-movielens

This repository has the code and analysis for the MovieLens Recommendation System Project, completed as part of [HarvardX PH125.9x: Data Science - Capstone](https://www.edx.org/course/data-science-capstone).

## Project Overview

The goal of this project is to build and evaluate a predictive movie recommendation system using the MovieLens 10M dataset. The workflow includes data cleaning, exploratory analysis, stepwise model building, regularization, and final model validation.

## Key Steps

- **Data Preparation:** Cleaning and structuring the MovieLens 10M dataset for analysis.
- **Exploratory Data Analysis:** Visualizing rating patterns and identifying factors that influence ratings, such as movie, user, genre, release year, and review date.
- **Model Development:** Building models that add effects for movies, users, genres, release year, and review date.
- **Regularization:** Applying penalty terms to avoid overfitting and optimize model performance.
- **Evaluation:** Assessing model accuracy on a hold-out validation set. The final model achieves RMSE below the project benchmark.

## Main Results

- The final regularized model achieved an **RMSE of 0.8642** on the validation set, surpassing the project objective (RMSE < 0.8649).
- Most improvement came from adding user effects; genre, release year, and review date had smaller impacts.
- All work was completed in **R** using **R Markdown**.
