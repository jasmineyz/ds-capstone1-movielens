################################################################################
# MovieLens Recommendation System Project
# HarvardX PH125.9x - Data Science Capstone 1
# Author: Jasmine Zhang
################################################################################

################################################################################
# STEP 1: Setup and Data Preparation
# - Load necessary packages
# - Download and prepare MovieLens data
# - Extract 'year' and 'review_date' features
################################################################################

# Install/load packages

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(tidyr)) install.packages("tidyr", repos = "http://cran.us.r-project.org")
if(!require(forcats)) install.packages("forcats", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# Download & unzip MovieLens dataset if not present
options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# Unzip ratings and movies data if not already present
ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

# Load and format ratings data
ratings <- as.data.frame(str_split(read_lines(ratings_file), 
                                   fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

# Read and format movies
movies <- as.data.frame(str_split(read_lines(movies_file), 
                                  fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

# Merge ratings and movies data
movielens <- left_join(ratings, movies, by = "movieId")


# Extract year from title column and create review_date column
movielens <- movielens %>% mutate(title = str_trim(title)) %>%
  # split title column to two columns: title and year
  extract(title, c("title_temp", "year"), regex = "^(.*) \\(([0-9 \\-]*)\\)$", remove = F) %>%
  # for series take debut date
  mutate(year = if_else(str_length(year) > 4, as.integer(str_split(year, "-", simplify = T)[1]), as.integer(year))) %>%
  # replace title NA's with original title
  mutate(title = if_else(is.na(title_temp), title, title_temp)) %>%
  # drop title_tmp column
  select(-title_temp) %>%
  mutate(review_date = round_date(as_datetime(timestamp), unit = "month"))

################################################################################
# STEP 2: Data Partitioning
# - Split movielens data into edx (training/testing) and validation sets
################################################################################

# Partition: 90% train (edx), 10% validation (final hold-out set)
set.seed(1, sample.kind = "Rounding") 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId, movieId, year, genres and review_date in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId") %>%
  semi_join(edx, by = "year") %>%
  semi_join(edx, by = "genres") %>%
  semi_join(edx, by = "review_date")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


################################################################################
# STEP 3: Exploratory Data Analysis
# - Summarize key dataset properties
# - Visualize main patterns with standardized plot_theme()
################################################################################

# Consistent plot theme for all ggplots
plot_theme <- function(base_size = 13) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 11, hjust = 0.5, margin = margin(b = 10)),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 10),
      plot.caption = element_text(size = 9, face = "italic", hjust = 1),
      plot.margin = margin(15, 15, 15, 15)
    )
}

################################################################################
# TABLE 1: Dataset Summary — Unique Users, Movies, Ratings, and Genres
################################################################################
edx %>%
  summarize(
    n_users = n_distinct(userId),
    n_movies = n_distinct(movieId),
    n_ratings = n(),
    n_genres = n_distinct(genres)
  )

################################################################################
# TABLE 2: Top 10 Most Rated Movies (by Number of Ratings)
################################################################################
edx %>%
  group_by(title) %>%
  summarize(count = n()) %>%
  arrange(desc(count)) %>%
  slice_head(n = 10)

################################################################################
# TABLE 3: Top 10 Highest Rated Movies (by Average Rating, min. 1000 ratings)
################################################################################
edx %>%
  group_by(title) %>%
  filter(n() >= 1000) %>%
  summarize(average_rating = mean(rating)) %>%
  arrange(desc(average_rating)) %>%
  slice_head(n = 10)

################################################################################
# FIGURE 1: Distribution of Movie Ratings
################################################################################

edx %>%
  group_by(movieId) %>%
  summarise(average_rating = mean(rating), .groups = 'drop') %>%
  ggplot(aes(average_rating)) +
  geom_histogram(binwidth = 0.1, color = "white", fill = "steelblue") +
  labs(
    x = "Average Rating",
    y = "Number of Movies"
  ) +
  plot_theme()

mean(edx$rating)

################################################################################
# FIGURE 2: Rating Density by Movie (Log Scale)
################################################################################
edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white") +
  scale_x_log10() +
  labs(title = "Rating Density by Movie (Log Scale)",
       x = "Number of Ratings", y = "Count of Movies") +
  plot_theme()

################################################################################
# FIGURE 3: Average Rating per User
################################################################################
edx %>%
  group_by(userId) %>%
  summarize(average_rating = mean(rating)) %>%
  ggplot(aes(average_rating)) +
  geom_histogram(binwidth = 0.1, fill = "steelblue", color = "white") +
  labs(title = "Average Rating per User",
       x = "Average Rating", y = "Number of Users") +
  plot_theme()

################################################################################
# FIGURE 4: Rating Distribution by Movie Genre (Combined)
################################################################################

edx %>%
  group_by(genres) %>%
  summarize(n = n(),
            avg = mean(rating),
            se = sd(rating)/sqrt(n()),
            .groups = 'drop') %>%
  filter(n >= 100000) %>%
  mutate(genres = reorder(genres, avg)) %>%
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) +
  geom_point(color = "steelblue", size = 2) +
  geom_errorbar(width = 0.4, color = "black") +
  coord_flip() +  # better readability
  labs(
    title = "Average Rating by Genre Combination",
    subtitle = "With 95% confidence intervals (n ≥ 100,000)",
    x = "Genre Combination",
    y = "Average Rating",
  ) +
  plot_theme()

################################################################################
# FIGURE 5: Number of Movies by Release Year
################################################################################
library(scales)  # for label_number()

edx %>%
  count(year) %>%
  ggplot(aes(x = year, y = n)) +
  geom_line(color = "steelblue", size = 1) +
  scale_y_continuous(
    labels = label_number(scale = 1/1000),  # Divide by 1,000
    breaks = seq(0, 800000, 200000)         # Customize as needed
  ) +
  labs(
    title = "Number of Ratings by Year",
    x = "Year",
    y = "Number of Ratings (thousands)"
  ) +
  plot_theme()

################################################################################
# FIGURE 6: Average Rating by Release Year
################################################################################

edx %>%
  group_by(year) %>%
  summarize(
    avg_rating = mean(rating),
    .groups = 'drop'
  ) %>%
  ggplot(aes(x = year, y = avg_rating)) +
  geom_jitter(width = 0.3, height = 0, color = "black", size = 1, alpha = 0.7) +
  geom_smooth(se = TRUE, color = "steelblue", size = 1) +
  labs(
    title = "Average Rating by Release Year",
    subtitle = "Each point shows the average rating of movies released that year (jittered)",
    x = "Release Year",
    y = "Average Rating") +
  plot_theme()

################################################################################
# FIGURE 7: Average Movie Rating by Date of Review
################################################################################

# Plot average rating by date of review in the edx dataset
edx %>% group_by(review_date) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(review_date, rating)) +
  geom_jitter(width = 0.3, height = 0, color = "black", size = 1, alpha = 0.7) +
  geom_smooth(se = TRUE, color = "steelblue", size = 1) +
  labs(x = "Date of Review", y = "Average Rating") +
  plot_theme()

################################################################################
# STEP 4: Data Partitioning for Cross-Validation
# - Partition edx into train and test sets
################################################################################

# Split edx into train/test for model development and tuning
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]


# Ensure no new levels in test_set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId") %>%
  semi_join(train_set, by = "year") %>%
  semi_join(train_set, by = "genres") %>%
  semi_join(edx, by = "review_date")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test_set) 
train_set <- rbind(train_set, removed)

# Remove temporary files to tidy environment
rm(test_index, temp, removed) 

################################################################################
# STEP 5: Model Building & Regularization
################################################################################

# Objective benchmark
rmse_objective <- 0.86490
rmse_results <- data.frame(Method = "Project objective", RMSE = "0.86490", Difference = "-")

# 1. Naive model (global average)
mu_hat <- mean(train_set$rating)
simple_rmse <- RMSE(test_set$rating, mu_hat)

# 2. Movie effect (b_i)
movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarise(b_i = mean(rating - mu_hat))

# Predict ratings adjusting for movie effects
predicted_b_i <- mu_hat + test_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  pull(b_i)

# Calculate RMSE based on movie effects model
movie_rmse <- RMSE(predicted_b_i, test_set$rating)

# 3. User effect (b_u)
user_avgs <- train_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu_hat - b_i))

# Predict ratings adjusting for movie and user effects
predicted_b_u <- test_set %>%
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by="userId") %>%
  mutate(pred = mu_hat + b_i + b_u) %>%
  pull(pred)

# Calculate RMSE based on user effects model
user_rmse <- RMSE(predicted_b_u, test_set$rating)

# 4. Genre effect (b_g)
genre_avgs <- train_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  group_by(genres) %>%
  summarise(b_g = mean(rating - mu_hat - b_i - b_u))

# Predict ratings adjusting for movie, user and genre effects
predicted_b_g <- test_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(genre_avgs, by = "genres") %>%
  mutate(pred = mu_hat + b_i + b_u + b_g) %>%
  pull(pred)

# Calculate RMSE based on genre effects model
genre_rmse <- RMSE(predicted_b_g, test_set$rating)

# 5. Release year effect (b_y)
year_avgs <- train_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(genre_avgs, by = "genres") %>%
  group_by(year) %>%
  summarise(b_y = mean(rating - mu_hat - b_i - b_u - b_g))

# Predict ratings adjusting for movie, user, genre and year effects
predicted_b_y <- test_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(genre_avgs, by = "genres") %>%
  left_join(year_avgs, by = "year") %>%
  mutate(pred = mu_hat + b_i + b_u + b_g + b_y) %>%
  pull(pred)

# Calculate RMSE based on year effects model
year_rmse <- RMSE(predicted_b_y, test_set$rating)

# 6. Review date effect (b_r)
date_avgs <- train_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(genre_avgs, by = "genres") %>%
  left_join(year_avgs, by = "year") %>%
  group_by(review_date) %>%
  summarise(b_r = mean(rating - mu_hat - b_i - b_u - b_g - b_y))

# Predict ratings adjusting for movie, user, genre, year and review date effects
predicted_b_r <- test_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(genre_avgs, by = "genres") %>%
  left_join(year_avgs, by = "year") %>%
  left_join(date_avgs, by = "review_date") %>%
  mutate(pred = mu_hat + b_i + b_u + b_g + b_y + b_r) %>%
  pull(pred)

# Calculate RMSE based on review date effects model
review_rmse <- RMSE(predicted_b_r, test_set$rating)

# 7. Regularization (grid search for lambda)
inc <- 0.1
lambdas <- seq(3, 6, inc)
# Regularise model, predict ratings and calculate RMSE for each value of lambda
rmses <- sapply(lambdas, function(l){
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu_hat)/(n()+l))
  b_u <- train_set %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - b_i - mu_hat)/(n()+l))
  b_g <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarise(b_g = sum(rating - b_i - b_u - mu_hat)/(n()+l))
  b_y <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_g, by="genres") %>%
    group_by(year) %>%
    summarise(b_y = sum(rating - b_i - b_u - b_g - mu_hat)/(n()+l))
  b_r <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_g, by="genres") %>%
    left_join(b_y, by="year") %>%
    group_by(review_date) %>%
    summarise(b_r = sum(rating - b_i - b_u - b_g - mu_hat)/(n()+l))
  predicted_ratings <- test_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_g, by="genres") %>%
    left_join(b_y, by="year") %>%
    left_join(b_r, by="review_date") %>%
    mutate(pred = mu_hat + b_i + b_u + b_g + b_y + b_r) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_set$rating))
})

# Assign optimal tuning parameter (lambda)
lambda <- lambdas[which.min(rmses)]
# Minimum RMSE achieved
regularised_rmse <- min(rmses) 

###########################################################################################################################
# STEP 6: Final Model Training on edx, Prediction on Validation
###########################################################################################################################

# Use the entire edx dataset to model effects, regularized with chosen value for lambda
b_i <- edx %>%
  group_by(movieId) %>%
  summarise(b_i = sum(rating - mu_hat)/(n()+lambda))

b_u <- edx %>%
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarise(b_u = sum(rating - b_i - mu_hat)/(n()+lambda))

b_g <- edx %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  group_by(genres) %>%
  summarise(b_g = sum(rating - b_i - b_u - mu_hat)/(n()+lambda))

b_y <- edx %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(b_g, by="genres") %>%
  group_by(year) %>%
  summarise(b_y = sum(rating - b_i - b_u - b_g - mu_hat)/(n()+lambda))

b_r <- edx %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(b_g, by="genres") %>%
  left_join(b_y, by="year") %>%
  group_by(review_date) %>%
  summarise(b_r = sum(rating - b_i - b_u - b_g - b_y - mu_hat)/(n()+lambda))

# Predict ratings in validation set using final model
predicted_ratings <- validation %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(b_g, by="genres") %>%
  left_join(b_y, by="year") %>%
  left_join(b_r, by="review_date") %>%
  mutate(pred = mu_hat + b_i + b_u + b_g + b_y + b_r) %>%
  pull(pred)

# Final RMSE on hold-out validation set
valid_rmse <- RMSE(validation$rating, predicted_ratings)

# Plot RMSE results against each tuning parameter (lambda) in order to find optimal tuner
data.frame(lambdas, rmses) %>%
  ggplot(aes(lambdas, rmses)) +
  geom_point() +
  geom_hline(yintercept=min(rmses), linetype='solid', col = "steelblue") +
  annotate("text", x = lambda, y = min(rmses), 
           label = lambda, vjust = -1, color = "steelblue") +
  labs(
    x = "Lambda",
    y = "RMSE",
    title = "RMSE by Lambda (Regularization)" ) +
  plot_theme()

###########################################################################################################################
# STEP 7: Summarize and Visualize Results
###########################################################################################################################

# 1. Add naive RMSE result to table
rmse_results <- rmse_results %>% 
  rbind(c("Naive model", round(simple_rmse,5), round(simple_rmse-rmse_objective,5)))

# 2. Plot movie effects distribution
movie_avgs %>%
  ggplot(aes(b_i)) +
  geom_histogram(bins = 10, fill = "steelblue", color = "black") +
  labs(
    x = "Movie effects (b_i)",
    title = "Distribution of Movie Effects" ) +
  plot_theme()

# Amend table to include movie effects model RMSE result
rmse_results <- rmse_results %>% 
  rbind(c("Movie effects (b_i)", 
          round(movie_rmse, 5), 
          round(movie_rmse-rmse_objective, 5)))

# 3. Plot user effects distribution
user_avgs %>%
  ggplot(aes(b_u)) +
  geom_histogram(bins = 10, fill = "steelblue", color = "black") +
  labs(
    x = "User effects (b_u)",
    title = "Distribution of User Effects"
  ) +
  plot_theme()

# Amend table to include user effects model RMSE result
rmse_results <- rmse_results %>% 
  rbind(c("Movie + User effects (b_u)", 
          round(user_rmse, 5), 
          round(user_rmse-rmse_objective, 5)))

# 4. Plot genre effects distribution
genre_avgs %>%
  ggplot(aes(b_g)) +
  geom_histogram(bins = 10, fill = "steelblue", color = "black") +
  labs(
    x = "Genre effects (b_g)",
    title = "Distribution of Genre Effects"
  ) +
  plot_theme()

# Amend table to include genre effects model RMSE result
rmse_results <- rmse_results %>% 
  rbind(c("Movie, User and Genre effects (b_g)", 
          round(genre_rmse, 5), 
          round(genre_rmse-rmse_objective, 5)))

# 5. Plot year of release effects distribution
year_avgs %>%
  ggplot(aes(b_y)) +
  geom_histogram(bins = 10, fill = "steelblue", color = "black") +
  labs(x="Year effects (b_y)",
       title = "Distribution of Release Year Effects") +
  plot_theme()

# Amend table to include year of release effects model RMSE result
rmse_results <- rmse_results %>% 
  rbind(c("Movie, User, Genre and Year effects (b_y)", 
          round(year_rmse, 5), 
          round(year_rmse-rmse_objective, 5)))

# 6. Plot review date effects distribution
date_avgs %>%
  ggplot(aes(b_r)) +
  geom_histogram(bins = 10, fill = "steelblue", color = "black") +
  labs(
    x = "Review date effects (b_r)",
    title = "Distribution of Review Date Effects"
  ) +
  plot_theme()

# Amend table to include review date effects model RMSE result
rmse_results <- rmse_results %>% 
  rbind(c("Movie, User, Genre, Year and Review Date effects (b_r)", 
          round(review_rmse, 5), 
          round(review_rmse-rmse_objective, 5)))

# 7. Amend table to include regularized model RMSE result
rmse_results <- rmse_results %>% 
  rbind(c("Regularised Movie, User, Genre, Year and Review Date effects", 
          round(regularised_rmse, 5), 
          format(round(regularised_rmse-rmse_objective, 5), scientific = F)))

# 7.2 Visualize RMSE Results
rmse_results <- rmse_results %>%
  mutate(RMSE = as.numeric(RMSE))

ggplot(rmse_results, aes(x = RMSE, y = reorder(Method, RMSE))) +
  geom_col(fill = "steelblue", width = 0.7) +
  geom_vline(xintercept = 0.8649, linetype = "dashed", color = "red", linewidth = 1) +
  geom_text(aes(label = format(RMSE, digits = 5, nsmall = 5)), 
            hjust = -0.1, size = 3) +  # Changed sprintf to format
  labs(
    title = "Model Performance: RMSE by Method",
    x = "RMSE",
    y = "Method",
    caption = "Red dashed line: Project Objective (0.8649)"
  ) +
  plot_theme() +
  scale_x_continuous(expand = expansion(mult = c(0, 0.1)))

###########################################################################################################################
#  STEP 8: Tabulate the final validation RMSE results
# - The table below summarizes the algorithm’s performance on the validation set,
#   after training and tuning all model effects using the entire edx dataset.
# - This represents the unbiased, out-of-sample estimate of model accuracy.
###########################################################################################################################

#Create table to show final validation RMSE result and project objective
final_results <- data.frame(Method = "Project objective", 
                            RMSE = "0.86490", Difference = "-") %>% 
  rbind(c("Validation of Final Model", 
          round(valid_rmse, 5), 
          format(round(valid_rmse-rmse_objective, 5), scientific = F)))
