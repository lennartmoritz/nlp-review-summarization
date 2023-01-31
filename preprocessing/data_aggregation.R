# Rprofile
#
# This file was created by collecting multiple R-Scripts that were used over the
# course of this project. The script in it's current form is not meant to be run
# like this, but rather give insights into the steps and processes our group did
# to reach our projects goal.
#
# Load and aggregate data, transform into tidy format and save for next steps.
# 
# Author: Christian "Doofnase" Schuler
# Date: 2023 Jan
################################################################################

print("Loading packages:")
library(tidyverse) # Many useful functions
library(dplyr)    # Many useful functions
library(ggplot2) # Fancy plotting-functions
library(ggpubr)
library(rstatix)
library(gridExtra) # Printing dataframe to pdf
library(data.table)
library(DataExplorer) # Amazing data exploration
library(rmarkdown)    # To work with html_document
library(prettydoc)    # For fancy html-themes
library("patchwork")   # Patching plots together

# Set initial working directory
dir_init <- getwd()
dir_input <- "/home/christianschuler/data/NLP-Project-Review-Summaries/"
dir_out <- paste0(dir_init, "/output/", sep="")
setwd(dir_init) # Set working directory

print("Ready to go:")


# Directories ##################################################################
print("Reading data:")
# Set working directory for input
setwd(dir_input)

################################################################################
# IMDB Dataset of 50K Movie Reviews
# https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
data_imdb <- read.csv(file = "IMDB Dataset.csv")
data_imdb <- data_imdb %>% rename(review_sentiment = sentiment)

################################################################################
# Rotten Tomatoes movies and critic reviews dataset
# https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset
data_roto_crit <- read.csv(file = "rotten_tomatoes_critic_reviews.csv")
data_roto_movi <- read.csv(file = "rotten_tomatoes_movies.csv")

data_roto <- data_roto_crit %>% dplyr::select("rotten_tomatoes_link", "review_score", "review_content", "critic_name")
data_movi <- data_roto_movi %>% dplyr::select("rotten_tomatoes_link", "movie_title")
data_roto <- data_roto %>% left_join(data_movi, by = "rotten_tomatoes_link")
data_roto <- data_roto %>% rename(review_id = rotten_tomatoes_link, author_id = critic_name)

################################################################################
# Movie Review Data
# https://www.cs.cornell.edu/people/pabo/movie-review-data/
# polarity dataset v2.0 
data_reviews <- list()
list_of_files <- list.files(path = paste0(dir_input,"review_polarity/txt_sentoken/neg/"), full.names = TRUE)
for (i in seq_along(list_of_files)) {
  review_name <- list_of_files[i]
  current_filename <- basename(review_name)
  current_filename <- sub(".txt", "", current_filename)
  current_review <- paste(readLines(review_name), collapse = "\n") 
  current_review <- t(current_review)
  current_review <- as.data.frame(current_review)
  current_review$review_id <- current_filename
  data_reviews[[i]] <- current_review
}
data_reviews <- data_reviews %>% reduce(merge, all = TRUE)
data_reviews <- data_reviews %>% rename(review_content = V1)

data_sent_pola_2000_nega <- data_reviews
data_sent_pola_2000_nega$review_sentiment <- "negative"

data_reviews <- list()
list_of_files <- list.files(path = paste0(dir_input,"review_polarity/txt_sentoken/pos/"), full.names = TRUE)
for (i in seq_along(list_of_files)) {
  review_name <- list_of_files[i]
  current_filename <- basename(review_name)
  current_filename <- sub(".txt", "", current_filename)
  current_review <- paste(readLines(review_name), collapse = "\n") 
  current_review <- t(current_review)
  current_review <- as.data.frame(current_review)
  current_review$review_id <- current_filename
  data_reviews[[i]] <- current_review
}
data_reviews <- data_reviews %>% reduce(merge, all = TRUE)
data_reviews <- data_reviews %>% rename(review_content = V1)

data_sent_pola_2000_posi <- data_reviews
data_sent_pola_2000_posi$review_sentiment <- "positive"

################################################################################
# Movie Review Data
# https://www.cs.cornell.edu/people/pabo/movie-review-data/
# Pool of 27886 unprocessed html files
# TODO      data_sent_pola_html

################################################################################
# Movie Review Data
# https://www.cs.cornell.edu/people/pabo/movie-review-data/
# Sentiment scale datasets
# Read the id, rating, subj (line by line and file by file)

# Author 1
current_filename <- "scale_data/scaledata/Dennis+Schwartz/id.Dennis+Schwartz"
review_id <- readLines(current_filename)

current_filename <- "scale_data/scaledata/Dennis+Schwartz/rating.Dennis+Schwartz"
review_score <- readLines(current_filename)

current_filename <- "scale_data/scaledata/Dennis+Schwartz/subj.Dennis+Schwartz"
review_content <- readLines(current_filename)

#current_reviews <- cbind(review_id, review_score, review_content)
current_reviews <- dplyr::bind_cols(review_id, review_score, review_content)
current_reviews <- current_reviews %>% rename(review_id = ...1, review_score = ...2, review_content = ...3)
current_reviews$author_id <- "DennisSchwartz"
data_sent_pola_scal_01 <- current_reviews

# Author 2
current_filename <- "scale_data/scaledata/James+Berardinelli/id.James+Berardinelli"
review_id <- readLines(current_filename)

current_filename <- "scale_data/scaledata/James+Berardinelli/rating.James+Berardinelli"
review_score <- readLines(current_filename)

current_filename <- "scale_data/scaledata/James+Berardinelli/subj.James+Berardinelli"
review_content <- readLines(current_filename)

current_reviews <- dplyr::bind_cols(review_id, review_score, review_content)
current_reviews <- current_reviews %>% rename(review_id = ...1, review_score = ...2, review_content = ...3)
current_reviews$author_id <- "JamesBerardinelli"
data_sent_pola_scal_02 <- current_reviews

# Author 3
current_filename <- "scale_data/scaledata/Scott+Renshaw/id.Scott+Renshaw"
review_id <- readLines(current_filename)

current_filename <- "scale_data/scaledata/Scott+Renshaw/rating.Scott+Renshaw"
review_score <- readLines(current_filename)

current_filename <- "scale_data/scaledata/Scott+Renshaw/subj.Scott+Renshaw"
review_content <- readLines(current_filename)

current_reviews <- dplyr::bind_cols(review_id, review_score, review_content)
current_reviews <- current_reviews %>% rename(review_id = ...1, review_score = ...2, review_content = ...3)
current_reviews$author_id <- "ScottRenshaw"
data_sent_pola_scal_03 <- current_reviews

# Author 4
current_filename <- "scale_data/scaledata/Steve+Rhodes/id.Steve+Rhodes"
review_id <- readLines(current_filename)

current_filename <- "scale_data/scaledata/Steve+Rhodes/rating.Steve+Rhodes"
review_score <- readLines(current_filename)

current_filename <- "scale_data/scaledata/Steve+Rhodes/subj.Steve+Rhodes"
review_content <- readLines(current_filename)

current_reviews <- dplyr::bind_cols(review_id, review_score, review_content)
current_reviews <- current_reviews %>% rename(review_id = ...1, review_score = ...2, review_content = ...3)
current_reviews$author_id <- "ScottRenshaw"
data_sent_pola_scal_04 <- current_reviews

################################################################################
# Movie Review Data
# https://www.cs.cornell.edu/people/pabo/movie-review-data/
# sentence polarity dataset v1.0
data_reviews <- read.delim(file = paste0(dir_input,"rt-polaritydata/rt-polaritydata/rt-polarity.neg"), 
                           header = FALSE, 
                           sep = "\n", 
                           quote = "",
                           dec = ".")
data_reviews <- data_reviews %>% rename(review_content = V1)
data_sent_pola_5331_nega <- data_reviews
data_sent_pola_5331_nega$review_sentiment <- "negative"

data_reviews <- read.delim(file = paste0(dir_input,"rt-polaritydata/rt-polaritydata/rt-polarity.neg"), 
                           header = FALSE, 
                           sep = "\n", 
                           quote = "",
                           dec = ".")
data_reviews <- data_reviews %>% rename(review_content = V1)
data_sent_pola_5331_posi <- data_reviews
data_sent_pola_5331_posi$review_sentiment <- "negative"

################################################################################
# Movie Review Data
# https://www.cs.cornell.edu/people/pabo/movie-review-data/
# subjectivity dataset v1.0 
data_reviews <- read.delim(file = paste0(dir_input,"rotten_imdb/plot.tok.gt9.5000"), 
                           header = FALSE, 
                           sep = "\n", 
                           quote = "",
                           dec = ".")
data_reviews <- data_reviews %>% rename(review_content = V1)
data_sent_pola_5000_obje <- data_reviews

data_reviews <- read.delim(file = paste0(dir_input,"rotten_imdb/quote.tok.gt9.5000"), 
                           header = FALSE, 
                           sep = "\n", 
                           quote = "",
                           dec = ".")
data_reviews <- data_reviews %>% rename(review_content = V1)
data_sent_pola_5000_subj <- data_reviews


################################################################################
print("Tidying Data up")
setwd(dir_input)

# Add variable for dataset-identification
data_imdb$dataset_id <- "imdb"

data_roto$dataset_id <- "roto"

data_sent_pola_2000 <- dplyr::bind_rows(data_sent_pola_2000_nega, data_sent_pola_2000_posi)
data_sent_pola_2000$dataset_id <- "sent_pola_2000"

data_sent_pola_5331 <- dplyr::bind_rows(data_sent_pola_5331_nega, data_sent_pola_5331_posi)
data_sent_pola_5331$dataset_id <- "sent_pola_5331"

data_sent_pola_5000 <- dplyr::bind_rows(data_sent_pola_5000_obje, data_sent_pola_5000_subj)
data_sent_pola_5000$dataset_id <- "sent_pola_5000"

data_sent_pola_scal <- rbind(data_sent_pola_scal_01, data_sent_pola_scal_02, data_sent_pola_scal_03, data_sent_pola_scal_04)
data_sent_pola_scal$dataset_id <- "sent_scal"

# TODO
#data_sent_pola_html$dataset_id <- "sent_pola_html"

# Collect all selected variables from all data sets combined in one
data_all <- data_imdb %>% rename(review_content = review)
data_all <- dplyr::bind_rows(data_all, data_roto)
data_all <- dplyr::bind_rows(data_all, data_sent_pola_2000)
data_all <- dplyr::bind_rows(data_all, data_sent_pola_5331)
data_all <- dplyr::bind_rows(data_all, data_sent_pola_5000)
data_all <- dplyr::bind_rows(data_all, data_sent_pola_scal)

# Add ID for all data
data_all <- tibble::rowid_to_column(data_all, "ID")

# Save to file
saveRDS(data_all, 
        file = "data.Rdata", 
        compress = FALSE) # FALSE=Fast TRUE=Small

write.csv(data_all, file = "data.csv", row.names = FALSE)

setwd(dir_init)