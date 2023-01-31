# R-Script
# 
# This file was created by collecting multiple R-Scripts that were used over the
# course of this project. The script in it's current form is not meant to be run
# like this, but rather give insights into the steps and processes our group did
# to reach our projects goal
#
# Explore data in different ways and create statistics for new insights.
#
# Author: Christian "Doofnase" Schuler
# Date: 2023 Jan
################################################################################

################################################################################
# Predicted Sentiment and normalised Scores from (roto) Review Data
print("Score normalisation via Python-Script prior to this step required:")

# Set working directory for input
setwd(dir_input)

data_sentiments <- read.csv(file = "data-sentiment.csv")
data_sentiments_normalised <- read.csv(file = "data-sentiment-normalised.csv")

setwd(dir_out)

data_base <- data_sentiments_normalised %>% 
  dplyr::select(normalised_score, bertbasemultiuncased_label, bertbasemultiuncased_score)
#data_base$bertbase_hit <- with(data_base, ifelse(normalised_score == bertbasemultiuncased_label, 'Hit', 'Miss'))

data_dist <- data_sentiments_normalised %>% 
  dplyr::select(normalised_pone, distilbertbaseuncasedsst2_label, distilbertbaseuncasedsst2_score)  
#data_dist$distbert_hit <- with(data_dist, ifelse(normalised_pone == distilbertbaseuncasedsst2_label, 'Hit', 'Miss'))

# Create a separate data frame that summarizes the count and confidence level 
# for each combination of true and predicted scores, and then use the color 
# aesthetic to map the confidence level to the color of the points.
data_base_sum <- data_base %>% 
  group_by(normalised_score, bertbasemultiuncased_label) %>%
  summarise(count = n(),
            avg_confidence = mean(bertbasemultiuncased_score))

plot_bertbase <- ggplot(data_base_sum, aes(x = normalised_score, y = bertbasemultiuncased_label, size = count, color = avg_confidence)) +
  geom_point() + 
  geom_abline(intercept = 0, slope = 1) + 
  #scale_size_continuous(guide = "none") +
  scale_color_gradient(low = "blue", high = "red") +
  labs(x = "True Scores", y = "Predicted Scores",
       size = "Count", color = "Average Confidence",
       title = "True Scores vs. Predicted Scores with Confidence and Count of 'Bert-Base'")

plot_bertbase


data_dist_sum <- data_dist %>% 
  group_by(normalised_pone, distilbertbaseuncasedsst2_label) %>%
  summarise(count = n(),
            avg_confidence = mean(distilbertbaseuncasedsst2_score))

plot_bertdist <- ggplot(data_dist_sum, aes(x = normalised_pone, y = distilbertbaseuncasedsst2_label, size = count, color = avg_confidence)) +
  geom_point() + 
  geom_abline(intercept = 0, slope = 1) + 
  #scale_size_continuous(guide = "none") +
  scale_color_gradient(low = "blue", high = "red") +
  labs(x = "True Scores", y = "Predicted Scores",
       size = "Count", color = "Average Confidence",
       title = "True Scores vs. Predicted Scores with Confidence and Count of 'Bert-Dist'")

plot_bertdist


################################################################################
# To increase the size of the dots based on frequency, first create a summary 
# data frame that contains the count of each combination of true and predicted 
# scores, and then use that count as the size aesthetic in the scatterplot.

data_base_sum <- data_base %>% 
  group_by(normalised_score, bertbasemultiuncased_label) %>%
  summarise(count = n())

plot_base <- ggplot(data_base_sum, aes(x = normalised_score, y = bertbasemultiuncased_label, size = count)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1) #+
#scale_size_continuous(guide = "none")
plot_base
################################################################################
plot_base <- ggplot(data_base, aes(x = normalised_score, y = bertbasemultiuncased_label, color = bertbasemultiuncased_score, size =..n..)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1) +
  scale_size_continuous(guide = "none")
#  geom_count(alpha = 0.5)
plot_base
################################################################################
plot_base <- ggplot(data_base, aes(x = normalised_score, y = bertbasemultiuncased_label, color = bertbasemultiuncased_score)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1) +
  geom_count(alpha = 0.5)
plot_base

################################################################################
library(Metrics)
true_scores <- data_base$normalised_score
pred_scores <- data_base$bertbasemultiuncased_label
base_mse <- mse(true_scores, pred_scores)

base_mse
################################################################################
# Check original-score with predicted-score for hit
data_base <- data_sentiments_normalised %>% 
  dplyr::select(normalised_score, bertbasemultiuncased_label, bertbasemultiuncased_score)
data_base$bertbase_hit <- with(data_base, ifelse(normalised_score == bertbasemultiuncased_label, 1, 0))

summary(data_base)

plot_bertbase <- ggplot(data_base, aes(x = normalised_score, y = bertbasemultiuncased_label)) +
  geom_count(alpha = 0.5)
plot_bertbase

data_test <- head(data_base, 1000)
cor(data_test$normalised_score, data_test$bertbasemultiuncased_label, method = "kendall")
# TODO: Very long running time for the entire data... 
#cor(data_base$normalised_score, data_base$bertbasemultiuncased_label, method = "kendall")
#cor(data_base$normalised_score, data_base$bertbasemultiuncased_label, method = "spearman")

oneway_anova <- aov(bertbase_hit ~ bertbasemultiuncased_score + normalised_score,
                    data = data_test)

summary(oneway_anova)


res_aov <- aov(normalised_score ~ bertbasemultiuncased_label,
               data = data_base)

hist(res_aov$residuals)


################################################################################
# Check original-score with predicted-score for hit
data_base <- data_sentiments_normalised %>% 
  dplyr::select(normalised_score, bertbasemultiuncased_label, bertbasemultiuncased_score)
data_base$bertbase_hit <- with(data_base, ifelse(normalised_score == bertbasemultiuncased_label, 'Hit', 'Miss'))

data_dist <- data_sentiments_normalised %>% 
  dplyr::select(normalised_pone, distilbertbaseuncasedsst2_label, distilbertbaseuncasedsst2_score)  
data_dist$distbert_hit <- with(data_dist, ifelse(normalised_pone == distilbertbaseuncasedsst2_label, 'Hit', 'Miss'))

# Quick Fix
table(data_base$bertbase_hit)
table(data_dist$distbert_hit)

#### Output preparation
setwd(dir_out)
identifier1 <- "bertbasemultiuncased"
identifier2 <- "distilbertbaseuncasedsst2"
sink_path1 <- paste0(identifier1, "-summary-statistics.txt")
sink_path2 <- paste0(identifier2, "-summary-statistics.txt")


summary_statistics_bertbase <- data_base %>% 
  get_summary_stats(normalised_score, bertbasemultiuncased_label, bertbase_hit, type ="common")
#group_by(normalised_score) %>% 
#get_summary_stats(bertbase_hit, type ="common")

sink(sink_path1, append = FALSE)
print("Summary Statistics for bertbasemultiuncased_label")
print(summary_statistics_bertbase)
sink()

summary_statistics_distbert <- data %>% 
  group_by(normalised_pone) %>%
  get_summary_stats(distilbertbaseuncasedsst2_label, type = "common")

sink(sink_path2, append = FALSE)
print("Summary Statistics for distilbertbaseuncasedsst2_label")
print(summary_statistics_distbert)
sink()

#### Visualization
bertbase_plot <- ggboxplot(data, x = "normalised_score", y = "bertbasemultiuncased_label", palette = "jco")
distbert_plot <- ggboxplot(data, x = "normalised_pone", y = "distilbertbaseuncasedsst2_label", palette = "jco")

png(paste0(identifier1, "-summary-statistics.png"))
bertbase_plot
dev.off()

png(paste0(identifier2, "-summary-statistics.png"))
distbert_plot
dev.off()

################################################################################

#data_all %>% group_by(dataset_id) %>% summarise(mean = mean(review_length), n = n())

# Get the length for each review (content)
# Encoding function to prevent "Error: invalid multibyte string"
#data_all$review_content <- iconv(enc2utf8(data_all$review_content),sub="byte")
#stringr::str_conv(data_all$review_content, "UTF-8") # Did not work
#data_all$review_length <- nchar(data_all$review_content)

################################################################################
exploreData <- function(data, name, title, dim, head, glimpse, summary, report) {
  # Dimensions
  if (dim == TRUE) {
    message(dim(data))
  }
  # Head
  if (head == TRUE) {
    #message(head(data))
    
    #samples <- data[1:3,]
    #sink(file = paste0(name, ".txt"), append = FALSE)
    #print(samples)
    #sink()
    
    #write.table(data[1:3,], file = paste0(name, ".txt"), sep = "\t", quote = FALSE, row.names = TRUE)
    
    png(paste0(name, ".png"), height = 50*3, width = 200*20)
    grid.table(data[1:3,])
    dev.off()
  }
  # Glimpse
  if (glimpse == TRUE) {
    #message(dplyr::glimpse(data))
    samples <- dplyr::glimpse(data)
    sink(file = paste0(name, ".txt"), append = FALSE)
    print(samples)
    sink()
  }
  # Summary (especially useful for numeric attributes)
  if (summary == TRUE) {
    message(summary(data))
  }
  # DataExplorer Report
  # https://www.rdocumentation.org/packages/DataExplorer/versions/0.8.2/topics/create_report
  if (report == TRUE) {
    filename <- paste0("report-", name, ".html")
    DataExplorer::create_report(
      data = data, 
      #output_format = html_document(toc = TRUE, toc_depth = 6, theme = "flatly"),
      output_file = filename,
      report_title = title,
      #output_dir = getwd(),
      y = "normalised_score",
      #config = configure_report(
      #  "introduce" = list(),
      #  "plot_intro" = list(),
      #...
      #)
    )
  }
}

dim <- FALSE
head <- FALSE
glimpse <- FALSE
summary <- FALSE 
report <- TRUE

################################################################################
#setwd(dir_input)
################################################################################
# Predicted Sentiment and normalised Scores from (roto) Review Data
#data_sentiments <- read.csv(file = "data-sentiment.csv")
#data_sentiments_normalised <- read.csv(file = "data-sentiment-normalised.csv")
setwd(dir_out)

data <- drop_columns(data_sentiments_normalised, "ID")
#data <- drop_columns(data, "Unnamed..0")
#data <- drop_columns(data, "X")
#
data <- drop_columns(data, "distilbertbaseuncasedsst2_score")
data <- drop_columns(data, "bertbasemultiuncased_score")

exploreData(data, "sent-roto", "Normalised Scores and Sentiment Prediction", dim, head, glimpse, summary, report)
#exploreData(data_sentiments, "imbd", "IMBD Dataset", dim, head, glimpse, summary, report)


################################################################################
# Aggregated data
exploreData(data_all, "all", "Aggregated Datasets", dim, head, glimpse, summary, report)


################################################################################
# Selected data (First try)
data <- data_roto_crit %>% dplyr::select("rotten_tomatoes_link", "review_score", "review_content")
# Count reviews per movie
data <- data %>% 
  group_by(rotten_tomatoes_link) %>% 
  mutate(n = n())
# Join to include variables from the movi-data
data_movi <- data_roto_movi %>% dplyr::select("rotten_tomatoes_link", "movie_title")#, "tomatometer_rotten_critics_count")
data <- data %>% left_join(data_movi, by = "rotten_tomatoes_link")

# Numbers
print(paste0("Number of Movies in movies: ", nrow(data_roto_movi)))
print(paste0("Number of Movies in reviews: ", length(unique(data_roto_crit[["rotten_tomatoes_link"]]))))
print(paste0("Mean of Reviews per Movie : ", mean(data$n)))



exploreData(data_imbd, "imbd", "IMBD Dataset", dim, head, glimpse, summary, report)

exploreData(data_roto_movi, "roto-movi", "Rotten Tomatoes Movies", dim, head, glimpse, summary, report)

exploreData(data_roto_crit, "roto-crit", "Rotten Tomatoes Critic", dim, head, glimpse, summary, report)

################################################################################


################################################################################

kruskall_object <- function() {
  #### Output preparation
  identifier1 <- "bertbasemultiuncased"
  identifier2 <- "distilbertbaseuncasedsst2"
  sink_path_1 <- paste0(identifier1, "-sink.txt")
  sink_path_2 <- paste0(identifier2, "-sink.txt")
  pdf_boolean <- TRUE
  png_boolean <- TRUE
  options(max.print=100000)
  options(dplyr.print_max = 100000)
  
  #### Kruskal-Wallis test  ######################################################
  #### Data preparation
  data <- data_sentiments_normalised %>% 
    select(ID, review_score, normalised_score, bertbasemultiuncased_label, distilbertbaseuncasedsst2_label)  
  #### Summary statistics
  summary_statistics_grade <- data %>% 
    group_by(normalised_score) %>%
    get_summary_stats(bertbasemultiuncased_label, type = "common")
  summary_statistics_rank <- data %>% 
    group_by(normalised_score) %>%
    get_summary_stats(distilbertbaseuncasedsst2_label, type = "common")
  
  sink(sink_path1, append = FALSE)
  print("Summary Statistics for bertbasemultiuncased_label")
  print(summary_statistics_grade)
  sink()
  
  sink(sink_path2, append = FALSE)
  print("Summary Statistics for distilbertbaseuncasedsst2_label")
  print(summary_statistics_rank)
  sink()
  
  #### Visualization
  ggboxplot(data, x = "normalised_score", y = "bertbasemultiuncased_label", palette = "jco")
  ggboxplot(data, x = "normalised_score", y = "distilbertbaseuncasedsst2_label", palette = "jco")
  
  #### Computation
  resGrade.kruskal <- data %>% kruskal_test(bertbasemultiuncased_label ~ normalised_score)
  resRank.kruskal <- data %>% kruskal_test(distilbertbaseuncasedsst2_label ~ normalised_score)
  sink(sink_path1, append = TRUE)
  print("######## Question: We want to know if there is any significant difference between the average ratings of items in the experimental conditions. ########")
  print("######## Kruskal Computation for bertbasemultiuncased_label ########")
  print(resGrade.kruskal)
  sink()
  
  sink(sink_path2, append = TRUE)
  print("######## Question: We want to know if there is any significant difference between the average ratings of items in the experimental conditions. ########")
  print("######## Kruskal Computation for distilbertbaseuncasedsst2_label ########")
  print(resRank.kruskal)
  sink()
  
  #### Effect size
  kruskal_effectsize_grade <- data %>% kruskal_effsize(bertbasemultiuncased_label ~ normalised_score)
  kruskal_effectsize_rank <- data %>% kruskal_effsize(distilbertbaseuncasedsst2_label ~ normalised_score)
  sink(sink_path1, append = TRUE)
  print("######## The eta squared, based on the H-statistic, can be used as the measure of the Kruskal-Wallis test effect size. ########")
  print("######## It is calculated as follow : eta2[H] = (H - k + 1)/(n - k); where H is the value obtained in the Kruskal-Wallis test; ########")
  print("######## k is the number of groups; n is the total number of observations (M. T. Tomczak and Tomczak 2014). ########")
  print("######## The eta-squared estimate assumes values from 0 to 1 and multiplied by 100 indicates the percentage of variance in the dependent variable explained by the independent variable. ########")
  print("######## The interpretation values commonly in published literature are: ########")
  print("######## 0.01- < 0.06 (small effect), 0.06 - < 0.14 (moderate effect) and >= 0.14 (large effect). ########")
  print("######## Kruskal Effect Size for bertbasemultiuncased_label ########")
  print(kruskal_effectsize_grade)
  sink()
  
  sink(sink_path2, append = TRUE)
  print("######## The eta squared, based on the H-statistic, can be used as the measure of the Kruskal-Wallis test effect size. ########")
  print("######## It is calculated as follow : eta2[H] = (H - k + 1)/(n - k); where H is the value obtained in the Kruskal-Wallis test; ########")
  print("######## k is the number of groups; n is the total number of observations (M. T. Tomczak and Tomczak 2014). ########")
  print("######## The eta-squared estimate assumes values from 0 to 1 and multiplied by 100 indicates the percentage of variance in the dependent variable explained by the independent variable. ########")
  print("######## The interpretation values commonly in published literature are: ########")
  print("######## 0.01- < 0.06 (small effect), 0.06 - < 0.14 (moderate effect) and >= 0.14 (large effect). ########")
  print("######## Kruskal Effect Size for Rank ########")
  print(kruskal_effectsize_rank)
  sink()
  
  #### Multiple pairwise-comparisons
  #### Pairwise comparisons using Dunn’s test:
  pwc_dunn_grade <- data %>% 
    dunn_test(bertbasemultiuncased_label ~ normalised_score, p.adjust.method = "bonferroni") 
  pwc_dunn_rank <- data %>% 
    dunn_test(distilbertbaseuncasedsst2_label ~ normalised_score, p.adjust.method = "bonferroni") 
  sink(sink_path1, append = TRUE)
  print("######## From the output of the Kruskal-Wallis test, we know that there is a significant difference between groups, but we don’t know which pairs of groups are different. ########")
  print("######## A significant Kruskal-Wallis test is generally followed up by Dunn’s test to identify which groups are different. ########")
  print("######## Compared to the Wilcoxon’s test, the Dunn’s test takes into account the rankings used by the Kruskal-Wallis test. It also does ties adjustments. ########")
  print("######## Pairwise Comparisons Dunn Test for bertbasemultiuncased_label ########")
  print(pwc_dunn_grade)
  sink()
  
  sink(sink_path2, append = TRUE)
  print("######## From the output of the Kruskal-Wallis test, we know that there is a significant difference between groups, but we don’t know which pairs of groups are different. ########")
  print("######## A significant Kruskal-Wallis test is generally followed up by Dunn’s test to identify which groups are different. ########")
  print("######## Compared to the Wilcoxon’s test, the Dunn’s test takes into account the rankings used by the Kruskal-Wallis test. It also does ties adjustments. ########")
  print("######## Pairwise Comparisons Dunn Test for distilbertbaseuncasedsst2_label ########")
  print(pwc_dunn_rank)
  sink()
  
  #### Pairwise comparisons using Wilcoxon’s test:
  pwc_wilcox_grade <- data %>% 
    wilcox_test(bertbasemultiuncased_label ~ normalised_score, p.adjust.method = "bonferroni")
  pwc_wilcox_rank <- data %>% 
    wilcox_test(distilbertbaseuncasedsst2_label ~ normalised_score, p.adjust.method = "bonferroni")
  sink(sink_path1, append = TRUE)
  print("######## It’s also possible to use the Wilcoxon’s test to calculate pairwise comparisons between group levels with corrections for multiple testing. ########")
  print("######## Pairwise Comparisons Wilcox Test for bertbasemultiuncased_label ########")
  print(pwc_wilcox_grade)
  sink()
  
  sink(sink_path2, append = TRUE)
  print("######## It’s also possible to use the Wilcoxon’s test to calculate pairwise comparisons between group levels with corrections for multiple testing. ########")
  print("######## Pairwise Comparisons Wilcox Test for distilbertbaseuncasedsst2_label ########")
  print(pwc_wilcox_rank)
  sink()
  
  #### Report
  # Visualization: box plots with p-values
  # For Dunn test
  pwc_dunn_grade <- pwc_dunn_grade %>% add_xy_position(x = "normalised_score")
  plot_pwc_dunn_grade <- ggboxplot(data, x = "normalised_score", y = "bertbasemultiuncased_label") +
    stat_pvalue_manual(pwc_dunn_grade, hide.ns = TRUE) +
    labs(
      subtitle = get_test_label(resGrade.kruskal, detailed = TRUE),
      caption = get_pwc_label(pwc_dunn_grade)
    )
  # For Wilcox test
  pwc_wilcox_grade <- pwc_wilcox_grade %>% add_xy_position(x = "normalised_score")
  plot_pwc_wilcox_grade <- ggboxplot(data, x = "normalised_score", y = "bertbasemultiuncased_label") +
    stat_pvalue_manual(pwc_wilcox_grade, hide.ns = TRUE) +
    labs(
      subtitle = get_test_label(resGrade.kruskal, detailed = TRUE),
      caption = get_pwc_label(pwc_wilcox_grade)
    )
  
  #### Report
  # Visualization: box plots with p-values
  # For Dunn test
  pwc_dunn_rank <- pwc_dunn_rank %>% add_xy_position(x = "normalised_score")
  plot_pwc_dunn_rank <- ggboxplot(data, x = "normalised_score", y = "distilbertbaseuncasedsst2_label") +
    stat_pvalue_manual(pwc_dunn_rank, hide.ns = TRUE) +
    labs(
      subtitle = get_test_label(resRank.kruskal, detailed = TRUE),
      caption = get_pwc_label(pwc_dunn_rank)
    )
  # For Wilcox test
  pwc_wilcox_rank <- pwc_wilcox_rank %>% add_xy_position(x = "normalised_score")
  plot_pwc_wilcox_rank <- ggboxplot(data, x = "normalised_score", y = "distilbertbaseuncasedsst2_label") +
    stat_pvalue_manual(pwc_wilcox_rank, hide.ns = TRUE) +
    labs(
      subtitle = get_test_label(resRank.kruskal, detailed = TRUE),
      caption = get_pwc_label(pwc_wilcox_rank)
    )
  
  # Saving data to pdf or png ####################################################
  ################################################################################
  if (png_boolean == TRUE) { # Print plots to sequence of png files:
    png(
      file = paste0("bertbasemultiuncased_label", "-pwc-dunn.png"),
    )
    print(plot_pwc_dunn_grade)
    dev.off()
    png(
      file = paste0("bertbasemultiuncased_label", "-pwc-wilcox.png"),
    )
    print(plot_pwc_wilcox_grade)
    dev.off()
    
    png(
      file = paste0("distilbertbaseuncasedsst2_label", "-pwc-dunn.png"),
    )
    print(plot_pwc_dunn_rank)
    dev.off()
    png(
      file = paste0("distilbertbaseuncasedsst2_label", "-pwc-wilcox.png"),
    )
    print(plot_pwc_wilcox_rank)
    dev.off()
  }
}

################################################################################
main <- function() {
  kruskall_object()
}

main()

################################################################################

setwd(dir_init)