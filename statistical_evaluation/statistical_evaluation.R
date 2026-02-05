#' ---
#' title: "Statistical assessment of 2-OMF algorithm comparison"
#' author: "Tommaso Mannelli Mazzoli"
#' date: "2026-02-05"
#' ---
#'
#' This script performs the statistical assessment of empirical results
#' comparing algorithms for the 2-Optimality Motif Finding problem.
#'
#' Dependencies:
#'   install.packages("devtools")
#'   devtools::install_github("b0rxa/scmamp")
#'   install.packages("ggplot2")

# Load data ---------------------------

library(scmamp)
library(ggplot2)

# Set up variables
data.file <- "results_balanced.csv"
delimiter <- ","
plot.dir <- "./"
results.file <- "statistical_results.txt"
significance_level <- 0.05

#' First of all, we load the results.

data <- read.csv(data.file, sep = delimiter, header = TRUE, check.names = FALSE)

#' Now we create a summarization table with the average results per instance

average.function <- mean
to.ignore <- c("inst")

summary.data <- summarizeData(data = data, fun = average.function,
                               ignore = to.ignore)

alg.columns <- c("SA", "Gurobi", "HG2")

# Open results file for writing
sink(results.file)

cat("================================================================\n")
cat("  2-OMF Statistical Evaluation Results\n")
cat("  Date:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("  Data:", data.file, "\n")
cat("  Significance level:", significance_level, "\n")
cat("================================================================\n\n")

#' ## Descriptive statistics

cat("--- DESCRIPTIVE STATISTICS ---\n\n")
cat("Summary:\n")
print(summary(data[, alg.columns]))
cat("\nStandard Deviations:\n")
print(sapply(data[, alg.columns], sd))
cat("\nMean Ranks:\n")
print(colMeans(apply(data[, alg.columns], 1, rank)))
cat("\n")

#################
#' OMNIBUS TEST #
#################
#' First, verify that there is at least one algorithm that is different from the rest.
#' The obtained p-values indicate that we can safely reject the null hypothesis
#' that all the algorithms perform the same.

cat("--- OMNIBUS TEST (Quade) ---\n\n")
quade_test <- quadeTest(data[, alg.columns])
print(quade_test)
cat("\n")

########################
#' POST-HOC COMPARISON #
########################

#' All-vs-all post-hoc comparison with Bergmann-Hommel correction.
#' Set decreasing=FALSE because the best results are the smallest.

cat("--- POST-HOC TEST (Quade + Bergmann-Hommel) ---\n\n")
all_vs_all <- postHocTest(data = data, algorithms = alg.columns, test = "quade",
                          control = NULL, use.rank = TRUE,
                          correct = "bergmann",
                          alpha = significance_level, decreasing = FALSE)

cat("Mean ranks:\n")
print(all_vs_all$summary)
cat("\nCorrected p-values:\n")
print(all_vs_all$corrected.pval)
cat("\n")

# Close results file
sink()

cat("Statistical results saved to:", results.file, "\n")

# Also save p-value matrix as CSV for programmatic access
write.csv(all_vs_all$corrected.pval, file = "pvalues_corrected.csv")

#' ## Plots
#'
#' Critical difference plot (Demsar, 2006)

pdf(file = paste0(plot.dir, "CD_plot_balanced.pdf"), width = 10, height = 3)
plotRanking(all_vs_all$corrected.pval, summary = all_vs_all$summary,
            alpha = significance_level, cex = 1.5)
dev.off()

cat("CD plot saved to: CD_plot_balanced.pdf\n")

# For reproducibility
sessionInfo()

#' ## References
#' Demsar, J. (2006) Statistical Comparisons of Classifiers over Multiple
#' Data Sets. _Journal of Machine Learning Research_, 7, 1-30.
