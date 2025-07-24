
# Reproducible R Environment Setup
# Generated for session: Quantum Computing Statistical Analysis - Multi-University Partnership
# Created: 2025-07-24T16:28:36.948958

# Install renv if not available
if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv")
}

# Initialize renv project
renv::init()

# Restore packages from lockfile
renv::restore()

# Load essential packages
library(tidyverse)
library(rmarkdown)
library(knitr)

# Set working directory
setwd("rstudio_collaboration/sessions/3a13232a-ae08-4859-86ba-82518fc682c6")

# Configure collaboration settings
options(repos = c(CRAN = "https://cran.rstudio.com/"))
options(warn = 1)  # Show warnings immediately

# Print session information
sessionInfo()
cat("\n✅ Reproducible environment ready for collaboration!\n")
