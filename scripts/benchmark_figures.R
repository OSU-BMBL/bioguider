#!/usr/bin/env Rscript
# scripts/benchmark_figures.R
# Generate 4 publication figures from BioGuider benchmark CSVs
#
# Required packages (install if missing):
# install.packages(c("ggplot2", "dplyr", "tidyr", "patchwork", "viridis", "scales", "optparse", "RColorBrewer"))

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
  library(viridis)
  library(scales)
  library(optparse)
  library(RColorBrewer)
})

# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
option_list <- list(
  make_option(
    c("--multi-run-dir"),
    type = "character",
    default = NULL,
    help = "Path to multi_file_stress run directory (contains _aggregate/). Defaults to latest run."
  ),
  make_option(
    c("--single-run-dir"),
    type = "character",
    default = NULL,
    help = "Path to single_file_stress run directory (contains SKILL_COMPARISON.csv). Defaults to latest run."
  ),
  make_option(
    c("--outdir"),
    type = "character",
    default = "outputs/figures",
    help = "Output directory for figures [default: outputs/figures]"
  )
)

opt_parser <- OptionParser(option_list = option_list,
                            description = "Generate BioGuider benchmark publication figures")
opt        <- parse_args(opt_parser)

# ---------------------------------------------------------------------------
# Resolve run directories (latest if not specified)
# ---------------------------------------------------------------------------
find_latest_run <- function(base_dir, required_file = NULL) {
  if (!dir.exists(base_dir)) stop("Directory not found: ", base_dir)
  runs <- list.dirs(base_dir, recursive = FALSE, full.names = TRUE)
  runs <- runs[grepl("run_", basename(runs))]
  if (length(runs) == 0) stop("No run_* directories found under: ", base_dir)
  runs <- runs[order(basename(runs), decreasing = TRUE)]
  if (!is.null(required_file)) {
    runs <- runs[file.exists(file.path(runs, required_file))]
    if (length(runs) == 0)
      stop("No run_* directory under '", base_dir, "' contains: ", required_file)
  }
  runs[1]
}

# Determine repo root: script lives in <repo>/scripts/, so parent is repo root.
# commandArgs(trailingOnly=FALSE) contains --file=<path> when invoked via Rscript.
script_args <- commandArgs(trailingOnly = FALSE)
script_file  <- sub("--file=", "", script_args[grep("--file=", script_args)])
if (length(script_file) > 0 && nchar(script_file) > 0) {
  # Resolve relative paths against getwd() before going up one level
  script_abs <- if (startsWith(script_file, "/")) script_file else file.path(getwd(), script_file)
  repo_root  <- normalizePath(file.path(dirname(script_abs), ".."), mustWork = FALSE)
} else {
  repo_root <- getwd()
}

multi_run_dir  <- opt$`multi-run-dir`
single_run_dir <- opt$`single-run-dir`

if (is.null(multi_run_dir)) {
  multi_run_dir <- find_latest_run(
    file.path(repo_root, "outputs", "multi_file_stress"),
    required_file = file.path("_aggregate", "AGGREGATE_TABLE.csv")
  )
}
if (is.null(single_run_dir)) {
  single_run_dir <- find_latest_run(
    file.path(repo_root, "outputs", "single_file_stress"),
    required_file = "SKILL_COMPARISON.csv"
  )
}

outdir <- opt$outdir
if (!startsWith(outdir, "/")) outdir <- file.path(repo_root, outdir)
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

message("Multi-file run:  ", multi_run_dir)
message("Single-file run: ", single_run_dir)
message("Output dir:      ", outdir)

# ---------------------------------------------------------------------------
# Load CSVs
# ---------------------------------------------------------------------------
agg_table_path    <- file.path(multi_run_dir, "_aggregate", "AGGREGATE_TABLE.csv")
agg_cat_path      <- file.path(multi_run_dir, "_aggregate", "AGGREGATE_CATEGORY_DETAIL.csv")
skill_cmp_path    <- file.path(single_run_dir, "SKILL_COMPARISON.csv")

# Prefer the richer skill matrix (E003: 5 files x 3 levels x 2 prompts) when present.
# Fall back to the single-file SKILL_COMPARISON.csv otherwise.
skill_matrix_path <- find_latest_run(
  file.path(repo_root, "outputs", "single_file_stress"),
  required_file = "SKILL_MATRIX_TABLE.csv"
)
if (!is.null(skill_matrix_path)) {
  skill_matrix_csv <- file.path(skill_matrix_path, "SKILL_MATRIX_TABLE.csv")
  message("Skill matrix run: ", skill_matrix_path)
} else {
  skill_matrix_csv <- NULL
}

for (p in c(agg_table_path, agg_cat_path, skill_cmp_path)) {
  if (!file.exists(p)) stop("Required file not found: ", p)
}

agg_raw     <- read.csv(agg_table_path,  stringsAsFactors = FALSE)
agg_cat     <- read.csv(agg_cat_path,    stringsAsFactors = FALSE)
skill_cmp   <- read.csv(skill_cmp_path,  stringsAsFactors = FALSE)
skill_mat   <- if (!is.null(skill_matrix_csv)) read.csv(skill_matrix_csv, stringsAsFactors = FALSE) else NULL

# Split "model+prompt" into separate columns (e.g. "gpt-4o+bioguider")
agg_raw <- agg_raw %>%
  tidyr::separate(model, into = c("model_name", "prompt"), sep = "\\+", extra = "merge", fill = "right") %>%
  mutate(
    model_name = trimws(model_name),
    prompt     = trimws(prompt)
  )

agg_cat <- agg_cat %>%
  tidyr::separate(model, into = c("model_name", "prompt"), sep = "\\+", extra = "merge", fill = "right") %>%
  mutate(
    model_name = trimws(model_name),
    prompt     = trimws(prompt)
  )

# ---------------------------------------------------------------------------
# Model ordering (E001 ranking: best to worst average F1)
# ---------------------------------------------------------------------------
MODEL_ORDER <- c("gpt-4o", "kimi-k2.5", "glm-5", "gpt-5.4", "gpt-oss")

# Keep only models present in data, append any extras
present_models <- unique(agg_raw$model_name)
model_order    <- c(MODEL_ORDER[MODEL_ORDER %in% present_models],
                    setdiff(present_models, MODEL_ORDER))

ERROR_LEVELS <- sort(unique(agg_raw$error_count))

# Apply factor ordering
agg_raw <- agg_raw %>%
  mutate(
    model_name  = factor(model_name,  levels = model_order),
    error_count = factor(error_count, levels = ERROR_LEVELS)
  )

# ---------------------------------------------------------------------------
# Shared theme
# ---------------------------------------------------------------------------
pub_theme <- theme_minimal(base_size = 10) +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text        = element_text(size = 8),
    axis.title       = element_text(size = 9),
    plot.title       = element_text(size = 10, face = "bold"),
    plot.subtitle    = element_text(size = 8, color = "grey40"),
    legend.text      = element_text(size = 8),
    legend.title     = element_text(size = 9),
    strip.text       = element_text(size = 9, face = "bold")
  )

heatmap_fill <- scale_fill_gradientn(
  colours = RColorBrewer::brewer.pal(11, "RdYlGn"),
  limits  = c(0, 1),
  name    = "F1"
)

# ---------------------------------------------------------------------------
# Helper: build a heatmap data frame + plot
# ---------------------------------------------------------------------------
build_heatmap <- function(df, value_col, title, subtitle = NULL) {
  plot_df <- df %>%
    select(model_name, error_count, value = all_of(value_col)) %>%
    mutate(
      label      = sprintf("%.2f", value),
      text_color = ifelse(value > 0.65, "black", "white")
    )

  p <- ggplot(plot_df, aes(x = error_count, y = model_name, fill = value)) +
    geom_tile(color = "white", linewidth = 0.4) +
    geom_text(aes(label = label, color = text_color), size = 2.5, show.legend = FALSE) +
    scale_color_identity() +
    scale_fill_gradientn(
      colours = RColorBrewer::brewer.pal(11, "RdYlGn"),
      limits  = c(0, 1),
      name    = "F1"
    ) +
    scale_y_discrete(limits = rev(levels(plot_df$model_name))) +
    labs(
      title    = title,
      subtitle = subtitle,
      x        = "Error count",
      y        = NULL
    ) +
    pub_theme +
    theme(
      axis.text.x  = element_text(angle = 45, hjust = 1),
      panel.border = element_rect(color = "grey80", fill = NA, linewidth = 0.5)
    )
  p
}

# ---------------------------------------------------------------------------
# Figure 1: Model Selection Heatmap — F1 (scorable)
# ---------------------------------------------------------------------------
message("Building Figure 1...")

fig1 <- build_heatmap(
  df        = agg_raw,
  value_col = "f1_score_scorable",
  title     = "Model Selection: F1 (scorable) by Error Level"
)

save_figure <- function(plot, name, width_mm, height_mm = 90) {
  w_in <- width_mm / 25.4
  h_in <- height_mm / 25.4
  pdf_path <- file.path(outdir, paste0(name, ".pdf"))
  png_path <- file.path(outdir, paste0(name, ".png"))
  ggsave(pdf_path, plot = plot, width = w_in, height = h_in, device = "pdf")
  ggsave(png_path, plot = plot, width = w_in, height = h_in, dpi = 300, device = "png")
  message("  Saved: ", pdf_path)
  message("  Saved: ", png_path)
}

save_figure(fig1, "fig1_model_selection_heatmap", width_mm = 85, height_mm = 70)

# ---------------------------------------------------------------------------
# Figure 2: CONTENT vs HYGIENE Side-by-Side (double column)
# ---------------------------------------------------------------------------
message("Building Figure 2...")

fig2a <- build_heatmap(
  df        = agg_raw,
  value_col = "f1_score_content",
  title     = "A: CONTENT F1"
)

fig2b <- build_heatmap(
  df        = agg_raw,
  value_col = "f1_score_hygiene",
  title     = "B: HYGIENE F1"
)

fig2 <- (fig2a | fig2b) +
  plot_annotation(
    title   = "CONTENT vs HYGIENE F1 by Model and Error Level",
    caption = "All models near-perfect on CONTENT (~0.92); differentiation comes from HYGIENE",
    theme   = theme(
      plot.title   = element_text(size = 10, face = "bold"),
      plot.caption = element_text(size = 8, color = "grey40", hjust = 0)
    )
  ) &
  theme(legend.position = "right")

save_figure(fig2, "fig2_content_vs_hygiene", width_mm = 170, height_mm = 75)

# ---------------------------------------------------------------------------
# Figure 3: Skill Comparison — grouped bar chart
# ---------------------------------------------------------------------------
message("Building Figure 3...")

# Prefer the matrix data (5 files x 3 levels x 2 prompts) for cross-file SD.
skill_source <- if (!is.null(skill_mat)) skill_mat else skill_cmp

skill_plot_df <- skill_source %>%
  mutate(
    skill = factor(skill, levels = c("bioguider", "skill_generic"),
                   labels = c("BioGuider", "Generic"))
  )

skill_colors <- c("BioGuider" = "#2166AC", "Generic" = "#D6604D")

if (!is.null(skill_mat)) {
  # E003 path: per-vignette paired dot plot.
  # Each (vignette, error_level) gives a paired BioGuider/Generic point pair
  # connected by a thin line — slope direction shows which prompt won that cell.
  paired <- skill_plot_df %>%
    select(file_stem, error_count, skill, f1_score_scorable) %>%
    mutate(error_count_f = factor(error_count))

  # Mean per (skill, level) for the heavier diamond marker
  level_means <- paired %>%
    group_by(error_count_f, skill) %>%
    summarise(mean_f1 = mean(f1_score_scorable, na.rm = TRUE), .groups = "drop")

  # Wide format for the connecting segments (paired by vignette x level)
  paired_wide <- paired %>%
    tidyr::pivot_wider(names_from = skill, values_from = f1_score_scorable)

  # Compute small dodge so BG and Generic don't overlap on x
  dodge <- 0.18
  paired <- paired %>%
    mutate(x_pos = as.numeric(error_count_f) + ifelse(skill == "BioGuider", -dodge, dodge))
  paired_wide <- paired_wide %>%
    mutate(
      x_bg  = as.numeric(error_count_f) - dodge,
      x_gen = as.numeric(error_count_f) + dodge
    )
  level_means <- level_means %>%
    mutate(x_pos = as.numeric(error_count_f) + ifelse(skill == "BioGuider", -dodge, dodge))

  fig3 <- ggplot() +
    # Pair-connecting segments: slope shows winner per (vignette, level) cell
    geom_segment(
      data = paired_wide,
      aes(x = x_bg, xend = x_gen,
          y = BioGuider, yend = Generic),
      colour = "grey60", linewidth = 0.4, alpha = 0.7
    ) +
    # Per-vignette dots
    geom_point(
      data = paired,
      aes(x = x_pos, y = f1_score_scorable, colour = skill),
      size = 2.6, alpha = 0.85
    ) +
    # Mean per (skill, level) — heavier diamond
    geom_point(
      data = level_means,
      aes(x = x_pos, y = mean_f1, fill = skill),
      shape = 23, size = 4, colour = "black", stroke = 0.6
    ) +
    scale_colour_manual(values = skill_colors, name = "Prompt") +
    scale_fill_manual(values = skill_colors, guide = "none") +
    scale_x_continuous(
      breaks = seq_along(levels(paired$error_count_f)),
      labels = levels(paired$error_count_f),
      expand = expansion(add = 0.4)
    ) +
    scale_y_continuous(limits = c(0, 1.05), expand = c(0, 0),
                       labels = scales::number_format(accuracy = 0.01)) +
    labs(
      title    = "Skill Comparison: BioGuider vs Generic (per vignette)",
      subtitle = "Each thin grey line = one vignette × error-level pair; diamond = mean across files; gpt-4o",
      x        = "Errors injected",
      y        = "F1 (scorable)"
    ) +
    pub_theme +
    theme(legend.position = "top")
} else {
  # Single-file fallback (E002-v2)
  skill_summary <- skill_plot_df %>%
    group_by(model, skill) %>%
    summarise(
      mean_f1  = mean(f1_score_scorable, na.rm = TRUE),
      sd_f1    = sd(f1_score_scorable,   na.rm = TRUE),
      mean_dur = mean(duration_s,        na.rm = TRUE),
      .groups  = "drop"
    ) %>%
    mutate(
      sd_f1     = ifelse(is.na(sd_f1), 0, sd_f1),
      dur_label = sprintf("%.0fs", mean_dur)
    )

  fig3 <- ggplot(skill_summary,
                 aes(x = model, y = mean_f1, fill = skill)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6, alpha = 0.9) +
    geom_errorbar(
      aes(ymin = mean_f1 - sd_f1, ymax = mean_f1 + sd_f1),
      position = position_dodge(width = 0.7),
      width    = 0.2,
      linewidth = 0.5
    ) +
    geom_text(
      aes(label = dur_label, y = mean_f1 + sd_f1 + 0.015),
      position  = position_dodge(width = 0.7),
      size      = 2.5,
      color     = "grey30"
    ) +
    scale_fill_manual(values = skill_colors, name = "Prompt") +
    scale_y_continuous(limits = c(0, 1.05), expand = c(0, 0),
                       labels = scales::number_format(accuracy = 0.01)) +
    labs(
      title    = "Skill Comparison: BioGuider vs Generic Prompt",
      subtitle = "Single file; labels show mean duration",
      x        = "Model",
      y        = "F1 (scorable)"
    ) +
    pub_theme +
    theme(legend.position = "top")
}

save_figure(fig3, "fig3_skill_comparison", width_mm = 85, height_mm = 75)

# ---------------------------------------------------------------------------
# Figure 4: F1 Degradation Curve
# ---------------------------------------------------------------------------
message("Building Figure 4...")

degrade_df <- agg_raw %>%
  mutate(error_count_num = as.numeric(as.character(error_count)))

# Okabe-Ito colorblind-safe palette (enough for 5 models)
okabe_ito <- c(
  "#E69F00", "#56B4E9", "#009E73",
  "#CC79A7", "#0072B2"
)
model_colors <- setNames(okabe_ito[seq_along(model_order)], model_order)

fig4 <- ggplot(degrade_df,
               aes(x = error_count_num, y = f1_score_scorable,
                   color = model_name, group = model_name)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  scale_color_manual(values = model_colors, name = "Model") +
  scale_x_continuous(
    breaks = ERROR_LEVELS,
    labels = as.character(ERROR_LEVELS)
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    labels = scales::number_format(accuracy = 0.01)
  ) +
  labs(
    title    = "F1 Degradation as Error Count Increases",
    subtitle = "F1 (scorable metric) across all models and error levels",
    x        = "Error count",
    y        = "F1 (scorable)"
  ) +
  pub_theme +
  theme(
    legend.position = "right",
    axis.text.x     = element_text(angle = 45, hjust = 1)
  )

save_figure(fig4, "fig4_f1_degradation", width_mm = 85, height_mm = 75)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message("\nAll figures saved to: ", outdir)
figs <- list.files(outdir, pattern = "^fig[1-4].*\\.(pdf|png)$", full.names = FALSE)
message("Files produced (", length(figs), "):")
for (f in sort(figs)) message("  ", f)
