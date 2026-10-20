"""Reusable NEJM-style plotting blocks.

Each block draws onto a Matplotlib Axes you supply, so any of them can be dropped into
a new figure layout:

    style      use_nejm_style, InchLayout, add_y_break, COLORS, MRS_COLORS
    km         kaplan_meier, logrank_test, KMArm, plot_km, add_at_risk_table,
               add_km_pvalue, add_km_annotation, label_curves
    forest     ForestRow, plot_forest
    grotta     plot_grotta, mrs_legend
    estimates  plot_point_range, dot_legend, plot_trajectory, add_difference_bracket,
               band_legend
    trajectory fit_trajectory, divergence_onset, plot_trajectory_scatter, tint
    groups     plot_groups, compare_groups, add_brackets, adjust_p, stars
    scatter    plot_correlation, plot_volcano, plot_roc, roc_curve, auc_score
    timecourse segment_positions, segmented_time_axis, plot_time_strip, plot_time_line,
               summarize, geo_summary, plot_violin_outline, plot_bar_points, bracket
    flow       plot_flow, flow_axes, add_gate
    flowchart  FlowNode, plot_flowchart (CONSORT-style boxes and arrows)
    assembly   Theme, Composite, Panel, Region, get_theme, WIDTHS
    panels     km_panel, roc_panel, groups_panel, paired_panel, volcano_panel,
               correlation_panel, trajectory_panel, forest_panel, grotta_panel,
               point_range_panel, timecourse_panel, violin_panel, bar_points_panel,
               flow_panel, legend_panel, custom_panel
"""
from .style import (COLORS, MRS_COLORS, InchLayout, add_y_break, auto_y_break, clean_axes,
                    log10_axis, sci_times_labels, use_nejm_style)
from .km import (KMArm, KMFit, add_at_risk_table, add_km_annotation, add_km_pvalue,
                 format_p, kaplan_meier, label_curves, logrank_test, plot_km)
from .forest import ForestRow, plot_forest
from .grotta import mrs_legend, plot_grotta
from .estimates import (add_difference_bracket, band_legend, dot_legend, plot_point_range,
                        plot_trajectory)
from .trajectory import (TrajectoryFit, divergence_onset, fit_trajectory,
                         plot_trajectory_scatter, rcs_basis, tint)
from .groups import add_brackets, adjust_p, compare_groups, plot_groups, sina_offsets, stars
from .scatter import auc_score, plot_correlation, plot_roc, plot_volcano, roc_curve
from .timecourse import (bracket, geo_summary, plot_bar_points, plot_time_line,
                         plot_time_strip, plot_violin_outline, segment_positions,
                         segmented_time_axis, summarize)
from .flow import add_gate, flow_axes, plot_flow
from .flowchart import FlowNode, plot_flowchart
from .assembly import WIDTHS, Composite, Panel, Region, Theme, get_theme
from .panels import *  # noqa: F401,F403  (figure makers)
