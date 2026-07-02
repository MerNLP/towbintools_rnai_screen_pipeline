from shiny import ui
from shinywidgets import output_widget


app_ui = ui.page_sidebar(
    ui.sidebar(
        # ---- Data ----------------------------------------------------------
        ui.h5("Data"),
        ui.div(
            ui.input_text("exp_dir", "Experiment directory", value="",
                          placeholder="/mnt/towbin.data/shared/.../", width="100%"),
            ui.input_action_button("browse_exp_btn", "Browse",
                                   class_="btn btn-sm btn-outline-secondary mt-1 w-100"),
        ),
        ui.input_action_button("scan_btn", "Scan for CSVs",
                               class_="btn btn-sm btn-outline-secondary w-100 mt-1"),
        ui.output_ui("csv_picker_ui"),
        ui.input_numeric("image_channel", "Image channel to display (0-indexed)",
                         value=1, min=0, max=10, step=1, width="100%"),
        ui.input_select(
            "overlay_mode",
            "Overlay",
            choices={
                "none": "None",
                "fill": "Segmentation mask (fill)",
                "outline": "Segmentation mask (outline)",
            },
            selected="none",
            width="100%",
        ),
        ui.input_slider("overlay_alpha", "Overlay opacity", min=0.05, max=0.85,
                        value=0.35, step=0.05),
        ui.input_action_button("load_btn", "Load CSV", class_="btn-primary w-100 mt-1"),

        ui.hr(),

        # ---- Normalization -------------------------------------------------
        ui.h5("Normalization"),
        ui.output_ui("norm_ui"),

        ui.hr(),

        # ---- Plot type ----------------------------------------------------
        ui.h5("Plot type"),
        ui.input_radio_buttons(
            "plot_type", None,
            choices={
                "volcano":       "Volcano (fold change)",
                "strain_linear": "Strain compare | linear",
                "strain_log":    "Strain compare | log scale",
                "log2fc_scatter":"log₂FC scatter (strain pair)",
            },
            selected="volcano",
        ),

        ui.hr(),

        # ---- Volcano settings ---------------------------------------------
        ui.panel_conditional(
            "input.plot_type === 'volcano'",
            ui.h5("Volcano settings"),
            ui.input_radio_buttons(
                "volcano_mode", "Strains to include",
                choices={"combined": "All strains (combined)", "per_strain": "Per strain"},
                selected="combined",
            ),
            ui.panel_conditional(
                "input.volcano_mode === 'per_strain'",
                ui.input_select("volcano_strain", "Strain", choices=[], width="100%"),
            ),
            ui.input_slider("pval_threshold", "Max p-value (significance)",
                            0.001, 0.1, 0.05, step=0.001),
            ui.hr(),
        ),

        # ---- Strain comparison (linear / log / log₂FC scatter) ------------
        ui.panel_conditional(
            "input.plot_type === 'strain_linear' || "
            "input.plot_type === 'strain_log' || "
            "input.plot_type === 'log2fc_scatter'",
            ui.h5("Strain comparison"),
            ui.input_select("strain_a", "Reference strain (X axis)", choices=[], width="100%"),
            ui.input_select("strain_b", "Comparison strain (Y axis)", choices=[], width="100%"),
            ui.input_select("cmp_day", "Day to display", choices=[], width="100%"),
            ui.input_slider(
                "log2fc_hit_thresh",
                "log₂(Y/X) hit threshold",
                0.1,
                10.0,
                0.5,
                step=0.1,
            ),
            ui.input_select(
                "hit_direction",
                "Hit direction (log₂FC scatter)",
                choices={
                    "symmetric": "Either side (|Y − X| ≥ threshold)",
                    "neg": "Lower in comparison strain Y (Y − X ≤ −threshold)",
                    "pos": "Higher in comparison strain Y (Y − X ≥ threshold)",
                },
                selected="symmetric",
                width="100%",
            ),
            ui.p(
                "Linear / log: |log₂(Y/X)| on the metric (default 0.5). "
                "log₂FC scatter: directional or |Y − X| Δlog₂FC vs EV (default 2.0). "
                "Use gene-level metric columns (*_gene or *_qc_gene)",
                style="font-size:0.75em; color:#888;",
            ),
            ui.hr(),
        ),

        # ---- Hit calling (volcano only) -----------------------------------
        ui.panel_conditional(
            "input.plot_type === 'volcano'",
            ui.h5("Hit calling"),
            ui.p(
                "Hits: |log₂FC vs EV| ≥ threshold and p-value ≤ max p-value "
                "(vertical dashed lines on the volcano).",
                style="font-size:0.85em; color:#666;",
            ),
            ui.input_slider(
                "volcano_log2fc_thresh",
                "|log₂FC| threshold (volcano)",
                0.0,
                5.0,
                1.0,
                step=0.1,
            ),
            ui.hr(),
        ),

        # ---- Day filter (volcano only) ------------------------------------
        ui.panel_conditional(
            "input.plot_type === 'volcano'",
            ui.h5("Days"),
            ui.output_ui("day_filter_ui"),
            ui.hr(),
        ),

        # ---- Library filter (all plot types) -------------------------------
        ui.h5("Library"),
        ui.output_ui("library_filter_ui"),
        ui.hr(),

        # ---- Visibility filters ------------------------------------------
        ui.h5("Show"),
        ui.input_checkbox("show_controls", "Controls", value=True),
        ui.input_checkbox("show_hits", "Hits", value=True),
        ui.input_checkbox("show_nonhits", "Non-hits", value=True),

        ui.hr(),

        # ---- Preload images -----------------------------------------------
        ui.h5("Preload images"),
        ui.input_action_button(
            "preload_hits_controls_btn",
            "Preload hits & controls only",
            class_="btn btn-sm btn-outline-secondary w-100 mb-1",
        ),
        ui.input_action_button(
            "preload_all_plot_btn",
            "Preload all plot images",
            class_="btn btn-sm btn-outline-secondary w-100 mb-1",
        ),
        ui.p(
            "Uses current overlay setting: None = raw only; fill/outline also "
            "prebuilds mask overlays (QC colors when object CSV exists).",
            style="font-size:0.72em; color:#888; margin:0 0 4px 0;",
        ),
        ui.p(
            "First: only points labeled hit or control on the current plot. "
            "Second: every visible point (hits, non-hits, controls).",
            style="font-size:0.72em; color:#888; margin:0 0 6px 0;",
        ),
        ui.output_ui("cache_status_ui"),

        # ---- Export -------------------------------------------------------
        ui.h5("Export"),
        ui.download_button("dl_plot_html", "Download plot (HTML)",
                           class_="btn btn-sm btn-outline-secondary w-100 mb-1"),
        ui.download_button("dl_hits_csv", "Download hits (CSV)",
                           class_="btn btn-sm btn-outline-secondary w-100 mb-1"),
        ui.download_button("dl_flagged_csv", "Download flagged CSV",
                           class_="btn btn-sm btn-outline-secondary w-100"),

        width=320,
    ),

    # ---- Main content -------------------------------------------------------
    ui.layout_columns(
        ui.card(
            ui.card_header("Plot  |  hover for details, click a point to view image"),
            output_widget("main_plot"),
            full_screen=True,
            style="resize: both; overflow: auto; min-height: 400px; min-width: 300px;",
        ),
        ui.card(
            ui.card_header(
                ui.div(
                    "Selected gene",
                    ui.div(
                        ui.output_ui("image_dl_picker"),
                        ui.input_action_button(
                            "close_image_btn", "✕ Close",
                            class_="btn btn-sm btn-outline-secondary",
                        ),
                        style="float:right; margin-top:-2px; display:flex; gap:4px; align-items:center;",
                    ),
                    style="width:100%;",
                )
            ),
            ui.output_ui("selected_info"),
            style="resize: both; overflow: auto; min-height: 400px; min-width: 250px;",
        ),
        col_widths=[7, 5],
    ),

    title="RNAi Screen Dashboard",
    fillable=True,
)
