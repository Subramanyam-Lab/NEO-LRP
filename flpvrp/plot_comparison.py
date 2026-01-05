from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

DIR_FLP = Path("/storage/group/azs7266/default/wzk5140/MLforVRP/Codes/vpr_model/NEOS-LRP-Codes_WK/flp/results/solutions_flpvrp")
DIR_NEO = Path("/storage/group/azs7266/default/wzk5140/MLforVRP/Codes/vpr_model/NEOS-LRP-Codes_WK/neos/prodhon/prodhon_solutions_for_plots_cost_over_fi")
RUN_INDEX = 1
INSTANCES = [
    "coord20-5-1",
    "coord20-5-1b",
    "coord50-5-3",
    "coord50-5-2bBIS",
]

@dataclass
class PlotStyle:
    fig_width: float = 3.8
    fig_height: float = 3.8
    tight_layout: bool = True
    pad_inches: float = 0.02

    color: str = "black"
    bg_color: Optional[str] = None

    customer_marker: str = "o"
    customer_markersize: float = 11
    customer_edgewidth: float = 1.2
    customer_facecolor: str = "white"
    customer_facealpha: float = 1.0
    customer_zorder: float = 5.0

    depot_box_mode: str = "frac"
    depot_box_frac: float = 0.045
    depot_box_size: float = 2.0

    depot_outer_edgewidth: float = 1.6
    depot_outer_linestyle_open: str | Tuple[int, Tuple[int, ...]] = "-"
    depot_outer_linestyle_closed: str | Tuple[int, Tuple[int, ...]] = "-"

    fill_depots: bool = True
    depot_facecolor: str = "white"
    depot_facealpha: float = 1.0

    show_closed_hatch: bool = True
    hatch_pattern: str = "///////"
    hatch_color: str = "0.6"
    hatch_linewidth: float = 0.8
    hatch_alpha: float = 1.0

    show_closed_inset: bool = False
    closed_inset_frac: float = 0.65
    closed_inset_edgewidth: float = 1.2
    closed_inset_linestyle: Tuple[int, Tuple[int,int]] = (0, (3, 3))

    show_depot_ids: bool = False
    depot_id_fontsize: float = 9.5
    depot_id_fontweight: str = "regular"
    depot_id_va: str = "center"
    depot_id_ha: str = "center"
    depot_id_dx: float = 0.0
    depot_id_dy: float = 0.0
    depot_id_bbox: Optional[dict] = None

    assignment_linewidth: float = 0.6
    route_linewidth: float = 1.0

    x_label: Optional[str] = None
    y_label: Optional[str] = None
    label_fontsize: float = 10
    tick_labelsize: float = 9
    tick_length: float = 3.5
    tick_width: float = 0.8
    tick_direction: str = "in"
    spine_width: float = 0.8

    show_grid: bool = False
    grid_linewidth: float = 0.3
    grid_linestyle: Tuple[int, Tuple[int,int]] = (0, (1, 3))

    xlim: Optional[Tuple[float,float]] = None
    ylim: Optional[Tuple[float,float]] = None
    margin_frac: float = 0.04

    legend: bool = False
    legend_fontsize: float = 9
    legend_loc: str = "lower right"
    legend_frame: bool = False

    equal_aspect: bool = True

STYLE = PlotStyle()

def _style_ax(ax, style: PlotStyle):
    ax.tick_params(
        axis="both", which="both",
        direction=style.tick_direction,
        length=style.tick_length,
        width=style.tick_width,
        labelsize=style.tick_labelsize,
        color=style.color
    )
    for sp in ax.spines.values():
        sp.set_linewidth(style.spine_width)
        sp.set_color(style.color)
    if style.bg_color is not None:
        ax.set_facecolor(style.bg_color)
    if style.show_grid:
        ax.grid(True, linewidth=style.grid_linewidth,
                linestyle=style.grid_linestyle, color=style.color, alpha=0.3)

def _auto_limits(ax, payload: Dict, style: PlotStyle):
    dep = np.asarray(payload["data"]["dep_coords"], dtype=float)
    cus = np.asarray(payload["data"]["cus_coords"], dtype=float)
    if dep.size == 0 and cus.size == 0:
        return
    pts = dep if cus.size == 0 else (np.vstack([dep, cus]) if dep.size else cus)
    xmin, ymin = np.min(pts, axis=0)
    xmax, ymax = np.max(pts, axis=0)
    if style.xlim is None:
        dx = xmax - xmin
        pad_x = style.margin_frac * (dx if dx > 0 else 1.0)
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
    else:
        ax.set_xlim(*style.xlim)
    if style.ylim is None:
        dy = ymax - ymin
        pad_y = style.margin_frac * (dy if dy > 0 else 1.0)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
    else:
        ax.set_ylim(*style.ylim)

def _scatter_customers(ax, customers: np.ndarray, style: PlotStyle):
    if customers.size == 0:
        return
    ax.scatter(
        customers[:, 0], customers[:, 1],
        s=style.customer_markersize**2 / 4.0,
        marker=style.customer_marker,
        facecolors=style.customer_facecolor,
        edgecolors=style.color,
        linewidths=style.customer_edgewidth,
        alpha=style.customer_facealpha,
        zorder=style.customer_zorder
    )

def _draw_customers(ax, customers: List[Tuple[float, float]], style: PlotStyle):
    C = np.asarray(customers, dtype=float)
    _scatter_customers(ax, C, style)

def _effective_box_side(ax, style: PlotStyle) -> float:
    if style.depot_box_mode == "frac":
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xspan = abs(xmax - xmin)
        yspan = abs(ymax - ymin)
        return style.depot_box_frac * max(1e-9, min(xspan, yspan))
    else:
        return max(1e-9, style.depot_box_size)

def _draw_depots(ax, depots: List[Tuple[float, float]], opened_indices: List[int], style: PlotStyle):
    D = np.asarray(depots, dtype=float)
    if D.size == 0:
        return
    opened = set(int(i) for i in (opened_indices or []))

    side = _effective_box_side(ax, style)
    half = side / 2.0
    inset_side = style.closed_inset_frac * side
    inset_half = inset_side / 2.0

    for i, (x, y) in enumerate(D):
        is_open = i in opened

        rect_fill = Rectangle(
            (x - half, y - half), side, side,
            fill=style.fill_depots,
            facecolor=(style.depot_facecolor if style.fill_depots else "none"),
            alpha=(style.depot_facealpha if style.fill_depots else 1.0),
            edgecolor="none",
            zorder=6
        )
        ax.add_patch(rect_fill)

        if (not is_open) and style.show_closed_hatch:
            with mpl.rc_context({
                "hatch.color": style.hatch_color,
                "hatch.linewidth": style.hatch_linewidth
            }):
                rect_hatch = Rectangle(
                    (x - half, y - half), side, side,
                    fill=False,
                    hatch=style.hatch_pattern,
                    edgecolor=style.hatch_color,
                    linewidth=0.0,
                    alpha=style.hatch_alpha,
                    zorder=7
                )
                ax.add_patch(rect_hatch)

        if (not is_open) and style.show_closed_inset:
            rect_in = Rectangle(
                (x - inset_half, y - inset_half), inset_side, inset_side,
                fill=False,
                linewidth=style.closed_inset_edgewidth,
                edgecolor=style.color,
                linestyle=style.closed_inset_linestyle,
                zorder=7.5
            )
            ax.add_patch(rect_in)

        rect_border = Rectangle(
            (x - half, y - half), side, side,
            fill=False,
            linewidth=style.depot_outer_edgewidth,
            edgecolor=style.color,
            linestyle=(style.depot_outer_linestyle_open if is_open else style.depot_outer_linestyle_closed),
            zorder=8
        )
        ax.add_patch(rect_border)

        if style.show_depot_ids:
            ax.text(
                x + style.depot_id_dx, y + style.depot_id_dy, f"{i+1}",
                ha=style.depot_id_ha, va=style.depot_id_va,
                fontsize=style.depot_id_fontsize, weight=style.depot_id_fontweight,
                color=style.color, bbox=style.depot_id_bbox, zorder=9
            )

def _draw_assignments(ax, customers: List[Tuple[float, float]], depots: List[Tuple[float, float]],
                      assign_depot_per_customer: List[int], style: PlotStyle):
    if not customers or not depots or not assign_depot_per_customer:
        return
    C = np.asarray(customers, dtype=float)
    D = np.asarray(depots, dtype=float)
    for i, d in enumerate(assign_depot_per_customer):
        if d is None or d < 0 or d >= len(D):
            continue
        x0, y0 = C[i]
        x1, y1 = D[d]
        ax.plot([x0, x1], [y0, y1], color=style.color, linewidth=style.assignment_linewidth, linestyle=(0, (3, 3)), zorder=1)

def _draw_routes_from_vroom(ax, base_payload: Dict, vroom_payload: Optional[Dict], style: PlotStyle):
    if not vroom_payload:
        return
    dep = np.asarray(base_payload["data"]["dep_coords"], dtype=float)
    cus = np.asarray(base_payload["data"]["cus_coords"], dtype=float)
    per_depot = vroom_payload.get("per_depot", {})

    for dkey, info in per_depot.items():
        d_idx = int(dkey)
        depot_xy = dep[d_idx]
        for rt in info.get("routes", []):
            ids = rt.get("Ids", [])
            pts = []
            for tok in ids:
                tok_str = str(tok).strip()
                if tok_str.startswith("Depot") or tok_str.startswith("D"):
                    pts.append(depot_xy)
                else:
                    try:
                        pts.append(cus[int(tok_str)])
                    except (ValueError, IndexError):
                        continue
            if len(pts) >= 2:
                xy = np.asarray(pts, dtype=float)
                ax.plot(xy[:, 0], xy[:, 1],
                        color=style.color,
                        linewidth=style.route_linewidth,
                        zorder=0)

def plot_panel(
    flp_payload: Dict,
    neo_payload: Dict,
    vroom_flp: Optional[Dict] = None,
    vroom_neo: Optional[Dict] = None,
    style: Optional[PlotStyle] = None,
    title_flp_loc: str = "FLP–VRP (location-allocations)",
    title_neo_loc: str = "NEO–DS (location-allocations)",
    title_flp_rt: str  = "FLP–VRP (routes)",
    title_neo_rt: str  = "NEO–DS (routes)",
    super_title: Optional[str] = None,
    save_path: Optional[str] = None
):
    style = style or STYLE
    fig, axs = plt.subplots(2, 2, figsize=(2 * style.fig_width, 2 * style.fig_height))

    ax = axs[0, 0]; _style_ax(ax, style)
    _draw_assignments(ax, flp_payload["data"]["cus_coords"], flp_payload["data"]["dep_coords"],
                      flp_payload["sets"].get("assign_depot_per_customer", []), style)
    _auto_limits(ax, flp_payload, style)
    _draw_customers(ax, flp_payload["data"]["cus_coords"], style)
    _draw_depots(ax, flp_payload["data"]["dep_coords"], flp_payload["sets"]["opened_indices"], style)
    if style.equal_aspect: ax.set_aspect("equal", adjustable="box")
    ax.set_title(title_flp_loc, fontsize=style.label_fontsize, color=style.color)

    ax = axs[0, 1]; _style_ax(ax, style)
    _draw_assignments(ax, neo_payload["data"]["cus_coords"], neo_payload["data"]["dep_coords"],
                      neo_payload["sets"].get("assign_depot_per_customer", []), style)
    _auto_limits(ax, neo_payload, style)
    _draw_customers(ax, neo_payload["data"]["cus_coords"], style)
    _draw_depots(ax, neo_payload["data"]["dep_coords"], neo_payload["sets"]["opened_indices"], style)
    if style.equal_aspect: ax.set_aspect("equal", adjustable="box")
    ax.set_title(title_neo_loc, fontsize=style.label_fontsize, color=style.color)

    ax = axs[1, 0]; _style_ax(ax, style)
    _auto_limits(ax, flp_payload, style)
    _draw_routes_from_vroom(ax, flp_payload, vroom_flp, style)
    _draw_customers(ax, flp_payload["data"]["cus_coords"], style)
    _draw_depots(ax, flp_payload["data"]["dep_coords"], flp_payload["sets"]["opened_indices"], style)
    if style.equal_aspect: ax.set_aspect("equal", adjustable="box")
    ax.set_title(title_flp_rt, fontsize=style.label_fontsize, color=style.color)

    ax = axs[1, 1]; _style_ax(ax, style)
    _auto_limits(ax, neo_payload, style)
    _draw_routes_from_vroom(ax, neo_payload, vroom_neo, style)
    _draw_customers(ax, neo_payload["data"]["cus_coords"], style)
    _draw_depots(ax, neo_payload["data"]["dep_coords"], neo_payload["sets"]["opened_indices"], style)
    if style.equal_aspect: ax.set_aspect("equal", adjustable="box")
    ax.set_title(title_neo_rt, fontsize=style.label_fontsize, color=style.color)

    if style.x_label:
        axs[1, 0].set_xlabel(style.x_label, fontsize=style.label_fontsize, color=style.color)
        axs[1, 1].set_xlabel(style.x_label, fontsize=style.label_fontsize, color=style.color)
    if style.y_label:
        axs[0, 0].set_ylabel(style.y_label, fontsize=style.label_fontsize, color=style.color)
        axs[1, 0].set_ylabel(style.y_label, fontsize=style.label_fontsize, color=style.color)

    if super_title:
        fig.suptitle(super_title, y=0.98)

    if style.tight_layout:
        plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            format="pdf",
            bbox_inches="tight",
            pad_inches=style.pad_inches
        )
        plt.close(fig)
    else:
        return fig, axs

def read_json(p: Path) -> Optional[Dict]:
    if not p or not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def find_file(base: Path, stem: str, method_dir: str, run_index: int) -> Optional[Path]:
    candidates = [
        base / stem / method_dir / f"run_{run_index}.json",
        base / stem / f"run_{run_index}.json"
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


if __name__ == "__main__":
    for stem in INSTANCES:
        flp_payload_path = find_file(DIR_FLP, stem, "FLPVRP", RUN_INDEX)
        vroom_flp_path   = find_file(DIR_FLP, stem, "VROOM", RUN_INDEX)

        neo_payload_path = find_file(DIR_NEO, stem, "NEO_LRP", RUN_INDEX)
        vroom_neo_path   = find_file(DIR_NEO, stem, "VROOM_EVAL", RUN_INDEX)

        flp_payload = read_json(flp_payload_path) if flp_payload_path else None
        neo_payload = read_json(neo_payload_path) if neo_payload_path else None
        vroom_flp   = read_json(vroom_flp_path) if vroom_flp_path else None
        vroom_neo   = read_json(vroom_neo_path) if vroom_neo_path else None

        if not (flp_payload and neo_payload):
            print(f"[skip] missing core JSON(s) for {stem}")
            continue

        out_pdf = f"{stem}_FLPvsNEO_2x2.pdf"
        plot_panel(
            flp_payload=flp_payload,
            neo_payload=neo_payload,
            vroom_flp=vroom_flp,
            vroom_neo=vroom_neo,
            style=STYLE,
            title_flp_loc="FLP–VRP (location-allocations)",
            title_neo_loc="NEO–DS (location-allocations)",
            title_flp_rt="FLP–VRP (routes)",
            title_neo_rt="NEO–DS (routes)",
            super_title=None,
            save_path=out_pdf
        )
        print(f"[saved] {out_pdf}")
