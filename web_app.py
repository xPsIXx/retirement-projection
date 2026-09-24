"""FastAPI web app wrapping the FIRE simulator with Plotly charts."""
from __future__ import annotations

import io
import json
import sys
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import plotly.utils
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from firemodel.schema import Config, Accounts, find_in_range
from firemodel.sim import DAYS, expenses_by_category, simulate
from firemodel.solve import Solution, solve, sweep, what_if, x_kind

HERE = Path(__file__).resolve().parent

app = FastAPI(title="FIRE Simulator")

# Templates
templates = Jinja2Templates(directory=str(HERE / "templates"))
templates.env.globals["enumerate"] = enumerate
templates.env.globals["zip"] = zip
templates.env.globals["len"] = len

app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

# Job store for async simulation runs
JOBS: dict = {}

# ── Load configs and plans ──────────────────────────────────────────────

def load_config(plan: str | None = None, no_accounts: bool = False) -> Config:
    import runpy
    mod = runpy.run_path(str(HERE / "config.py"))
    if plan is None and not no_accounts:
        return mod["CONFIG"]
    from firemodel.loader import load_plan
    return mod["make_config"](
        load_plan(plan) if plan else mod["PLAN"],
        accounts=mod["NO_ACCOUNTS"] if no_accounts else None,
    )


def get_available() -> dict:
    from firemodel.loader import available
    return available()


def money(v: float) -> str:
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1e6:
        return f"{sign}${v / 1e6:,.2f}M"
    if v >= 1e3:
        return f"{sign}${v / 1e3:,.0f}k"
    return f"{sign}${v:,.0f}"


def yw(years: float) -> str:
    days = round(years * DAYS)
    y, d = divmod(days, DAYS)
    return f"{y}y {d // 7}w" if d // 7 else f"{y}y"


def describe_solution(sol: Solution) -> str:
    r = sol.best
    if r is None:
        return "No solution within cap"
    if sol.kind == "years":
        return f"Work {yw(r.x)} ({r.x:.2f} yrs) at terminal salary → stop at age {r.x_end_age:.1f}"
    return f"Needs salary of {money(r.x)}/yr in the X segment"


# ── Plotly charts ──────────────────────────────────────────────────────

def plot_trajectories(sols: list[Solution]) -> str:
    fig = go.Figure()
    colors = ["#00bcd4", "#ff9800", "#e91e63", "#4caf50"]
    for i, s in enumerate(sols):
        if s.best is None:
            continue
        r = s.best
        ages = [x.age for x in r.rows]
        balances = [x.balance / 1e6 for x in r.rows]
        fig.add_trace(go.Scatter(
            x=ages, y=balances,
            mode="lines", name=s.scenario.name,
            line=dict(color=colors[i % len(colors)]),
        ))
        if r.x_end_age is not None:
            fig.add_vline(x=r.x_end_age, line_dash="dash",
                          line_color=colors[i % len(colors)])
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.update_layout(
        title="Net worth by age at minimum X (real $, millions)",
        xaxis_title="Age", yaxis_title="$M",
        height=400, template="plotly_white",
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def plot_expenses_chart(cfg: Config, retire: float | None = None) -> str:
    cats = expenses_by_category(cfg, retire)
    order = sorted(cats, key=lambda c: -sum(cats[c]))
    if len(order) > 8:
        rest = order[7:]
        cats = {c: cats[c] for c in order[:7]} | {
            "other": [sum(v) for v in zip(*(cats[c] for c in rest))]
        }
        order = order[:7] + ["other"]
    ages = list(range(cfg.start_age, cfg.end_age))
    fig = go.Figure()
    colors = ["#2196f3", "#00bcd4", "#4caf50", "#ff9800", "#e91e63",
              "#9c27b0", "#607d8b", "#795548"]
    for i, cat in enumerate(order):
        fig.add_trace(go.Bar(
            x=ages, y=[v / 1e3 for v in cats[cat]],
            name=cat, marker_color=colors[i % len(colors)],
        ))
    fig.update_layout(
        title=f"Expenses per year by category ($k, today's dollars)",
        xaxis_title="Age", yaxis_title="$k",
        barmode="stack", height=400, template="plotly_white",
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def plot_cashflow_chart(r) -> str:
    ages = [x.age for x in r.rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ages, y=[x.gross / 1e3 for x in r.rows],
                              mode="lines", name="Gross income",
                              line=dict(color="#00bcd4")))
    fig.add_trace(go.Scatter(x=ages, y=[x.net_income / 1e3 for x in r.rows],
                              mode="lines", name="Take-home",
                              line=dict(color="#2196f3")))
    fig.add_trace(go.Scatter(x=ages, y=[x.expenses / 1e3 for x in r.rows],
                              mode="lines", name="Expenses",
                              line=dict(color="#ff9800")))
    fig.add_trace(go.Scatter(x=ages, y=[x.sold / 1e3 for x in r.rows],
                              mode="lines", name="Portfolio sold",
                              line=dict(color="#e91e63")))
    fig.update_layout(
        title=f"Cash flow: {r.scenario} (X={r.x:.2f}), $k/yr",
        xaxis_title="Age", yaxis_title="$k",
        height=400, template="plotly_white",
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def plot_accounts_chart(r) -> str:
    ages = [x.age for x in r.rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ages,
                              y=[(x.balance - x.k401 - x.roth - x.c529) / 1e6 for x in r.rows],
                              mode="lines", name="Taxable",
                              line=dict(color="#00bcd4")))
    fig.add_trace(go.Scatter(x=ages, y=[x.k401 / 1e6 for x in r.rows],
                              mode="lines", name="401(k) (pre-tax)",
                              line=dict(color="#ff9800")))
    fig.add_trace(go.Scatter(x=ages, y=[x.roth / 1e6 for x in r.rows],
                              mode="lines", name="Roth",
                              line=dict(color="#4caf50")))
    if any(x.c529 for x in r.rows):
        fig.add_trace(go.Scatter(x=ages, y=[x.c529 / 1e6 for x in r.rows],
                                  mode="lines", name="529",
                                  line=dict(color="#e91e63")))
    fig.update_layout(
        title=f"Accounts: {r.scenario} (X={r.x:.2f}), $M at end of year",
        xaxis_title="Age", yaxis_title="$M",
        height=400, template="plotly_white",
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


# ── Routes ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    av = get_available()
    plans = [(n, pkg) for n, pkg in av.get("plans", [])]
    grids = [(n, pkg) for n, pkg in av.get("grids", [])]
    return templates.TemplateResponse(request, "index.html", {
        "plans": plans,
        "grids": grids,
        "whatif_amounts": "10000, 25000, 50000",
        "whatif_ages": "25, 30, 35, 40, 45",
    })


def run_job(job_id: str, plan: str, no_accounts: bool, whatif_amounts: str, whatif_ages: str, table: bool):
    """Heavy solve in a worker thread; stores rendered HTML under job_id."""
    t0 = time.time()
    try:
        cfg = load_config(plan, no_accounts)
        scenarios = cfg.scenarios

        w_amounts = [float(x.strip().replace(",", "").replace("_", ""))
                     for x in whatif_amounts.split(",") if x.strip()]
        w_ages = [float(x.strip()) for x in whatif_ages.split(",") if x.strip()]

        sols = [solve(cfg, s) for s in scenarios]
        without = ([solve(replace(cfg, accounts=Accounts()), s) for s in scenarios]
                   if cfg.accounts.active else [])

        whatif_results = {}
        if w_amounts and w_ages:
            from firemodel.solve import what_if as wi
            for si, s in enumerate(sols):
                if not s.best:
                    continue
                for a in w_amounts:
                    for g in w_ages:
                        v = wi(cfg, s, a, g)
                        whatif_results.setdefault(si, {})[f"{a}@{g}"] = v

        first_sol = next((s for s in sols if s.best), None)
        charts = {}
        if first_sol:
            charts["networth"] = plot_trajectories(sols)
            charts["expenses"] = plot_expenses_chart(cfg, first_sol.best.x_end_age)
            charts["cashflow"] = plot_cashflow_chart(first_sol.best)
            if cfg.accounts.active:
                charts["accounts"] = plot_accounts_chart(first_sol.best)

        html = templates.env.get_template("results.html").render(
            plan=plan,
            sols=sols,
            without=without,
            money=money,
            yw=yw,
            describe=describe_solution,
            cfg=cfg,
            charts=charts,
            whatif_results=whatif_results,
            w_amounts=w_amounts,
            w_ages=w_ages,
            show_table=table,
        )
        JOBS[job_id] = {"status": "done", "html": html,
                        "elapsed": round(time.time() - t0, 1)}
    except Exception as e:
        import traceback
        traceback.print_exc()
        JOBS[job_id] = {"status": "error",
                        "html": f"<h2>Simulation failed</h2><p>{e}</p>",
                        "elapsed": round(time.time() - t0, 1)}


@app.post("/run")
def run_submit(
    plan: str = Form("sf_family"),
    no_accounts: bool = Form(False),
    whatif_amounts: str = Form(""),
    whatif_ages: str = Form(""),
    table: bool = Form(False),
):
    """Kick off a solve in a worker thread; returns a job id immediately."""
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "running"}
    threading.Thread(
        target=run_job,
        args=(job_id, plan, no_accounts, whatif_amounts, whatif_ages, table),
        daemon=True,
    ).start()
    return JSONResponse({"job": job_id})


@app.get("/status/{job_id}")
def run_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"status": "unknown"})
    if job["status"] == "done":
        # detach from cache so browsers can't 304 a result page
        resp = HTMLResponse(job["html"])
        resp.headers["Cache-Control"] = "no-store"
        return resp
    return JSONResponse({"status": job["status"]})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8177)
