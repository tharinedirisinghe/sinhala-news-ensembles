"""Streamlit dashboard for the ensemble comparison. Reads the CSVs in results/.

python -m streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

HERE = Path(__file__).parent
RESULTS = HERE / "results"
MAIN = "Text + Source"  # the feature set used everywhere except the input comparison

# purple for the winner, rust for mistakes, grey for the rest
INK, GREY, LIGHT, RULE = "#262626", "#8c8c8c", "#d4d4d4", "#ececec"
WINNER, PROBLEM = "#5b4bb7", "#c0532f"
FONT = '"Source Sans", "Source Sans 3", "Segoe UI", sans-serif'

SHORT_NAMES = {"Bagging (decision trees)": "Bagging", "Logistic Regression (baseline)": "Logistic Regression"}
SINGLE_MODELS = ["Bagging", "Random Forest", "XGBoost", "Logistic Regression"]

st.set_page_config(page_title="Sinhala news: ensemble comparison", layout="wide")


# helpers

@st.cache_data
def load():
    names = ["audit", "corpus_overview", "cv_metrics", "oof_predictions", "bootstrap_diff",
             "train_times", "robustness", "by_length", "extra_comparisons"]
    missing = [n for n in names if not (RESULTS / f"{n}.csv").exists()]
    if missing:
        return {}, missing
    data = {n: pd.read_csv(RESULTS / f"{n}.csv") for n in names}
    for table in data.values():
        if "model" in table:
            table["model"] = table["model"].replace(SHORT_NAMES)
    return data, []


def draw(fig, height, alt, left=10, right=20, bottom=30, top=6):
    fig.update_layout(height=height, margin=dict(l=left, r=right, t=top, b=bottom), showlegend=False,
                      font=dict(family=FONT, size=15, color=INK), plot_bgcolor="white", paper_bgcolor="white",
                      hoverlabel=dict(bgcolor="white", font_family=FONT))
    fig.update_xaxes(fixedrange=True, zeroline=False, automargin=True, tickfont_color="#666")
    fig.update_yaxes(fixedrange=True, zeroline=False, automargin=True, ticksuffix="  ")
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False}, alt=alt)


def chart_title(title, note):
    st.markdown(f"**{title}**  \n:gray[{note}]")


def spread_chart(groups):
    """One row per group: a band from the worst to the best model, and a dot for Stacking."""
    fig = go.Figure()
    for label, scores in groups.items():
        low, high = scores.min(), scores.max()
        fig.add_scatter(x=[low, high], y=[label, label], mode="lines+markers", line=dict(color=LIGHT, width=9),
                        marker=dict(color=LIGHT, size=9), hovertemplate="%{x:.3f}<extra></extra>")
        fig.add_scatter(x=[scores["Stacking"]], y=[label], mode="markers", marker=dict(color=WINNER, size=13),
                        hovertemplate="Stacking %{x:.3f}<extra></extra>")
        text = f"{low:.3f}" if round(low, 3) == round(high, 3) else f"{low:.3f} to {high:.3f}"
        fig.add_annotation(x=1, xref="paper", y=label, text=text, showarrow=False, xanchor="left", xshift=10)
    fig.update_yaxes(autorange="reversed")
    return fig


# load results

data, missing = load()
if missing:
    st.error(f"Missing from the results folder: {', '.join(missing)}. Copy the CSV files from Colab into results/, "
             "then run  python extra_checks.py")
    st.stop()

cv = data["cv_metrics"].query("features == @MAIN").groupby("model")
auc = cv["ROC-AUC"].mean().sort_values()
auc_sd = cv["ROC-AUC"].std()
accuracy = cv["Accuracy"].mean()
seconds = data["train_times"].query("features == @MAIN").groupby("model")["seconds"].mean()
gaps = data["bootstrap_diff"].query("features == @MAIN and metric == 'ROC-AUC'").set_index("question")
extra = data["extra_comparisons"]
stack_vs = extra[extra["question"].str.startswith("Stacking vs")].set_index("question")
audit = data["audit"].set_index("step")["count"]
articles = data["corpus_overview"]
n_articles, n_sites = len(articles), articles["domain"].nunique()
always_uncertain = articles["label"].mean()

st.markdown("# Which ensemble works best on Sinhala news?")
st.caption("Bagging, boosting and stacking compared on the LIRNEasia Sinhala news corpus (CC BY 4.0)")

overview, method, results, imperfect, robust = st.tabs(
    ["Overview", "How I tested", "Results", "Why not perfect?", "Robustness"])


# tab 1: overview

with overview:
    st.markdown("## The task is to spot news that needs fact-checking")
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        numbers = st.columns(3)
        numbers[0].metric("Sinhala articles", f"{n_articles:,}")
        numbers[1].metric("News websites", n_sites)
        numbers[2].metric("Labelled uncertain", f"{always_uncertain:.0%}")
        st.markdown(
            "People at LIRNEasia labelled each article *credible* or *uncertain*. "
            "Uncertain means they couldn't confirm it. It doesn't mean fake.")
        st.markdown(
            "I cleaned the text, then gave each model the writing style, 2,000 common words and word pairs, "
            "and the source: the website the article came from.")
        st.table(pd.DataFrame([
            ("Bagging", "300 trees, each trained on a random sample of articles"),
            ("Random Forest", "Bagging, plus a random set of features at each split"),
            ("XGBoost", "Boosting: each new tree corrects the trees before it"),
            ("Stacking", "A small model learns how to combine Random Forest, XGBoost and Logistic Regression"),
            ("Simple average", "The plain average of the same three models"),
            ("Logistic Regression", "One simple model, as the baseline to beat"),
        ], columns=["Model", "How it works"]), hide_index=True, border="horizontal", alt="The six models")
    with right:
        share = articles.groupby("domain")["label"].mean().sort_values()
        one_sided = share >= 0.9
        chart_title("Some websites are almost always labelled uncertain",
                    f"Share of each website's articles. {one_sided.sum()} websites are at 90% or more.")
        fig = go.Figure(go.Bar(
            x=share, y=share.index, orientation="h", marker_color=[PROBLEM if x else LIGHT for x in one_sided],
            text=[f"{v:.0%}" for v in share], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:.0%}<extra></extra>"))
        fig.update_xaxes(range=[0, 1.1], visible=False)
        fig.update_yaxes(tickfont_size=12, dtick=1)
        fig.update_traces(textfont_size=13)
        fig.update_layout(bargap=0.25)
        draw(fig, 485, "Share of articles labelled uncertain, by website", bottom=4)

    with st.expander("Model settings and why"):
        st.table(pd.DataFrame([
            ("Bagging", "300 full-depth trees, all features",
             "Same tree count as RF, so the difference is the random feature choice"),
            ("Random Forest", "300 trees, about 45 features per split",
             "Square root of the feature count is the usual rule"),
            ("XGBoost", "300 trees, depth 6, learning rate 0.1, 80% of rows and columns per tree",
             "Default depth, a common rate and tree count, sampling to limit overfitting"),
            ("Stacking", "RF, XGBoost and LR into Logistic Regression, 3 inner folds",
             "Different kinds of model, with a simple combiner"),
            ("Simple average", "Mean of RF, XGBoost and LR",
             "Shows whether learning the weights is worth it"),
            ("Logistic Regression", "scikit-learn defaults", "An untuned baseline"),
        ], columns=["Model", "Settings", "Reason"]), hide_index=True, border="horizontal", alt="Model settings")
        st.caption("All use random seed 42. The base models give extra weight to the smaller class (credible). "
                   "The stacking combiner does not.")
    with st.expander("What was cleaned"):
        st.table(data["audit"].rename(columns={"step": "Step", "count": "Count", "why": "Reason"}),
                 hide_index=True, border="horizontal", alt="Data cleaning steps")
    if (HERE / "architecture.png").exists():
        with st.expander("System diagram"):
            st.image(str(HERE / "architecture.png"), width="stretch", alt="System architecture")


# tab 2: how I tested

with method:
    st.markdown("## Every model is tested on articles it hasn't seen")
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        part_size = round(n_articles / 5, -1)
        chart_title("How the articles were split: 5-fold cross-validation",
                    f"Five parts of about {part_size:,.0f} articles. Each round trains on four and tests on one.")
        z = [[1 if part == rnd else 0 for part in range(5)] for rnd in range(5)]
        words = [["Test" if part == rnd else "Train" for part in range(5)] for rnd in range(5)]
        fig = go.Figure(go.Heatmap(
            z=z, x=[f"Part {p}" for p in range(1, 6)], y=[f"Round {r}" for r in range(1, 6)], text=words,
            texttemplate="%{text}", textfont_size=14, colorscale=[[0, "#e8e8e8"], [1, "#595959"]],
            showscale=False, xgap=4, ygap=4, hoverinfo="skip"))
        fig.update_xaxes(side="top", showgrid=False, ticks="")
        fig.update_yaxes(autorange="reversed", showgrid=False, ticks="")
        draw(fig, 245, "Five rounds of training and testing", top=30, bottom=4)
        st.markdown(
            f"- Every article is tested once, by a model that never saw it.\n"
            f"- Each part keeps the same mix: {always_uncertain:.0%} uncertain, {1 - always_uncertain:.0%} credible.\n"
            f"- Copies of the same story on different websites stay in one part, "
            f"so a model is never tested on a story it has already read.\n"
            f"- All six models use the same five parts, so they face exactly the same test.")
    with right:
        chart_title("What I tested", "Each test answers one question. The last column shows where to find it.")
        st.table(pd.DataFrame([
            ("Six models on the same five parts", "Which ensemble ranks the articles best?", "Results"),
            ("Pairs of models, 1,000 resamples", "Is a gap real, or could it be luck?", "Results"),
            ("Three inputs: text, source, both", "Does adding the source to the text help?", "In the talk"),
            ("Results by article length", "Where do the models struggle?", "Why not perfect?"),
            ("Unseen websites, later news", "Would it hold up in real use?", "Robustness"),
        ], columns=["What I did", "Question", "Tab"]), hide_index=True, border="horizontal", alt="List of tests")
        st.markdown("### How I scored them")
        st.markdown(
            "Mainly **ROC-AUC**: the chance that a model gives a random uncertain article a higher score "
            "than a random credible one. 0.5 is guessing, 1 is perfect. "
            "Unlike accuracy, it isn't fooled by the label mix. I also report accuracy and training time.")


# tab 3: results

with results:
    st.markdown("## Stacking scores highest, but not by much")
    best_single = auc[SINGLE_MODELS].idxmax()
    st.markdown(
        f"It is clearly ahead of bagging, Random Forest and XGBoost, and just ahead of {best_single}. "
        "Random Forest and XGBoost are tied.")

    left, right = st.columns(2, gap="large")
    with left:
        chart_title("How well each model ranks the articles",
                    "ROC-AUC: 0.5 is guessing, 1 is perfect. Lines show ± one standard deviation.")
        colours = [WINNER if m == "Stacking" else GREY for m in auc.index]
        fig = go.Figure()
        for model, colour in zip(auc.index, colours):
            fig.add_scatter(x=[auc[model] - auc_sd[model], auc[model] + auc_sd[model]], y=[model, model],
                            mode="lines", line=dict(color=LIGHT, width=3), hoverinfo="skip")
            fig.add_scatter(x=[auc[model]], y=[model], mode="markers", marker=dict(color=colour, size=14),
                            hovertemplate="%{y}: %{x:.3f}<extra></extra>")
            fig.add_annotation(x=1.01, xref="paper", y=model, text=f"{auc[model]:.3f}", showarrow=False,
                               xanchor="left", font=dict(color=colour if model == "Stacking" else INK, size=16))
        fig.update_xaxes(range=[0.79, 0.85], dtick=0.02, tickformat=".2f", gridcolor=RULE)
        draw(fig, 290, "Average ROC-AUC of the six models", right=60)
    with right:
        chart_title("Head to head", "Gap in ROC-AUC between two models. Tie means the gap could be luck.")
        pairs = [
            ("Bagging vs boosting", "Random Forest", "XGBoost", gaps.loc["Bagging vs boosting"]),
            ("Do random features help?", "Random Forest", "Bagging", gaps.loc["What random features add"]),
            ("Stacking vs bagging", "Stacking", "Random Forest", stack_vs.loc["Stacking vs Random Forest"]),
            ("Stacking vs boosting", "Stacking", "XGBoost", stack_vs.loc["Stacking vs XGBoost"]),
            ("Stacking vs best single model", "Stacking", best_single, gaps.loc["Stacking vs best single model"]),
            ("Is learning the weights worth it?", "Stacking", "Simple average",
             gaps.loc["Learned vs simple combination"]),
        ]
        rows = []
        for question, a, b, gap in pairs:
            real = gap["real difference?"] == "yes"
            winner = a if gap["difference"] > 0 else b
            verdict = f":violet[**{winner} wins**]" if real else ":gray[Tie]"
            rows.append((question, f"{a} vs {b}", f"{gap['difference']:+.3f}", verdict))
        st.table(pd.DataFrame(rows, columns=["Question", "Pair", "Gap", "Verdict"]), hide_index=True,
                 border="horizontal", alt="Head-to-head comparisons of pairs of models")
        st.caption("How I decided: I re-drew the test articles 1,000 times, keeping copies of a story together, "
                   "and measured the gap each time. If the middle 95% of those gaps were all above zero, "
                   "the win is real. If not, the winner could flip, so it's a tie.")

    order = auc.index[::-1]
    chart_title("Accuracy and cost",
                f"Always answering uncertain would already be right {always_uncertain:.0%} of the time. "
                "Stacking uses a different cut-off, so compare the models on ROC-AUC.")
    st.table(pd.DataFrame(
        [[f"{accuracy[m]:.0%}" for m in order], [f"{seconds[m]:.0f} s" for m in order]],
        columns=list(order), index=["Accuracy", "Training time per fold"]),
        border="horizontal", alt="Accuracy and training time")


# tab 4: why not perfect

with imperfect:
    st.markdown("## About one article in four is still misjudged")
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        stack = data["oof_predictions"].query("features == @MAIN and model == 'Stacking'")
        lengths = data["by_length"].query("features == @MAIN").copy()
        lengths["band"] = lengths["length"].str.split().str[0]
        flagged = stack["score"] >= 0.5
        credible = stack["label"] == 0
        n_credible, n_uncertain = int(credible.sum()), int((~credible).sum())
        false_alarms, caught = int((credible & flagged).sum()), int((~credible & flagged).sum())
        missed, left_alone = n_uncertain - caught, n_credible - false_alarms

        chart_title("Confusion matrix for Stacking",
                    f"Most mistakes are false alarms: {false_alarms / n_credible:.0%} of credible articles get flagged.")
        cells = [[(caught, "caught", n_uncertain), (missed, "missed", n_uncertain)],
                 [(false_alarms, "false alarms", n_credible), (left_alone, "rightly left alone", n_credible)]]
        fig = go.Figure(go.Heatmap(
            z=[[0, 1], [1, 0]], x=["Flagged as uncertain", "Not flagged"],
            y=[f"Uncertain<br>({n_uncertain:,})", f"Credible<br>({n_credible:,})"],
            text=[[f"<b>{n:,}</b> {w}<br>{n / total:.0%} of the row" for n, w, total in row] for row in cells],
            texttemplate="%{text}", textfont_size=15, colorscale=[[0, "#ececec"], [1, PROBLEM]],
            showscale=False, xgap=5, ygap=5, hoverinfo="skip"))
        fig.update_xaxes(side="top", showgrid=False, ticks="", title_text="Stacking's answer", title_font_size=13)
        fig.update_yaxes(autorange="reversed", showgrid=False, ticks="", title_text="Real label", title_font_size=13)
        draw(fig, 225, "Confusion matrix for Stacking", top=30, bottom=4)

        chart_title("Short articles are the hardest to rank",
                    "Worst to best ROC-AUC of the six. The purple dot is Stacking.")
        groups = {}
        for band in ["Short", "Medium", "Long"]:
            part = lengths[lengths["band"] == band].set_index("model")
            words = part["length"].iloc[0].split("(")[1].rstrip(")")
            groups[f"{band}, {words}"] = part["ROC-AUC"]
        fig = spread_chart(groups)
        fig.update_xaxes(range=[0.73, 0.87], dtick=0.05, tickformat=".2f", gridcolor=RULE)
        draw(fig, 190, "ROC-AUC range by article length", right=125)
    with right:
        mixed = int(audit.get("Kept: near-copy groups with different labels", 0))
        st.markdown("### The labels are opinions")
        st.markdown(f"{mixed} stories appear as near-identical copies with different labels. "
                    "A model is unlikely to get both versions right.")
        st.markdown("### Counting words isn't checking facts")
        st.markdown("The models see how an article is written and where it was published. "
                    "They never check whether a claim is true.")
        st.markdown("### Short articles give few clues")
        st.markdown("With fewer than about 90 words there is little to learn from, "
                    "and every model scores lowest there.")
        st.markdown("### So, what is it good for?")
        st.markdown("Choosing what a fact-checker reads first. It shouldn't decide alone.")


# tab 5: robustness

with robust:
    st.markdown("## Scores fall when the test gets closer to real use")
    tests = {
        "Main: 5-fold, near-copies grouped": "Normal 5-fold test",
        "Unseen websites": "Websites not seen in training",
        "Temporal: train ≤ Sep 2020, test Oct–Nov": "Later news, Oct and Nov 2020",
    }
    by_test = data["robustness"].query("features == @MAIN").groupby(["setup", "model"])["ROC-AUC"].mean()
    chart_title("ROC-AUC of the same six models in three tests",
                "The normal test is the 5-fold split from the How I tested tab. Further left is worse.")
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.045,
                        subplot_titles=list(tests.values()))
    for col, setup in enumerate(tests, start=1):
        s = by_test[setup].reindex(auc.index)
        colours = [WINNER if m == "Stacking" else GREY for m in s.index]
        fig.add_scatter(x=s, y=s.index, mode="markers+text", marker=dict(color=colours, size=13),
                        text=[f"{v:.3f}" for v in s], textposition="middle right", cliponaxis=False,
                        hovertemplate="%{y}: %{x:.3f}<extra></extra>", row=1, col=col)
        fig.update_xaxes(range=[0.64, 0.89], dtick=0.1, tick0=0.7, tickformat=".1f", gridcolor=RULE, row=1, col=col)
        fig.update_yaxes(showgrid=True, gridcolor=RULE, row=1, col=col)
    draw(fig, 300, "ROC-AUC of six models in three tests", top=34)

    train = articles[articles["date"] < "2020-10-01"]
    later = articles[articles["date"] >= "2020-10-01"]
    new_site = ~later["domain"].isin(train["domain"])
    later_scores = by_test["Temporal: train ≤ Sep 2020, test Oct–Nov"]
    same_size = by_test[next(s for s in by_test.index.levels[0] if s.startswith("Random split"))]

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### Why later news is so hard")
        st.markdown(
            f"{new_site.mean():.0%} of the later articles come from websites that weren't in the training months, "
            f"and the credible share rises from {1 - train['label'].mean():.0%} to {1 - later['label'].mean():.0%}. "
            "Topics probably changed too, but this test can't separate those causes. "
            f"Less training data isn't the main one: a random split of that size still scores {same_size.mean():.2f}.")
    with right:
        st.markdown("### What I take from this")
        st.markdown(
            f"Stacking is still on top for later news ({later_scores['Stacking']:.3f}), "
            f"and Logistic Regression falls the most (to {later_scores['Logistic Regression']:.3f}). "
            "**On this data, the choice of ensemble matters less than the features, the labels "
            "and where the test articles come from.**")
