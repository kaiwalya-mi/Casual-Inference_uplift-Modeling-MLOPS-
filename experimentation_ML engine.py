"""IncrementalEngage: a  honest uplift + streaming demo.

SETUP (Python 3.11/3.12 and Java 17, ideally in a fresh virtual environment):
    pip install causalml==0.17.0 pyspark==3.5.3 pandas numpy scikit-learn joblib
RUN:
    python incremental_engage.py
    python incremental_engage.py --sessions 3000 --seed 7 --skip-stream

Everything here is FAKE shopping data. No Kaggle data or real emails are used.
An email is randomly assigned AFTER the browsing window. A purchase is then
observed over a fixed, simulated follow-up period. Only earlier browsing goes
into the model. This keeps tomorrow's answer out of today's features.

This is an S-learner baseline, not a reproduction of the KDD papers below:
  KDD 2026, TEUM: https://github.com/ZimingWu020/TEUM
  KDD 2026, Uplift evaluation: https://github.com/Uplift-Bench/Uplift_Evaluation_Bench
  KDD 2025, UMLC: https://doi.org/10.1145/3690624.3709293

Real data needs a logged treatment, its random assignment chance, pre-email
features, and a later outcome. Click logs alone cannot prove an email worked.
S-learners do not remove hidden selection bias. They also cannot tell us the
two actual outcomes for the same person. CATE is an average for similar users.

Local MLOps demo: saved model, versions, code hash, metrics, and checkpoints.
Not a hosted service: scoring collects small batches on one machine. Before
real use, add consent checks, contact limits, monitoring, a global budget,
distributed scoring, durable deduplication, and a new randomized policy test.
The validation email cap does NOT guarantee the same share on future traffic.
"""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from causalml.inference.meta import BaseSClassifier
from causalml.metrics import qini_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

FEATURES = ["scroll_depth", "dwell_seconds", "idle_seconds", "velocity_decay"]
EVENT_SCHEMA = (
    "session_id long, event_time timestamp, scroll_depth double, "
    "dwell_seconds double, idle_seconds double, scroll_delta_px double, "
    "interval_seconds double, early boolean"
)


def make_events(n, seed, start_id=0, start="2026-01-01T00:00:00"):
    """Make six small browsing events per user, all before any email."""
    rng = np.random.default_rng(seed)
    start_time = datetime.fromisoformat(start)
    events = []
    for user in range(n):
        # Each row covers a different ten-second slice, so dwell can be summed.
        depth = rng.uniform(0.1, 1.0)
        dwell = rng.uniform(3, 55)
        idle = rng.uniform(0, 25)
        early_speed = rng.uniform(3, 10)
        late_speed = early_speed * rng.uniform(0.05, 1.0)
        minute = start_time + timedelta(minutes=user // 100)
        speeds = [early_speed] * 3 + [late_speed] * 3
        depths = depth * np.cumsum(speeds) / sum(speeds)
        for step in range(6):
            events.append((
                user + start_id, minute + timedelta(seconds=step * 10),
                float(depths[step]), float(dwell / 6), float(idle),
                float(speeds[step] * 10), 10.0, step < 3,
            ))
    return events


def features(events):
    """Use the SAME Spark feature recipe for saved data and new events."""
    from pyspark.sql import functions as F

    # Speed = pixels moved / seconds. Decay = early speed minus later speed.
    speed = F.col("scroll_delta_px") / F.col("interval_seconds")
    return (
        events.groupBy("session_id", F.window("event_time", "1 minute"))
        .agg(
            F.max("scroll_depth").alias("scroll_depth"),
            F.sum("dwell_seconds").alias("dwell_seconds"),
            F.max("idle_seconds").alias("idle_seconds"),
            F.avg(F.when(F.col("early"), speed)).alias("early_speed"),
            F.avg(F.when(~F.col("early"), speed)).alias("late_speed"),
        )
        .withColumn("velocity_decay", F.col("early_speed") - F.col("late_speed"))
        .select("session_id", "window", *FEATURES)
    )


def add_trial_labels(frame, seed):
    """Flip a fair coin for email vs no email, then simulate a later purchase."""
    rng = np.random.default_rng(seed)
    frame = frame.copy()
    engaged = frame.dwell_seconds / 60
    base = np.clip(0.04 + 0.35 * engaged + 0.2 * frame.scroll_depth, 0.01, 0.85)
    slowing = (frame.velocity_decay > 2) & (frame.idle_seconds > 8)
    # The email helps some shoppers, hurts some, and barely changes others.
    effect = np.where(slowing & (engaged > 0.3), 0.22,
                      np.where(frame.idle_seconds < 4, -0.10, 0.0))
    after_email = np.clip(base + effect, 0.01, 0.95)
    frame["treatment"] = rng.binomial(1, 0.5, len(frame))
    frame["outcome"] = rng.binomial(
        1, np.where(frame.treatment == 1, after_email, base)
    )
    # These answers exist only in a simulator. NEVER use them as model inputs.
    frame["true_effect"] = after_email - base
    return frame


def score(model, frame):
    """Ask the same model twice: with email and without email. Subtract."""
    x = frame[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Some features are missing or invalid; do not send an action.")
    # CausalML adds the treatment flag internally and fits ONE classifier.
    # Its prediction is P(buy | email, X) - P(buy | no email, X).
    return model.predict(x, verbose=False).reshape(-1)


def policy_gain(frame, send):
    """Estimate added purchases per user from a held-out randomized trial.

    The email chance is 0.5 in this demo. Weight each seen outcome by the
    inverse of that chance. This is IPW, not a before/after comparison.
    Do not reuse 0.5 for a real campaign with a different assignment scheme.
    """
    t = frame.treatment.to_numpy()
    y = frame.outcome.to_numpy()
    added = np.asarray(send) * (t * y / 0.5 - (1 - t) * y / 0.5)
    return added


def choose_cutoff(validation, uplift, max_fraction):
    """Pick a rule on validation data, not on the final exam (test data)."""
    cuts = np.unique(np.r_[0.0, np.quantile(uplift, np.linspace(0, 1, 41))])
    best_cut, best_gain = max(0.0, float(uplift.max())), 0.0  # None is allowed.
    for cut in cuts:
        send = uplift > cut
        if cut >= 0 and send.mean() <= max_fraction:
            gain = float(policy_gain(validation, send).mean())
            if gain > best_gain:
                best_cut, best_gain = float(cut), gain
    return best_cut


def evaluate(test, uplift, cutoff):
    """Keep the test set untouched until the model and rule are fixed."""
    send = uplift > cutoff
    added = policy_gain(test, send)
    gain = float(added.mean())
    error = float(1.96 * added.std(ddof=1) / np.sqrt(len(added)))
    control_rate = float(test.loc[test.treatment == 0, "outcome"].mean())
    all_gain = float(policy_gain(test, np.ones(len(test))).mean())
    table = test[["outcome", "treatment"]].copy()
    table["uplift"] = uplift
    # CausalML's unnormalized Qini score: area above random targeting.
    # This is NOT a normalized 0-to-1 coefficient. Always state the convention.
    qini = float(qini_score(table, outcome_col="outcome",
                           treatment_col="treatment", normalize=False)["uplift"])
    return {
        "data": "synthetic randomized trial; not real campaign results",
        "test_users": len(test),
        "qini_causalml_unnormalized": qini,
        "incremental_conversion_percentage_points_ipw": 100 * gain,
        "gain_95pct_normal_ci_pp": [100 * (gain - error), 100 * (gain + error)],
        "relative_lift_pct_ipw": 100 * gain / control_rate if control_rate else None,
        "relative_lift_note": "point estimate vs no email; no ratio CI calculated",
        "email_reduction_pct_vs_send_all": 100 * float(1 - send.mean()),
        "send_all_gain_pp_ipw": 100 * all_gain,
        "random_same_email_share_gain_pp_ipw": 100 * float(send.mean()) * all_gain,
        "synthetic_true_policy_gain_pp": 100 * float((send * test.true_effect).mean()),
        "cutoff": cutoff,
    }


def replay_stream(spark, model, cutoff, folder, seed):
    """Replay fake events through real Spark Structured Streaming. Dry run only.

    Spark closes each one-minute window after a 10-second event-time watermark.
    This waits for newer events, so it is near-real-time, NOT instant delivery.
    Two clock-only records let the last real windows close in this finite demo.
    A live source needs its own idle-partition and late-event handling.
    """
    inbox, actions = folder / "inbox", folder / "actions"
    inbox.mkdir()
    actions.mkdir()
    rows = make_events(300, seed, start_id=1_000_000, start="2026-02-01T00:00:00")
    for batch, offset in enumerate(range(0, len(rows), 600)):
        spark.createDataFrame(rows[offset:offset + 600], EVENT_SCHEMA).coalesce(1) \
            .write.mode("error").json(str(inbox / f"part-{batch}"))
    for k in range(2):
        clock = [(-1, datetime(2026, 2, 1, 1, k), 0., 0., 0., 0., 10., True)]
        spark.createDataFrame(clock, EVENT_SCHEMA).coalesce(1) \
            .write.mode("error").json(str(inbox / f"clock-{k}"))

    # Make file order explicit, even on disks with coarse timestamp precision.
    ordered_dirs = [inbox / f"part-{k}" for k in range(3)] + \
                   [inbox / f"clock-{k}" for k in range(2)]
    for k, directory in enumerate(ordered_dirs):
        for path in directory.glob("*.json"):
            os.utime(path, (1_800_000_000 + k, 1_800_000_000 + k))

    def route(batch, batch_id):
        began = time.perf_counter()
        pdf = batch.where("session_id >= 0").toPandas()
        if pdf.empty:
            return
        uplift = score(model, pdf)
        # The rule is frozen. Do not recompute a top fraction per micro-batch.
        # A stable key lets an external consumer deduplicate retries later.
        output = pd.DataFrame({
            "action_key": [f"{r.session_id}:{r.window.start.isoformat()}"
                           for r in pdf.itertuples()],
            "session_id": pdf.session_id,
            "cate": uplift,
            "action": np.where(uplift > cutoff, "WOULD_EMAIL", "SKIP"),
            "model_run": folder.name,
        })
        path = actions / f"batch-{batch_id}.jsonl"
        temp = path.with_suffix(".tmp")
        output.to_json(temp, orient="records", lines=True)
        os.replace(temp, path)  # A retried batch replaces the same local file.
        print(f"Batch {batch_id}: {len(pdf)} users, "
              f"{(uplift > cutoff).sum()} would email, "
              f"{time.perf_counter() - began:.2f}s including collection and writing")

    # File timestamps control replay order; later clock files come last.
    stream = spark.readStream.schema(EVENT_SCHEMA).option("recursiveFileLookup", "true") \
        .option("maxFilesPerTrigger", 1).json(str(inbox))
    query = features(stream.withWatermark("event_time", "10 seconds")) \
        .writeStream.outputMode("append").foreachBatch(route) \
        .option("checkpointLocation", str(folder / "checkpoint")) \
        .trigger(availableNow=True).start()
    query.awaitTermination()
    files = sorted(actions.glob("*.jsonl"))
    assert files, "No windows closed. Inspect event time and the watermark."
    decisions = pd.concat([pd.read_json(p, lines=True) for p in files])
    assert len(decisions) == 300, "Replay lost or repeated users. Inspect event order."
    assert decisions.action_key.is_unique, "Duplicate action keys found."
    # Reuse this checkpoint ONLY with this model, cutoff, query, and source.
    (folder / "stream_progress.json").write_text(json.dumps(query.recentProgress, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-email-fraction", type=float, default=0.75)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--skip-stream", action="store_true")
    args = parser.parse_args()
    if args.sessions < 1000 or not 0 < args.max_email_fraction <= 1:
        parser.error("Use at least 1000 sessions and an email fraction in (0, 1].")
    from pyspark.sql import SparkSession

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    spark = SparkSession.builder.master("local[2]").appName("IncrementalEngage") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.sql.session.timeZone", "UTC") \
        .config("spark.sql.shuffle.partitions", "2").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = args.out.resolve() / run
    folder.mkdir(parents=True, exist_ok=False)
    try:
        events = spark.createDataFrame(make_events(args.sessions, args.seed), EVENT_SCHEMA)
        frame = features(events).toPandas().sort_values("session_id").reset_index(drop=True)
        frame = add_trial_labels(frame, args.seed + 1)
        # Each user has one session. No user is shared between the three sets.
        train, rest = train_test_split(frame, test_size=0.4, random_state=args.seed,
                                      stratify=frame.treatment)
        valid, test = train_test_split(rest, test_size=0.5, random_state=args.seed,
                                      stratify=rest.treatment)
        assert set(train.session_id).isdisjoint(test.session_id)
        assert set(valid.session_id).isdisjoint(test.session_id)
        assert not set(FEATURES) & {"outcome", "treatment", "true_effect"}

        # THE CAUSALML PART: learn buy/no-buy from features plus the email flag.
        model = BaseSClassifier(learner=GradientBoostingClassifier(
            n_estimators=120, max_depth=3, min_samples_leaf=40,
            learning_rate=0.05, random_state=args.seed,
        ), control_name=0)
        model.fit(X=train[FEATURES].to_numpy(), treatment=train.treatment.to_numpy(),
                  y=train.outcome.to_numpy())
        cutoff = choose_cutoff(valid, score(model, valid), args.max_email_fraction)
        test_scores = score(model, test)
        report = evaluate(test, test_scores, cutoff)
        joblib.dump({"model": model, "cutoff": cutoff, "features": FEATURES}, folder / "model.joblib")
        loaded = joblib.load(folder / "model.joblib")  # Only load artifacts you trust.
        np.testing.assert_allclose(score(loaded["model"], test), test_scores)
        report["versions"] = {name: metadata.version(name) for name in
                              ["causalml", "pyspark", "numpy", "pandas", "scikit-learn"]}
        report["seed"] = args.seed
        report["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report["max_email_fraction_validation_only"] = args.max_email_fraction
        (folder / "metrics.json").write_text(json.dumps(report, indent=2))
        test.assign(cate=test_scores, would_email=test_scores > cutoff) \
            .drop(columns="window").to_csv(folder / "test_predictions.csv", index=False)
        print(json.dumps(report, indent=2))
        if not args.skip_stream:
            replay_stream(spark, loaded["model"], cutoff, folder, args.seed + 2)
        print(f"Saved run: {folder}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
