from CP1_starter_code import IrisLoaderUCI, Processor, ensure_dir, main
import CP1_starter_code as cp
import os
import json
import tempfile
import pandas as pd
import matplotlib
matplotlib.use("Agg")


SPECIES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
passed = 0


def check(name, cond, detail=""):
    global passed
    assert cond, f"FAILED: {name} {detail}"
    passed += 1
    print(f"ok  {name}")


def read_json(p):
    with open(p) as f:
        return json.load(f)


os.chdir(tempfile.mkdtemp())
ensure_dir("./outputs")

# ---------- IrisLoaderUCI ----------
loader = IrisLoaderUCI()
df = loader.load()
check("load shape", df.shape == (150, 5), df.shape)
check("load columns", list(df.columns) == cp.REQUIRED_COLS)
check("load no NaN rows", df.isna().all(axis=1).sum() == 0)
check("load sets self.df", loader.df is df)

bal = loader.check_class_balance()
check("balance total", bal["total"] == 150 and type(bal["total"]) is int)
check("balance counts", bal["counts"] == {s: 50 for s in SPECIES})
check("balance count types", all(
    type(v) is int for v in bal["counts"].values()))
check("balance proportions", all(v == 0.333333 and type(v) is float
                                 for v in bal["proportions"].values()))

loader.save_class_balance("./outputs/class_balance.json")
check("save_class_balance round trip", read_json(
    "./outputs/class_balance.json") == bal)

loader.save_head(10, "./outputs/head_a.csv", 96)
loader.save_head(10, "./outputs/head_b.csv", 96)
loader.save_head(10, "./outputs/head_c.csv", 7)
a, b, c = (pd.read_csv(f"./outputs/head_{x}.csv") for x in "abc")
check("save_head shape", a.shape == (10, 5), a.shape)
check("save_head no index column", list(a.columns) == cp.REQUIRED_COLS)
check("save_head deterministic", a.equals(b))
check("save_head seed changes order", not a.equals(c))

# ---------- Processor ----------
proc = Processor(df)
check("Processor copies df", proc.df is not df)

proc.add_numeric_label("./outputs/label_map.json")
expected_map = {s: i for i, s in enumerate(SPECIES)}
check("label_map json", read_json("./outputs/label_map.json") == expected_map)
check("label column values", proc.df["label"].tolist().count(0) == 50
      and set(proc.df["label"]) == {0, 1, 2})
check("label matches species",
      (proc.df["species"].map(expected_map) == proc.df["label"]).all())
check("original df untouched", "label" not in df.columns)

stats = proc.stats()
check("stats keys", set(stats) == set(cp.NUM_COLS))
for col in cp.NUM_COLS:
    s = stats[col]
    check(f"stats {col} keys", set(s) == {
          "min", "max", "mean", "median", "std"})
    check(f"stats {col} min/max/mean/median",
          abs(s["min"] - df[col].min()
              ) < 1e-9 and abs(s["max"] - df[col].max()) < 1e-9
          and abs(s["mean"] - df[col].mean()) < 1e-9
          and abs(s["median"] - df[col].median()) < 1e-9)
    # pandas std is ddof=1, numpy std is ddof=0; accept either
    check(f"stats {col} std", min(abs(s["std"] - df[col].std(ddof=1)),
                                  abs(s["std"] - df[col].std(ddof=0))) < 1e-9)
json.dumps(stats)  # raises TypeError if numpy types leaked in
check("stats JSON serializable", True)

info = proc.train_val_split(val_ratio=0.2, seed=96)
tr, va = pd.read_csv("./outputs/train.csv"), pd.read_csv("./outputs/val.csv")
check("split return", info == {"train_size": 120, "val_size": 30}, info)
check("split file sizes", len(tr) == 120 and len(va) == 30)
check("split no index column", "Unnamed: 0" not in tr.columns)
check("split covers all rows",
      len(pd.concat([tr, va]).drop_duplicates()) == len(df.drop_duplicates()))
proc.train_val_split(val_ratio=0.2, seed=96)
check("split deterministic", pd.read_csv("./outputs/val.csv").equals(va))

# clamp edge cases
tiny = Processor(df.head(10))
check("clamp low", tiny.train_val_split(
    val_ratio=0.001, seed=0)["val_size"] == 1)
r = tiny.train_val_split(val_ratio=0.999, seed=0)
check("clamp high", r["val_size"] == 9 and r["train_size"] == 1, r)
proc.train_val_split(val_ratio=0.2, seed=96)  # restore normal files

proc.plot_hist("petal_length", "./outputs/hist.png")
proc.plot_label_bar("./outputs/bar.png")
proc.plot_scatter("petal_length", "petal_width", out="./outputs/scatter.png")
for f in ("hist", "bar", "scatter"):
    p = f"./outputs/{f}.png"
    check(f"{f} png exists", os.path.exists(p) and os.path.getsize(p) > 1000)
check("figures closed", len(matplotlib.pyplot.get_fignums()) == 0)

proc.save_processed("./outputs/processed.csv")
pr = pd.read_csv("./outputs/processed.csv")
check("processed shape", pr.shape == (150, 6), pr.shape)
check("processed has label", "label" in pr.columns)

# ---------- main end to end ----------
try:
    main(initials="TS", student_id="12345")
    check("main rejects bad id", False)
except ValueError:
    check("main rejects bad id", True)

main(initials="TS", student_id="968892796", seed=96)
expected = ["class_balance.json", "head.csv", "label_map.json", "train.csv",
            "val.csv", "stats.json", "hist_petal_length.png", "label_bar.png",
            "scatter_petal.png", "processed.csv"]
check("main creates all outputs",
      all(os.path.exists(f"./outputs/{f}") for f in expected),
      [f for f in expected if not os.path.exists(f"./outputs/{f}")])

print(f"\nAll {passed} checks passed")
