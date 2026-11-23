"""Write a copy of 06_heatmap_plot_locations.ipynb with the other metric (WIS <-> AE)."""
import json, sys
src, dst = sys.argv[1], sys.argv[2]
nb = json.load(open(src))
cell = nb["cells"][1]
s = "".join(cell["source"])
W = ('metric        = "wis_scaled_relative_skill"', 'metric_label  = "WIS"')
A = ('metric        = "ae_median_scaled_relative_skill"', 'metric_label  = "AE"')
frm, to = (W, A) if W[0] in s else (A, W)
for f, t in zip(frm, to):
    assert f in s, f
    s = s.replace(f, t)
cell["source"] = s.splitlines(keepends=True)
json.dump(nb, open(dst, "w"), indent=1)
print("metric:", to[1])
