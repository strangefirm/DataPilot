# Loaded automatically in sandbox subprocesses (via PYTHONPATH). It does not change any
# numbers; it only records chart metadata on savefig so guardrails can verify that
# every chart has a title, axis labels and units.
import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")

try:
    import matplotlib.figure as _mfig

    _orig_savefig = _mfig.Figure.savefig

    def _savefig(self, fname, *args, **kwargs):
        result = _orig_savefig(self, fname, *args, **kwargs)
        try:
            axes = []
            for ax in self.get_axes():
                if not ax.get_visible() or (not ax.has_data() and not ax.get_title()):
                    continue  # skip colorbars / empty axes
                axes.append({
                    "title": ax.get_title() or ax.get_title(loc="left") or ax.get_title(loc="right"),
                    "xlabel": ax.get_xlabel(),
                    "ylabel": ax.get_ylabel(),
                    "is_pie": bool(ax.patches) and all(type(p).__name__ == "Wedge" for p in ax.patches),
                })
            sup = self._suptitle.get_text() if getattr(self, "_suptitle", None) else ""
            with open("_charts_meta.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"file": os.path.basename(str(fname)), "suptitle": sup, "axes": axes}) + "\n")
        except Exception:
            pass
        return result

    _mfig.Figure.savefig = _savefig
except Exception:
    pass
