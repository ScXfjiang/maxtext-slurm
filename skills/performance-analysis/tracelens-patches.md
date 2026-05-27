# TraceLens Patches for TF 2.19+ / xprof Environments

When `TraceLens_generate_perf_report_jax` fails in environments with TensorFlow >= 2.19 and `tensorboard-plugin-profile` >= 2.17 (renamed to `xprof` internally), apply the patches below to the installed TraceLens package.

Find the install location:
```bash
python3 -c "import TraceLens; print(TraceLens.__path__[0])"
```

## Patch sequence

Apply these patches in order. There are 6 files, 14 patches total.

---

## File 1: `TraceLens/util.py`

### Problem
`tensorboard_plugin_profile.convert.raw_to_tool_data` no longer exists; the function moved to `xprof.convert._pywrap_profiler_plugin`. The tool name also changed from `trace_viewer@^` to `trace_viewer@`. Stale protobuf descriptors in `tensorboard_plugin_profile` raise `TypeError` (not just `ImportError`), so the except clause must catch all three: `ImportError`, `AttributeError`, and `TypeError`.

### Patch 1a: `load_data` method (~line 34)

```python
# OLD
        if filename_path.endswith("pb"):
            from tensorboard_plugin_profile.convert import raw_to_tool_data as convert

            data, _ = convert.xspace_to_tool_data([filename_path], "trace_viewer@^", {})

# NEW
        if filename_path.endswith("pb"):
            try:
                from tensorboard_plugin_profile.convert import raw_to_tool_data as convert
                data, _ = convert.xspace_to_tool_data([filename_path], "trace_viewer@^", {})
            except (ImportError, AttributeError, TypeError):
                from xprof.convert import _pywrap_profiler_plugin as _xprof
                data, _ = _xprof.xspace_to_tools_data([filename_path], "trace_viewer@", {})
```

### Patch 1b: `process_protobuf_file` method (~line 82)

```python
# OLD
    @staticmethod
    def process_protobuf_file(protobuf_file_name, module_name):
        from tensorboard_plugin_profile.convert import raw_to_tool_data as convert

        # look to see if the protobuf file has already been extracted
        dir_name = os.path.dirname(protobuf_file_name) + "/"
        hlo_filename = glob.glob(dir_name + os.path.sep + module_name + "*hlo_proto.pb")
        if len(hlo_filename) != 1:
            convert.xspace_to_tool_names([protobuf_file_name])

# NEW
    @staticmethod
    def process_protobuf_file(protobuf_file_name, module_name):
        try:
            from tensorboard_plugin_profile.convert import raw_to_tool_data as convert
        except (ImportError, AttributeError, TypeError):
            from xprof.convert import _pywrap_profiler_plugin as _xprof
            class convert:
                xspace_to_tool_data = staticmethod(lambda paths, tool, params: _xprof.xspace_to_tools_data(paths, tool, params))
                xspace_to_tool_names = staticmethod(lambda paths: _xprof.xspace_to_tools_data(paths, "tool_names", {}))

        # look to see if the protobuf file has already been extracted
        dir_name = os.path.dirname(protobuf_file_name) + "/"
        hlo_filename = glob.glob(dir_name + os.path.sep + module_name + "*hlo_proto.pb")
        if len(hlo_filename) != 1:
            convert.xspace_to_tool_names([protobuf_file_name])
```

---

## File 2: `TraceLens/TreePerf/tree_perf.py`

### Problem
`JaxTraceToTree.build_tree()` requires `metadata_events` as an argument but `from_file` wasn't extracting or passing it.

### Patch 2a: `from_file` static method

```python
# OLD
    @staticmethod
    def from_file(profile_filepath, *args, **kwargs) -> "JaxTreePerfAnalyzer":
        data = DataLoader.load_data(profile_filepath)
        data_pb = data["traceEvents"]
        categorizer = TraceEventUtils.prepare_event_categorizer(data_pb)
        events = TraceEventUtils.non_metadata_events(data_pb)
        linking_key = "correlation_id"
        tree = JaxTraceToTree(
            events, linking_key=linking_key, event_to_category=categorizer
        )
        return JaxTreePerfAnalyzer(
            tree,
            event_to_category=categorizer,
            pb_file_name=profile_filepath,
            *args,
            **kwargs,
        )

# NEW (adds metadata_events extraction and passing)
    @staticmethod
    def from_file(profile_filepath, *args, **kwargs) -> "JaxTreePerfAnalyzer":
        data = DataLoader.load_data(profile_filepath)
        data_pb = data["traceEvents"]
        categorizer = TraceEventUtils.prepare_event_categorizer(data_pb)
        metadata_events = TraceEventUtils.get_metadata(data_pb)
        events = TraceEventUtils.non_metadata_events(data_pb)
        linking_key = "correlation_id"
        tree = JaxTraceToTree(
            events, linking_key=linking_key, event_to_category=categorizer
        )
        return JaxTreePerfAnalyzer(
            tree,
            event_to_category=categorizer,
            pb_file_name=profile_filepath,
            metadata_events=metadata_events,
            *args,
            **kwargs,
        )
```

### Patch 2b: `__init__` method

```python
# OLD
    def __init__(
        self,
        tree: JaxTraceToTree,
        event_to_category: Callable[[dict], str] = TraceEventUtils.default_categorizer,
        pb_file_name=None,
        arch=None,
        python_path=None,
        kernel_metadata_keyword_filters: list[str] = None,
    ):
        # ...
        self.tree.build_tree(pb_file_name=pb_file_name)

# NEW (adds metadata_events parameter)
    def __init__(
        self,
        tree: JaxTraceToTree,
        event_to_category: Callable[[dict], str] = TraceEventUtils.default_categorizer,
        pb_file_name=None,
        metadata_events=None,
        arch=None,
        python_path=None,
        kernel_metadata_keyword_filters: list[str] = None,
    ):
        # ...
        if metadata_events is None:
            metadata_events = {}
        self.tree.build_tree(metadata_events=metadata_events, pb_file_name=pb_file_name)
```

---

## File 3: `TraceLens/TreePerf/gpu_event_analyser.py`

### Problem
`JaxGPUEventAnalyser` filters GPU PIDs with `pid < 100`, but `xprof` remaps GPU PIDs to the 1001-1008 range.

### Patch 3a: `__init__` gpu_pids filter

```python
# OLD
        self.gpu_pids = list(
            set([event["pid"] for event in events if event["pid"] < 100])
        )

# NEW
        self.gpu_pids = list(
            set([event["pid"] for event in events if event["pid"] < 100 or (1000 < event["pid"] < 1100)])
        )
```

### Patch 3b: `get_gpu_event_lists` pid filter

```python
# OLD
            if "ts" in event:
                if pid < 100:

# NEW
            if "ts" in event:
                if pid < 100 or (1000 < pid < 2000):
```

### Patch 3c: `get_breakdown_df_multigpu` filter

```python
# OLD
            filter(lambda x: x[0] < 100, dict_gpu_event_lists.items())

# NEW
            filter(lambda x: x[0] < 100 or (1000 < x[0] < 2000), dict_gpu_event_lists.items())
```

---

## File 4: `TraceLens/Reporting/generate_perf_report_jax.py`

### Problem
`get_df_kernel_launchers` may fail with `KeyError: 'gpu_kernel_op_cat'` when xprof doesn't provide kernel category metadata. Wrapping in try-except allows a partial report.

### Patch 4a: Kernel launcher section

```python
# OLD
    df_kernel_launchers = perf_analyzer.get_df_kernel_launchers(
        include_kernel_details=True
    )
    df_kernel_launchers_summary = perf_analyzer.get_df_kernel_launchers_summary(
        df_kernel_launchers
    )
    df_kernel_launchers_summary_by_category = (
        perf_analyzer.get_df_kernel_launchers_summary_by_category(df_kernel_launchers)
    )
    df_kernel_launchers_unique_args = perf_analyzer.get_df_kernel_launchers_unique_args(
        df_kernel_launchers, agg_metrics=agg_metrics, include_pct=True
    )

# NEW
    try:
        df_kernel_launchers = perf_analyzer.get_df_kernel_launchers(
            include_kernel_details=True
        )
        df_kernel_launchers_summary = perf_analyzer.get_df_kernel_launchers_summary(
            df_kernel_launchers
        )
        df_kernel_launchers_summary_by_category = (
            perf_analyzer.get_df_kernel_launchers_summary_by_category(df_kernel_launchers)
        )
        df_kernel_launchers_unique_args = perf_analyzer.get_df_kernel_launchers_unique_args(
            df_kernel_launchers, agg_metrics=agg_metrics, include_pct=True
        )
    except (KeyError, Exception) as e:
        logging.warning(f"Kernel launcher analysis failed ({e}), producing partial report")
        import pandas as pd
        df_kernel_launchers = pd.DataFrame()
        df_kernel_launchers_summary = pd.DataFrame()
        df_kernel_launchers_summary_by_category = pd.DataFrame()
        df_kernel_launchers_unique_args = pd.DataFrame()
```

### Patch 4b: XLA events section

```python
# OLD
    df_xla_events = perf_analyzer.get_df_kernel_launchers(
        include_kernel_details=True,
        gpu_kernel_op_cats=[
            "Uncategorized Events/XLA",
        ],
    )
    df_xla_perf = perf_analyzer.get_df_xla_perf(df_xla_events)
    df_xla_events_agg_name_col = df_xla_events.copy()
    df_xla_events_agg_name_col["name"] = df_xla_events.name.apply(
        lambda x: "".join([i for i in x if not i.isdigit()])
    )
    df_xla_summary = perf_analyzer.get_df_kernel_launchers_summary(
        df_xla_events_agg_name_col
    )

# NEW
    try:
        df_xla_events = perf_analyzer.get_df_kernel_launchers(
            include_kernel_details=True,
            gpu_kernel_op_cats=[
                "Uncategorized Events/XLA",
            ],
        )
        df_xla_perf = perf_analyzer.get_df_xla_perf(df_xla_events)
        df_xla_events_agg_name_col = df_xla_events.copy()
        df_xla_events_agg_name_col["name"] = df_xla_events.name.apply(
            lambda x: "".join([i for i in x if not i.isdigit()])
        )
        df_xla_summary = perf_analyzer.get_df_kernel_launchers_summary(
            df_xla_events_agg_name_col
        )
    except (KeyError, Exception) as e:
        logging.warning(f"XLA event analysis failed ({e}), skipping")
        import pandas as pd
        df_xla_perf = pd.DataFrame()
        df_xla_summary = pd.DataFrame()
```

---

## File 5: `TraceLens/Trace2Tree/trace_to_tree.py`

### Problem
`_categorize_gpu_kernel_ops` only processes events with `pid <= 100`, but multi-node xplane traces remap GPU device pids to 1001-1008, 2001-2008, etc. This causes `gpu_kernel_op_cat` to never be set on those events, leading to a KeyError in `get_kernel_launchers` and empty `kernel_launchers_summary_by_category.csv`.

### Fix (1 patch)
Remove the pid guard — `cat == "kernel"` already identifies GPU events. Also guard against `args` being None.

```python
# In _categorize_gpu_kernel_ops(), change:
        for event in self.events:
            if event.get("pid") <= 100:

                if event.get("cat") == "kernel":
                    name = event.get("name")
                    ...
                    if "hlo_op" in event.get("args").keys():
# To:
        for event in self.events:
            # Categorise every kernel event regardless of pid.
            # Multi-node xplane traces remap GPU device pids to 1001+.
            if event.get("cat") == "kernel":
                name = event.get("name", "")
                ...
                args = event.get("args") or {}
                if "hlo_op" in args:
```

## File 6a: `TraceLens/TreePerf/tree_perf.py` (XLA dtype map)

### Problem
`get_df_xla_perf` has a small `dtype_to_bytes` dict that crashes with `KeyError: 'c64'` (or other dtypes) when an HLO op operand uses a dtype not in the dict. Common offenders: `c64`/`c128` (complex), `s8`/`s64`, `f8*` (fp8 variants), `pred`, `token`.

### Fix (1 patch)
Expand the dict to cover all common XLA dtypes and use `.get(dtype, 0)` so unknown dtypes degrade gracefully instead of crashing.

```python
# OLD
        dtype_to_bytes = {
            "f32": 4,
            "bf16": 2,
            "s32": 4,
            "fp16": 2,
            "u32": 4,
            "f16": 2,
            "u64": 8,
        }
        # ...
                if shape and dtype:
                    total_input_bytes = (
                        total_input_bytes + np.prod(shape) * dtype_to_bytes[dtype]
                    )

# NEW
        dtype_to_bytes = {
            "pred": 1,
            "s4": 1, "u4": 1,
            "s8": 1, "u8": 1, "f8": 1, "f8e4m3fn": 1, "f8e5m2": 1, "f8e4m3fnuz": 1, "f8e5m2fnuz": 1,
            "s16": 2, "u16": 2, "f16": 2, "bf16": 2, "fp16": 2,
            "s32": 4, "u32": 4, "f32": 4,
            "s64": 8, "u64": 8, "f64": 8,
            "c64": 8, "c128": 16,
            "token": 0,
        }
        # ...
                if shape and dtype:
                    total_input_bytes = (
                        total_input_bytes + np.prod(shape) * dtype_to_bytes.get(dtype, 0)
                    )
```

---

## File 6b: `TraceLens/TreePerf/tree_perf.py` (launch latency)

### Problem
`get_GPU_kernel_launch_latency` accesses `self.tree.events_by_uid[event.get("parent")]` but kernel events from remapped multi-node pids may have `parent=None` (not linked to CPU launch events). This causes `KeyError: None`.

### Fix (1 patch)
Guard against missing parent UID and missing parent event.

```python
# In get_GPU_kernel_launch_latency(), change:
    def get_GPU_kernel_launch_latency(self, event: dict) -> float:
        GPU_kernel_launch_latency = event.get("ts") - self.tree.events_by_uid[
            event.get("parent")
        ].get("ts")
        return GPU_kernel_launch_latency
# To:
    def get_GPU_kernel_launch_latency(self, event: dict) -> float:
        parent_uid = event.get("parent")
        if parent_uid is None:
            return 0.0
        parent_event = self.tree.events_by_uid.get(parent_uid)
        if parent_event is None:
            return 0.0
        ts = event.get("ts")
        parent_ts = parent_event.get("ts")
        if ts is None or parent_ts is None:
            return 0.0
        return ts - parent_ts
```

---

## Dependency notes

If protobuf version errors occur, the known-working combination with TF 2.19.1:
```bash
pip install --force-reinstall --no-deps protobuf==5.29.5 tensorboard-plugin-profile==2.17.0
```

The patches above handle the remaining API differences via try-except fallbacks.
