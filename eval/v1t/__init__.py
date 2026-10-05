"""DanZero-V1T (DanLM's MLP reproduction of DanZero) without DanLM's binaries.

``encoder`` rebuilds the agent's 964-float input rows, ``player`` plays it on
our engine with only numpy and onnxruntime, ``verify`` checks it decision by
decision against the compiled agent (macOS with DanLM only), and ``arena``
runs duplicate evaluations with our engine as the only referee. The model is
DanLM's published int8 ONNX file (Apache 2.0 with a non-commercial clause; not
vendored, path from ``V1T_ONNX`` or the DanLM checkout).
"""
