"""Small, semantics-preserving ONNX rewrites for embedded runtimes."""
from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path


_ASYMMETRIC_PAD_PATCH_VERSION = 2


def rewrite_asymmetric_conv_padding(model_path: str | Path,
                                    cache_dir: str | Path = "models/onnx_compat") -> Path:
    """Return a cached graph whose asymmetric Conv is a symmetric Conv followed by Slice.

    cuDNN 9.6's frontend can fail to create an engine for otherwise valid FP16 convolutions with
    different leading/trailing padding.  Padding both sides equally and slicing off the additional
    border is mathematically identical, while cuDNN only sees the supported symmetric convolution.
    The source graph is never modified.
    """
    source = Path(model_path).resolve()
    stat = source.stat()
    fingerprint = hashlib.sha256(
        f"{source}\0{stat.st_size}\0{stat.st_mtime_ns}\0{_ASYMMETRIC_PAD_PATCH_VERSION}".encode()
    ).hexdigest()[:12]
    target_dir = Path(cache_dir)
    target = target_dir / f"{source.stem}-{fingerprint}-pad-v{_ASYMMETRIC_PAD_PATCH_VERSION}.onnx"
    if target.exists():
        return target

    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError as exc:
        raise RuntimeError(
            "the Jetson cuDNN compatibility rewrite needs onnx==1.17.0; "
            "run `python -m pip install -r requirements-jetson.txt`"
        ) from exc

    model = onnx.load(str(source), load_external_data=True)
    default_opset = next((item.version for item in model.opset_import if item.domain in ("", "ai.onnx")), 0)
    if default_opset < 10:
        raise RuntimeError(f"cannot rewrite {source.name}: ONNX Slice inputs require opset 10+, got {default_opset}")

    rewritten_nodes = []
    rewritten = 0
    for index, node in enumerate(model.graph.node):
        pads_attr = next((attr for attr in node.attribute if attr.name == "pads"), None)
        pads = list(pads_attr.ints) if pads_attr is not None else []
        spatial_rank = len(pads) // 2
        asymmetric = (spatial_rank > 0 and len(pads) == 2 * spatial_rank
                      and pads[:spatial_rank] != pads[spatial_rank:])
        if node.op_type == "Conv" and asymmetric:
            prefix = f"__instanthmr_asymmetric_pad_{index}"
            starts_name, ends_name, axes_name = (f"{prefix}_starts", f"{prefix}_ends", f"{prefix}_axes")
            original_output, conv_output = node.output[0], f"{prefix}_conv_output"
            leading, trailing = pads[:spatial_rank], pads[spatial_rank:]
            symmetric = [max(a, b) for a, b in zip(leading, trailing)]
            starts = [s - a for s, a in zip(symmetric, leading)]
            ends = [-(s - b) if s != b else 2 ** 63 - 1 for s, b in zip(symmetric, trailing)]
            axes = list(range(2, 2 + spatial_rank))
            for name, values in ((starts_name, starts), (ends_name, ends), (axes_name, axes)):
                model.graph.initializer.append(
                    helper.make_tensor(name, TensorProto.INT64, [len(values)], values)
                )
            node.output[0] = conv_output
            del pads_attr.ints[:]
            pads_attr.ints.extend([*symmetric, *symmetric])
            rewritten_nodes.append(node)
            rewritten_nodes.append(helper.make_node(
                "Slice", [conv_output, starts_name, ends_name, axes_name], [original_output],
                name=f"{prefix}_slice"
            ))
            rewritten += 1
        else:
            rewritten_nodes.append(node)

    if rewritten == 0:
        return source

    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    marker = next((prop for prop in model.metadata_props
                   if prop.key == "instanthmr_asymmetric_padding_rewrite"), None)
    if marker is None:
        marker = model.metadata_props.add()
        marker.key = "instanthmr_asymmetric_padding_rewrite"
    marker.value = f"v{_ASYMMETRIC_PAD_PATCH_VERSION}; convs={rewritten}"
    onnx.checker.check_model(model)

    target_dir.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        onnx.save_model(model, str(temporary))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Jetson ONNX compatibility: rewrote {rewritten} asymmetric Conv as symmetric Conv + Slice -> {target}")
    return target
