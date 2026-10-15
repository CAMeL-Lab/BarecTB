#!/usr/bin/env python3
"""Run the cached six-model predictor with CAMeL Parser multiroot decoding.

The underlying predictor is intentionally kept unchanged so the completed
single-root run remains reproducible.  This wrapper applies the same decode
override used by experiments/camelparser_cameltb before each checkpoint is
used.
"""

from __future__ import annotations

from types import MethodType

import predict_all_models_cached as predictor


_single_root_load = predictor.Parser.load


def load_with_multiroot(*args, **kwargs):
    parser = _single_root_load(*args, **kwargs)
    # predictor.main() adds --camel-parser-root to sys.path before loading a
    # checkpoint, so this is deliberately imported lazily here.
    from src.dependency_parser.biaff_parser import _decode_multiroot

    parser.model.decode = MethodType(_decode_multiroot, parser.model)
    print("MULTIROOT DECODER ENABLED", flush=True)
    return parser


# Parser.load is a classmethod.  A staticmethod wrapper prevents Python from
# injecting the Parser class into the already-bound original loader.
predictor.Parser.load = staticmethod(load_with_multiroot)


if __name__ == "__main__":
    predictor.main()
