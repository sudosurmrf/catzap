"""NanoDet-Plus integration for the edge-model PoC.

Houses the upstream NanoDet repo's training wrappers, configs, and
canonical checkpoint paths. Kept in its own package because NanoDet
upstream pins ``torch<2.0`` (incompatible with the ultralytics version
the rest of the eval harness uses) — the actual training/inference run
expects the isolated venv created by ``setup_venv.sh``.

The eval-harness adapter at ``training/edge/eval/adapters/nanodet_adapter.py``
is intentionally main-env-importable (lazy upstream imports) so it can
run inference against either:

  * a ``.pth`` / ``.ckpt`` checkpoint via the ``nanodet`` pip module
    (requires the isolated venv), or
  * a ``.onnx`` checkpoint via ``onnxruntime`` (works in the main env).

US-007 records the upstream-pretrained baseline through the harness; the
fine-tune lives in US-008.
"""
