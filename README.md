# gcb-host

C++ pilot driver, Python MMIO register simulator, model manifest tooling

Part of the GreenCoyote-Bay photonic inference fabric (ODIN_Bridge).

**Corrected 2026-09-12 (`TKT-032`, this line never updated when the
finding landed): `proxy/gcb_proxy.py` contains zero FastAPI code** -
confirmed by direct grep, not assumption. It is a self-checking,
pure-Python MMIO register simulator exercising the real v0.6.0 register
map, not an HTTP proxy of any kind.

