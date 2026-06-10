"""apcir.interactive — iKAT'26 interactive search.

Two roles, deliberately split:
  * a long-running SEARCH SERVER (`search_server.py`) that loads the dense index into
    RAM + builds GPU faiss + opens BM25 lucene ONCE at startup and answers `POST /search`;
  * a thin Sim.API DRIVER (`driver.py`) that drives the official user-simulation API
    (start/continue loop) and, per turn, calls our search server.

The HTTP client for the Sim.API (`sim_client.py`) has NO ML imports so it can be used
read-only (auth/verify, budget/check) without loading torch/faiss.
"""
