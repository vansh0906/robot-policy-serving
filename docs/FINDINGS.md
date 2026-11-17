# Findings

Numbers are from `python run_all.py` unless a row says "during development". The customer reports are modeled: they
describe problems a robotics customer would plausibly hit, and each one is reproduced before it is fixed.

| ID | Finding | Evidence | Change |
|---|---|---|---|
| F-01 | **Features must be built where they were built in training.** If each robot computed the 18 policy inputs itself, any difference (units, ordering, scaling) would silently change the policy's behaviour. | — | Robots send raw state in SI units (`q`, `qd`, `target`); the server validates it and builds the features, vectorised once per batch. ONNX output matches NumPy to ≤ 1.1e-6. |
| F-02 | **Customer report 1: the research model misses targets near the workspace edge.** It was trained on the lab's narrow target range. | reach-v1: 100% inside the lab range, **60.9%** outside (512 of 600 customer tasks are outside). | Retrained on the full reachable workspace (reach-v2): **99.8%** outside. A release gate (≥ 95% on the customer task set) now blocks reach-v1 from being served. |
| F-03 | **Customer report 2: robots on a slow link oscillate.** With one command per reply, the robot acts on a state that is a round trip old. | reach-v2 holding the newest command: 99.7–100% up to 150 ms RTT, **94.3%** at 200, **67.7%** at 250, **8.7%** at 300. Real WebSockets at 300 ms: **15%**. | reach-v3 predicts a 25-step (0.5 s) action chunk: **99.7%** at 250 ms, **98.7%** at 300, 88.7% at 400. Real WebSockets at 300 ms: **100%**. |
| F-04 | **Chunks only help if the robot skips the time spent in flight.** | Same reach-v3 chunks played from their first element when they arrive: **30.0%** at 300 ms, vs 98.7% time-indexed. At 500 ms RTT the 0.5 s chunk runs out and both fail. | Each reply carries the step its state was taken at; the robot applies element `now - obs_step`, and stops when the chunk runs out. Rule for deployment: chunk horizon > p99 RTT. |
| F-05 | **Replies arrive out of order when the network jitters.** | 300 ms + 75 ms jitter, 4 robots: **1,219** replies arrived after a newer one (reach-v3 run). | Sequence numbers; the robot drops any reply older than the newest one it has. |
| F-06 | **Bad robot input must not move the robot or take the server down.** | Tests: a NaN encoder reading gets `non_finite`; joint angles in degrees get `q_out_of_range` with "values look like degrees; send radians"; the server answers the next valid request normally. | Plain-Python validation on every request with error codes; the robot client stops the arm on any reject. |
| F-07 | **One inference call per request does not scale.** Each call crosses into a worker thread and ONNX Runtime separately. | One call per request: 0.08% late at 25 robots, **33% late at 50**, all late at 100. Dynamic batching: **0 late at 100 robots** (p99 7.6 ms, mean batch 5.2), 1.6% at 150, 13% at 200. | Dynamic batcher: takes every request already queued (up to 256) and runs one ONNX Runtime call per model. A 2 ms wait window was tested during development and made latency worse (p50 1.3 → 3.5 ms at 1 robot; more late replies at 100 robots), so it is off by default. |
| F-08 | **After batching, the server was limited by per-request Python overhead.** | Profiling during development: per-element rounding of the 75 reply numbers and per-request NumPy feature building were the largest costs. At 100 robots, p99 went from **20.2 ms (1.1% late) to 7.6 ms (0 late)** after the fix. | Round replies with one NumPy call; validate in plain Python; build features once per batch. |
| F-09 | **The console's robot table swallowed clicks.** It was rebuilt 20 times a second, so a row could be replaced between mouse-down and mouse-up. Found by the browser test. | Playwright click timed out: "element was detached from the DOM". | Rows are updated in place and rebuilt only when the robot list changes. |

## Limits of this project (next steps)

- **Simulator:** my own NumPy arm (velocity-controlled joints, actuator lag, limits). Next: the same policy loop in
  MuJoCo, with a manipulation task and a camera observation.
- **Scale:** one Python server process on a 2-vCPU machine shared with the load generator. Next: several worker
  processes behind a load balancer, a binary message format, and GPU batching for larger models.
- **Security:** no authentication or TLS. Next: per-robot tokens and `wss://`.
- **Policy:** behaviour cloning from a scripted expert. The serving path is model-agnostic: anything that exports to
  ONNX with the same observation schema can be registered with a model card.
