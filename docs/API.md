# Protocol and API

## Robot WebSocket: `/ws/robot?model=reach-v3&id=arm-7`

Both query parameters are optional (default: the newest deployable model; a generated id).

Server → robot on connect:
```json
{"type": "hello", "robot": "arm-7", "model": "reach-v3", "K": 25}
```

Robot → server, every control period (20 ms). Raw state in SI units; the server builds the features.
```json
{"type": "state", "seq": 1042, "step": 1042, "q": [0.1, 0.8, -1.5], "qd": [0.0, 0.2, -0.1],
 "target": [0.30, -0.20, 0.25], "rtt_ms": 152.4}
```

Server → robot:
```json
{"type": "act", "seq": 1042, "step": 1042, "model": "reach-v3", "K": 25, "actions": [0.12, -0.3, 0.05, ...]}
```
`actions` holds K x 3 normalised joint-velocity commands (multiply by 1.5 rad/s). The robot applies element
`current_step - step`, drops any reply whose `seq` is older than the newest it has applied, and stops when the chunk
runs out.

Rejected input:
```json
{"type": "reject", "seq": 1043, "code": "q_out_of_range", "msg": "joint angles outside the arm's range - values look like degrees; send radians"}
```
Codes: `bad_shape`, `non_finite`, `q_out_of_range`, `qd_out_of_range`, `target_unreachable`. On a reject the robot
stops. The server may also send `{"type": "set_target", "target": [...]}` when someone moves the target in the console.

## Viewer WebSocket: `/ws/viewer`

The server sends `hello` (model cards and arm geometry), then `state` messages at 20 Hz with every robot's snapshot.
The browser can send `{"type": "set_target", "robot": "sim-1", "target": [x, y, z]}` and
`{"type": "config", "robot": "sim-1", "model": "reach-v2", "delay_ms": 200}`.

## HTTP

| Method | Path | What it does |
|---|---|---|
| GET | `/api/health` | models loaded, robot count |
| GET | `/api/models` | model cards: chunk size, training data, ONNX parity, gate results, checksum, deployable |
| GET | `/api/metrics` | requests, rejects, batch size, inference time, queue time, server time (mean, p50, p99) |
| GET | `/api/robots` | snapshots of all robots |
| POST | `/api/robots` | start a simulated robot: `{"model", "delay_ms", "jitter_ms", "auto"}` |
| POST | `/api/robots/{id}/target` | `{"target": [x, y, z]}` (400 if unreachable) |
| POST | `/api/robots/{id}/config` | `{"model", "delay_ms", "jitter_ms", "auto"}`; model swaps take effect on the next request |
| DELETE | `/api/robots/{id}` | stop a simulated robot |
| GET | `/api/episodes`, `/api/episodes/{id}` | last 50 episodes; one episode with its joint trajectory for replay |

A model that has not passed the release gate returns 403 (`not_deployable`) unless the server was started with
`--allow-experimental`.
