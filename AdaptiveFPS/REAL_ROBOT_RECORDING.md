# Real-robot episode recording

The fixed and adaptive evaluators now share the passive recorder in
`scripts/real_episode_recorder.py`. No separate recorder process is needed.
The evaluator owns TD3 inference, scan gating, timeout handling, and the final
zero-velocity safety command. The recorder never publishes a ROS message and
never decides whether navigation continues.

## Current sources of truth

The implementation uses the current repository settings and real environment:

| Item | Current source/value |
| --- | --- |
| Success | `DrlStep.done` with `DrlStep.success == SUCCESS` |
| Collision | `COLLISION_WALL` or `COLLISION_OBSTACLE` returned by `DrlStep` |
| Timeout | evaluator-generated `TIMEOUT`; default `EPISODE_TIMEOUT_SECONDS` = 50 s |
| Goal threshold | `REAL_THRESHOLD_GOAL` = 0.35 m (metadata only in recorder) |
| Collision threshold | `REAL_THRESHOLD_COLLISION` = 0.30 m (metadata only in recorder) |
| Odometry | `/robot1/odom` |
| Goal | `/goal_pose` |
| Velocity command | `/robot1/cmd_vel_unstamped` |
| Raw scan | `/robot1/scan` |
| Gated scan | `/robot1/scan_gated` |

The real environment remains authoritative for success and collision. The
recorder does not reclassify either result using distance or LiDAR data.
`--timeout` overrides the 50-second default, and the evaluator stops its loop,
sends the existing safety stop, and passes `TIMEOUT` to the recorder.

## Episode lifecycle and files

Both evaluators wait for a new `/goal_pose` message and a matching `/goal_comm`
notification. The notification stays pending until the evaluator initializes
the episode; a service poll cannot consume it before the coordinates arrive.
Initialization acknowledges the goal and resets the real environment's
per-episode outcome, step, and distance state. A cached goal from an earlier
episode cannot start a new recording. Only after a goal, odometry, initial
state, and forced fresh scan are available is an episode directory created.

`--policy fixed_5Hz` selects this output directory:

```text
RealRobot/eval/
├── fixed_5Hz/
│   ├── episode_0001/
│   │   ├── trajectory.csv
│   │   └── metadata.json
│   ├── episode_0002/
│   │   ├── trajectory.csv
│   │   └── metadata.json
│   └── episodes.csv
└── adaptive_fc_0.0025/
    ├── episode_0001/
    │   ├── trajectory.csv
    │   └── metadata.json
    └── episodes.csv
```

The next number is one greater than the highest directory matching
`episode_XXXX`. Existing directories are never overwritten. On Ctrl+C, an
active episode keeps `trajectory.csv` and gets `metadata.partial.json`; it is
not added to `episodes.csv`.

One `trajectory.csv` row represents a 0.1-second recorder snapshot: the latest
command sent through `DrlStep`, odometry and goal, environment terminal state,
and sensing information visible when the independent record timer fires. An
initial row is written at episode time zero, and a terminal event forces one
final row immediately so it cannot be missed.
Its columns are:

```text
episode_time_s,x,y,yaw,goal_x,goal_y,cmd_linear,cmd_angular,
goal_distance,distance_traveled_m,current_fps,fresh_obs,
fresh_obs_counter,outcome,done
```

For example:

```csv
episode_time_s,x,y,yaw,goal_x,goal_y,cmd_linear,cmd_angular,goal_distance,distance_traveled_m,current_fps,fresh_obs,fresh_obs_counter,outcome,done
2.31,0.34,0.08,0.11,2.0,1.0,0.16,-0.23,1.823,0.352,5.0,1,12,RUNNING,0
```

`fresh_obs` is 1 only if at least one newly arrived physical LiDAR message was
actually forwarded by the integrated deadline-based gate since the previous
control row. `fresh_obs_counter` resets at episode start and increases only in
that forwarding callback. The initial forced-fresh scan is registered once.
`current_fps` is the scan rate at which the gate is currently configured. For
adaptive evaluation it is the last action that became causal and is controlling
the next scan interval. It is not an independently measured physical LiDAR rate.
`mean_fps` in `metadata.json` and `episodes.csv` is the arithmetic mean of all
`current_fps` values recorded by the 0.1-second trajectory timer.

Distance is calculated inside the passive recorder from consecutive, uniquely
timestamped `/robot1/odom` positions while an episode is active. It is reset to
zero at episode start, ignores motion before the start, and does not use the
real environment's sparsely updated `total_distance` value.

Episode time uses `time.monotonic()` and zero is the instant immediately before
the first TD3 transition. Rows are scheduled every 0.1 seconds independently
of TD3 transitions and scan arrivals.

Example `metadata.json` fields:

```json
{
  "episode": 1,
  "policy_name": "fixed_5Hz",
  "policy_type": "fixed",
  "success": true,
  "collision": false,
  "collision_wall": false,
  "collision_obstacle": false,
  "timeout": false,
  "tumble": false,
  "outcome_code": 1,
  "outcome_str": "SUCCESS",
  "episode_duration_s": 32.48,
  "distance_traveled_m": 4.83,
  "n_fresh_observations": 96,
  "mean_fps": 5.0,
  "start_x": 0.01,
  "start_y": -0.02,
  "final_x": 2.41,
  "final_y": 1.44,
  "goal_x": 2.5,
  "goal_y": 1.5,
  "final_goal_distance": 0.11,
  "goal_threshold": 0.35,
  "collision_threshold": 0.30,
  "timeout_seconds": 120.0,
  "fixed_requested_fps": 5.0
}
```

Adaptive metadata additionally contains the checkpoint path, sensing choices,
causal decision count, and per-FPS causal action counts. `episodes.csv` contains
one summary row per completed episode using the same result and aggregate
fields, for example:

```csv
episode,policy_name,policy_type,success,collision,collision_wall,collision_obstacle,tumble,timeout,outcome_code,outcome_str,episode_duration_s,distance_traveled_m,mean_fps,n_fresh_observations,start_x,start_y,final_x,final_y,goal_x,goal_y,final_goal_distance
1,fixed_5Hz,fixed,True,False,False,False,False,False,1,SUCCESS,32.48,4.83,5.0,96,0.01,-0.02,2.41,1.44,2.5,1.5,0.11
```

Every completed episode prints its number, policy, outcome, success/collision/
timeout flags, duration, path distance, mean current FPS, fresh-observation count,
start/goal/final coordinates, final goal distance, and save directory.

## Build once

The evaluator/recorder scripts run directly from the repository, but the real
environment contains the one-shot goal notification used to reject stale
goals. Build and re-source it once after updating the repository:

```bash
cd /home/turtlebot3_drlnav
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select turtlebot3_drl
source install/setup.bash
```

This machine is currently using ROS 2 Humble, so use `humble` here even if an
older upstream README mentions Foxy. With `--symlink-install`, later edits only
to files under `AdaptiveFPS/scripts/` require restarting the evaluator, not a
colcon rebuild.

## Run fixed-rate evaluation

Start/open the existing host-networked container in each terminal as needed:

```bash
docker start tb3_drl
docker exec -it tb3_drl bash
```

In every container terminal use the same environment as the robot. The current
robot domain from this setup is 5:

```bash
cd /home/turtlebot3_drlnav
source /opt/ros/humble/setup.bash
source install/setup.bash
export TURTLEBOT3_MODEL=burger
export ROS_DOMAIN_ID=5
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export DRLNAV_BASE_PATH=/home/turtlebot3_drlnav
```

Terminal 1 — the only real environment:

```bash
ros2 run turtlebot3_drl real_environment
```

Terminal 2 — fixed 5 Hz, one episode, 120-second timeout:

```bash
python3 AdaptiveFPS/scripts/real_eval_fixed.py \
  --fps 5 \
  --policy fixed_5Hz \
  --timeout 120 \
  --episodes 1 \
  --nav-model-run examples/td3_0_stage9 \
  --nav-episode 7400
```

For fixed 1 Hz, use:

```bash
python3 AdaptiveFPS/scripts/real_eval_fixed.py \
  --fps 1 \
  --policy fixed_1Hz \
  --timeout 120 \
  --episodes 1 \
  --nav-model-run examples/td3_0_stage9 \
  --nav-episode 7400
```

Terminal 3 — only after Terminal 2 says it is waiting for a new goal:

```bash
./spawn_goal.sh 1.0 1.0
```

The results are under `/home/turtlebot3_drlnav/RealRobot/eval/fixed_5Hz/`
or `/home/turtlebot3_drlnav/RealRobot/eval/fixed_1Hz/`.

## Run adaptive evaluation

Use the same Terminal 1 and Terminal 3. In Terminal 2 replace the fixed command
with:

```bash
python3 AdaptiveFPS/scripts/real_eval_adaptive.py \
  --adaptive-model AdaptiveFPS/models/adaptive_fps_model_fc_0_005_500k.pt \
  --policy adaptive_fc_0.005_500k \
  --timeout 120 \
  --episodes 1 \
  --nav-model-run examples/td3_0_stage9 \
  --nav-episode 7400
```

Results are saved under
`/home/turtlebot3_drlnav/RealRobot/eval/adaptive_fc_0.005_500k/`.

Do not simultaneously run `real_agent`, `real_scan_gate.py`, another fixed or
adaptive evaluator, another `real_environment`, or any other node publishing
competing velocity commands. Each evaluator already integrates the scan gate
and recorder; the deadline scheduler, TD3 behavior, recurrent PPO timing, and
adaptive next-interval causality are unchanged.
