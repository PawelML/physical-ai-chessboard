# Embodied AI Chess Lab Plan

## Project Goal

Build a safe physical AI chess demonstrator using a deterministic chess engine, robotic pick-and-place, RGB-D board verification, constrained AI agents, and an optional VLM/Cosmos observer layer.

The core principle is simple:

```text
game state -> legal move -> physical move plan -> robot execution -> camera verification -> state update
```

AI must not directly control the robot. AI can propose moves, explain decisions, comment on the scene, and compare observations with deterministic verification, but every physical action must pass through validators and safety gates.

## Target Hardware

- Robot: Waveshare High-torque Serial Bus Servo RoArm-M2 Desktop Robotic Arm Kit - M2-S
- Camera: Orbbec Gemini 336 / CAMERA DEPTH MV GEMINI 336 BULK
- Compute: local workstation with RTX 3090 or equivalent GPU
- Board: fixed chessboard with visual calibration markers
- Pieces: robot-friendly pieces with repeatable bases and enough spacing between squares

## System Architecture

```text
Web UI / Demo Console
  - live board
  - camera preview
  - planned move
  - robot status
  - verification result
  - logs

Game Orchestrator
  - game state
  - turn handling
  - event timeline
  - replay/export

Chess Arbiter
  - python-chess board state
  - legal move validation
  - game-end detection
  - PGN export

Move Agents
  - random agent
  - Stockfish agent
  - constrained LLM persona agent

Move Planner
  - chess move to physical steps
  - captures
  - castling
  - promotion
  - en passant

Safety Supervisor
  - low-speed mode
  - workspace limits
  - stop/pause state
  - validation gates before robot motion

Robot Controller
  - robot abstraction interface
  - simulated adapter
  - RoArm-M2 adapter
  - pick/place primitives

Perception Service
  - RGB-D capture
  - board calibration
  - ArUco/AprilTag marker detection
  - square occupancy verification
  - debug overlays

LLM / VLM Layer
  - legal-candidate move selection
  - commentary
  - scene description
  - optional Cosmos/VLM observer
```

## Source of Truth

Hard sources of truth:

- `Chess Arbiter`: validates chess legality.
- `Safety Supervisor`: validates whether a physical action may run.
- `Perception Service`: verifies board occupancy after robot execution.

Soft AI layers:

- LLM move persona
- LLM commentary
- VLM/Cosmos scene observer
- generated reports

The LLM may only choose from legal candidate moves provided by the deterministic system. It must never emit arbitrary robot commands.

## MVP Definition

The first working MVP should complete one reliable physical loop:

```text
legal move -> robot moves one piece -> camera verifies result -> game state updates
```

MVP requirements:

- Calibrated board relative to camera and robot.
- Known initial chess position.
- `python-chess` validates the requested move.
- Robot executes simple non-capture moves.
- Camera verifies that the source square is empty and the target square is occupied.
- UI shows planned move, robot command, verification result, board state, and logs.
- Event log records every step.

Do not include in MVP:

- Full piece classification from camera.
- Full autonomous board-state reconstruction.
- Promotion handling.
- Complex grip strategies.
- End-to-end robot control by LLM/Cosmos.
- Real-time Cosmos action generation.

## Implementation Phases

### Phase 1: Software-Only Chess Orchestrator

Build a complete chess workflow without hardware.

Deliverables:

- `python-chess` state machine.
- Legal move validator.
- Move agents:
  - random
  - Stockfish
  - LLM persona selecting from legal candidates
- Move decomposition layer.
- PGN export.
- Event log.
- Simple web UI with board state and logs.
- Simulated robot adapter.

Tests:

- Illegal moves are rejected.
- Legal moves update board state correctly.
- Captures decompose correctly.
- Castling decomposes into king move and rook move.
- Promotion is represented as a special physical case.
- En passant is represented as a special physical case.

### Phase 2: Board Calibration and Perception

Add camera-based board mapping and square occupancy detection.

Deliverables:

- Four ArUco or AprilTag markers on board corners.
- Camera-to-board homography.
- Square-center mapping.
- RGB-D capture pipeline.
- Occupancy detection per square.
- Debug image with detected markers and projected board grid.
- Calibration report with estimated square-center error.

Success criteria:

- Board grid overlay matches the real board.
- Occupancy detection is reliable for empty and occupied squares.
- Calibration can be repeated without code changes.

### Phase 3: Robot Executes Simple Moves

Connect the RoArm-M2 through a robot abstraction interface.

Initial supported moves:

- `e2e4`
- `g1f3`
- `b1c3`
- Other simple non-capture moves after calibration is stable.

Robot API shape:

```python
robot.pick(square="e2")
robot.place(square="e4")
perception.verify_empty("e2")
perception.verify_occupied("e4")
```

Deliverables:

- RoArm-M2 adapter.
- Safe home pose.
- Pick/place primitives.
- Square-to-robot target mapping.
- Low-speed demo mode.
- Emergency pause/stop state.
- Verification after every move.

Success criteria:

- Robot can repeatedly execute non-capture moves.
- Failed verification pauses the game.
- The board state updates only after successful verification.

### Phase 4: Full Physical Chess Semantics

Add special physical move handling.

Capture:

```text
1. remove captured piece to tray
2. move attacking piece to target square
```

Castling:

```text
1. move king
2. move rook
```

Promotion:

```text
1. move pawn to promotion square
2. replace pawn with promoted piece
```

En passant:

```text
1. move pawn
2. remove captured pawn from adjacent square
```

Example move decomposition:

```json
{
  "move": "e1g1",
  "type": "castling",
  "physical_steps": [
    {"action": "move_piece", "from": "e1", "to": "g1"},
    {"action": "move_piece", "from": "h1", "to": "f1"}
  ]
}
```

Deliverables:

- `MoveDecomposer`.
- Capture tray support.
- Special move tests.
- Physical replay logs.
- Recovery behavior on mismatch.

### Phase 5: AI, VLM, and Cosmos Demo Layer

Add AI capabilities after the deterministic system is stable.

LLM move selection:

```text
1. python-chess owns the current board state.
2. Stockfish generates top N legal candidate moves.
3. LLM receives only those legal candidates.
4. LLM chooses one move according to a persona.
5. Chess Arbiter validates the selected move again.
6. Robot executes only the approved PhysicalMovePlan.
```

Example LLM contract:

```json
{
  "move": "e2e4",
  "reason": "White takes central space and opens lines for development."
}
```

Supported personas:

- aggressive tactical
- defensive
- positional
- risk-taking
- technician mode

VLM/Cosmos observer:

- Receives image or short clip after a move.
- Describes what appears to have happened.
- Compares the scene with the planned move.
- Reports confidence and visible problems.
- Does not override deterministic validation.

UI comparison:

```text
AI observation:
The robot appears to have moved a white pawn forward.

Deterministic verification:
e2 empty, e4 occupied, move e2e4 confirmed.
```

## Safety Model

Create `docs/safety.md` early and keep safety visible in the repository.

Required safety constraints:

- Low-speed demo mode by default.
- Software workspace boundaries.
- Fixed board pose during a run.
- No robot movement until chess move validation passes.
- No robot movement until physical plan validation passes.
- No board-state update until camera verification passes.
- Pause on perception mismatch.
- Pause on robot error.
- Manual stop command available from UI and CLI.
- Logs for every robot command and validation result.

This is not a certification project, but the system should be designed and documented as a safety-gated robotic demo.

## Evaluation Plan

Create `docs/evals.md` and track measurable results.

Metrics:

- Legal move validation rate.
- LLM illegal proposal rejection count.
- Physical execution success rate.
- Occupancy verification accuracy.
- False occupied rate.
- False empty rate.
- Calibration error in millimeters.
- Recovery rate after mismatch.
- VLM/Cosmos agreement with deterministic verification.
- Commentary correctness.

Example report:

```md
# Eval Report

Run: 2026-06-05
Board: fixed ArUco board
Robot mode: low-speed
Games: 2
Physical moves: 84

| Metric | Result |
| --- | ---: |
| Legal move validation | 100% |
| Physical move success | 94.0% |
| Occupancy verification accuracy | 98.2% |
| Auto-pause on mismatch | 100% |
| LLM illegal proposals rejected | 7/7 |
| Average calibration error | 2.8 mm |
```

## Suggested Repository Structure

```text
physical-ai-chessboard/
  README.md
  plans/
    embodied-ai-chess-lab-plan.md
  docs/
    architecture.md
    safety.md
    calibration.md
    move-decomposition.md
    ai-agents.md
    evals.md
    demo-script.md
  src/
    chess_arena/
      orchestrator/
      chess/
      agents/
      perception/
      robot/
      safety/
      ui/
  tests/
    test_move_validation.py
    test_move_decomposition.py
    test_safety_supervisor.py
    test_board_mapping.py
  evals/
    perception/
    robot_runs/
    reports/
  media/
    screenshots/
    demo.mp4
  AGENTS.md
  docker-compose.yml
```

## Public Release Roadmap

### v0.1: Software Demo

- Software-only chess orchestrator.
- `python-chess` validator.
- Move decomposition.
- Robot abstraction interface.
- Simulated robot adapter.
- Simple web UI.
- Tests.
- `docs/architecture.md`.
- `docs/safety.md`.
- `docs/evals.md`.
- Demo GIF with simulator.

### v0.2: Camera Calibration

- Camera integration.
- ArUco/AprilTag board detection.
- Occupancy detection.
- Debug screenshots with board grid overlay.
- Calibration report.

### v0.3: Real Robot Non-Capture Moves

- RoArm-M2 adapter.
- Physical execution of non-capture moves.
- Verification after each move.
- Short demo video.

### v0.4: Full Physical Move Handling

- Captures.
- Castling.
- Promotion.
- En passant.
- Physical game replay.

### v0.5: AI Demonstration Layer

- LLM personas.
- Candidate-only LLM move selection.
- VLM/Cosmos observer.
- Generated match report.
- UI showing AI observation vs deterministic verification.

## Final Target Demo

The final demo should show:

- Two AI personas playing chess through constrained legal move selection.
- Deterministic chess validation for every move.
- Robot executing only approved physical plans.
- RGB-D camera verifying the board after each move.
- Safety supervisor pausing on errors or mismatches.
- UI showing board state, camera feed, tool calls, safety checks, and commentary.
- Match report with PGN, timeline, verification results, failures, recoveries, and AI commentary.

The project should communicate one core engineering message:

```text
Agentic AI in the physical world should be constrained, observable, validated, and safety-gated.
```
