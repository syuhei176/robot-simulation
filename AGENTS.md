# AGENTS.md

## Project Goal

Find a practical body plan for a small indoor robot by using simulation to compare morphology, actuation requirements, and locomotion behavior before committing to hardware.

The current work is exploratory. The repo should help answer questions such as:

- What shape can move reliably indoors with low mass and low power?
- Can a floor-crawling snake or modular chain climb stairs without oversized motors?
- How do link count, body length, module mass, and gait shape change the required torque?
- When is a reconfigurable modular robot worth the added docking, wiring, and control complexity?

The earlier six-identical-module snake/quadruped concept remains a candidate, not a fixed requirement. A successful prototype should be guided by simulation results first, then narrowed into a hardware architecture that can actually be built.

## Current Focus

- Use simple static models first to size motors and reject obviously impractical shapes.
- Add dynamic contact simulation for stairs, edges, slipping, and impact loads.
- Compare candidate morphologies on mass, torque margin, energy use, mechanical complexity, and controllability.
- Keep the simulation code organized so results can be traced back to assumptions.
- Prefer concrete numeric outputs over speculative design discussion.

## Design Constraints

- Do not assume the final body plan too early; use simulation to justify it.
- Treat mass, torque, grip, stair geometry, and gait stability as first-class constraints.
- Prefer lightweight designs that keep actuator requirements in cheap, available motor classes.
- Separate static sizing from dynamic validation; both are needed before hardware decisions.
- When evaluating modular or reconfigurable designs, include docking, undocking, electrical connection, and communication complexity in the cost.
- If using identical modules, justify the number of modules and connection topology from locomotion results rather than aesthetics.
- Avoid designs that only work in a simplified simulator and ignore contact, slipping, or holding torque.

## Practical Milestones

1. Static motor-sizing model for candidate chain shapes.
2. Shape sweep across length, mass, link count, stair dimensions, and gait assumptions.
3. Dynamic contact simulation with stairs, friction, and motor torque logging.
4. Ranked shortlist of viable body plans with estimated actuator classes and failure modes.
5. Hardware-oriented design freeze for the most promising morphology.
6. Prototype build and validation against the simulation predictions.
