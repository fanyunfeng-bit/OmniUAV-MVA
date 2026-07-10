# MM-UAVBench Question Types Summary

## Overview

The MM-UAVBench dataset contains **19 task categories** organized into 3 capability dimensions:

- **Perception**: Basic visual understanding
- **Cognition**: Complex reasoning and prediction
- **Planning**: Decision-making and coordination

---

## Single-UAV Scenario Questions

### Perception Tasks

| Task Type | Typical Question Template | Example |
|-----------|--------------------------|---------|
| **Scene Classification** | "What type of scene is shown in the image/region?" | What type of scene is shown in the aerial view (e.g., tennis court, traffic road, soccer field, parking lot)? |
| **Class Agnostic Counting** | "How many [objects] are visible in the image?" | How many sheep/yaks/cars are visible in this aerial image? |
| **Environment State Classification** | "What is the environmental condition in this scene?" | What is the weather/lighting condition (e.g., daytime, nighttime, foggy, rainy)? |
| **Orientation Classification** | "What is the motion direction of [target]?" | Given the vehicle in the bounding box, is it moving straight, turning left, turning right, or stationary? |
| **Referring Expression Counting** | "Count the number of objects matching the description." | How many red cars are parked near the building entrance? |
| **Urban OCR** | "What text is visible in the image?" | What text is visible on the road signs/buildings in this scene? |
| **Scene Damage Assessment** | "What is the damage level of the buildings?" | Assess the building damage severity: no damage, minor damage, major damage, or total destruction? |
| **Scene Attribute Understanding** | "What attributes describe the scene?" | What are the key characteristics of this urban/rural scene (e.g., road type, building density, vegetation)? |

### Cognition Tasks

| Task Type | Typical Question Template | Example |
|-----------|--------------------------|---------|
| **Event Understanding** | "What event is occurring in the video?" | What event is shown in the video (e.g., a baseball play, vehicle movement, pedestrian activity)? Describe the sequence of actions. |
| **Event Prediction** | "What will happen next/after the event?" | What is the most likely outcome of this baseball play? / What will happen after the video ends? |
| **Event Tracing** | "What is the complete sequence of the event?" | Describe the full trajectory and actions of the person/vehicle from start to finish. |
| **Target Backtracking** | "What was the target's previous state/location?" | Given the current position of the car, where did it come from and what was its path? |
| **Scene Analysis and Prediction** | "Analyze the scene and predict future states." | Based on the current traffic flow, what will happen in the next few minutes? |
| **Cross Object Reasoning** | "What is the relationship between objects?" | How are the vehicles and pedestrians interacting in this scene? |

### Planning Tasks

| Task Type | Typical Question Template | Example |
|-----------|--------------------------|---------|
| **Ground Target Planning** | "What is the optimal target location/action?" | Given the disaster scenario, which location is best for placing rescue supplies? / Which route should the vehicle take? |

---

## Multi-UAV Scenario Questions

### Perception Tasks
*Most perception tasks can support multi-UAV with multiple images, but the core questions remain similar to single-UAV.*

### Cognition Tasks

| Task Type | Typical Question Template | Example |
|-----------|--------------------------|---------|
| **Intent Analysis and Prediction** | "What is the intent of the target based on multiple perspectives?" | Given multiple views of a car at a T-junction, which direction will it turn (straight, left, right)? |
| **Temporal Ordering** | "What is the correct chronological order of these images?" | Arrange the images from different cameras/timeframes in the correct chronological sequence of the event. |

### Planning Tasks

| Task Type | Typical Question Template | Example |
|-----------|--------------------------|---------|
| **Swarm Collaborative Planning** | "How should multiple UAVs coordinate their actions?" | Given 3 UAVs with different viewing angles, where should a new UAV be positioned to capture facial expressions of the tracked person? Which regions need additional coverage? |
| **Air-Ground Collaborative Planning** | "How should UAV and ground agents coordinate?" | Between two ground agents, who should be dispatched to complete a task, and how should the UAV assist them (altitude adjustment, monitoring, guidance)? |

---

## Open-Ended Question Templates

For implementation in OmniUAV, here are converted open-ended question templates:

### Single-UAV - Perception

1. "Classify the scene type shown in this image."
2. "Count and report the number of [object category] visible."
3. "Identify the motion direction and state of the vehicle in the marked region."
4. "Extract and transcribe all visible text in the scene."
5. "Assess the damage level of buildings in this disaster area."

### Single-UAV - Cognition

1. "Describe the event unfolding in this video sequence."
2. "Predict what will happen immediately after this frame/video."
3. "Trace the complete path and history of the tracked target."
4. "Analyze the relationship and interactions between objects in the scene."

### Single-UAV - Planning

1. "Identify the optimal location for [action/resource placement] given the current scene."

### Multi-UAV - Cognition

1. "Using multiple camera perspectives, predict the target's intended action/direction."
2. "Determine the correct chronological ordering of these multi-view images."

### Multi-UAV - Planning

1. "Determine optimal UAV positioning for maximum coverage of the target/scene."
2. "Design a coordination plan between multiple UAVs and ground agents to complete the task."

---

## Key Characteristics

| Aspect | Single-UAV | Multi-UAV |
|--------|-----------|-----------|
| **Input** | Single image/video | Multiple images from different views |
| **Questions** | Direct perception/reasoning | Requires perspective integration |
| **Focus** | What/How questions | Where/How to coordinate questions |
| **Complexity** | Basic to intermediate | Advanced reasoning |

---

## Task Categories by Capability Dimension

### Perception Tasks
- Scene Classification
- Class Agnostic Counting
- Environment State Classification
- Orientation Classification
- Referring Expression Counting
- Urban OCR
- Scene Damage Assessment
- Scene Attribute Understanding

### Cognition Tasks
- Event Understanding
- Event Prediction
- Event Tracing
- Target Backtracking
- Scene Analysis and Prediction
- Cross Object Reasoning
- Intent Analysis and Prediction
- Temporal Ordering

### Planning Tasks
- Ground Target Planning
- Swarm Collaborative Planning
- Air-Ground Collaborative Planning

---

## Data Sources

The dataset incorporates data from multiple sources:
- **VisDrone**: UAV-based detection/tracking
- **ERA**: Event recognition and action
- **AIDER**: Disaster damage assessment
- **RescueNet**: Post-disaster building damage
- **MavRec**: MAV recordings
- **MDOT/Three-MDOT**: Multi-view UAV scenarios
- **AnimalDrone**: Animal counting from aerial views

---

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Video Clips | 1,549 |
| Total Images | 2,873 |
| Average Resolution | 1622 × 1033 |
| Total QA Pairs | ~226,000+ |
| Task Categories | 19 |
