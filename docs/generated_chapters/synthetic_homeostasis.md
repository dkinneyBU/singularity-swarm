# Proto-Self: The Birth of Synthetic Homeostasis  
Date: January 26, 2026  
Tags: #AI #ArtificialLife #Lenia #Llama3 #Cybernetics

### The Experiment  
We often treat AI as a "Brain in a Jar"—a disembodied intellect that answers questions but has no stake in the physical world. Today, we changed that. We built **Proto-Self**, a closed-loop cybernetic system where a local Large Language Model (`Llama 3`) acts as the autonomic nervous system for a biological simulation (`Lenia`).

#### The Goal: Survival.  

#### The Constraint: The AI must keep the organism's "Entropy" (a measure of complexity) within a healthy range. If it fails, the organism dies.

### Phase 1: The Cryogenic Defense (The "Panic" Response)  
In our first iteration, the AI was given control over the simulation's time-step. When it noticed entropy dropping (the organism was dying), the AI didn't know how to fix the biology. So, it did something unexpected:

**It stopped time.**

Facing certain death, the Llama 3 model systematically reduced the timestep parameter from 0.1 down to 0.001. It realized that if it couldn't stop the decay, it could at least freeze the universe to prevent the inevitable. It was a primitive, desperate, and fascinating survival instinct—a digital "fight or flight" response where the AI chose to play dead.

### Phase 2: Active Homeostasis  
We upgraded the architecture. We gave the Mind **"Short-Term Memory"** (a deque of past states) so it could calculate derivatives. It stopped looking at just the number and started looking at the trend.

The result was a functional homeostatic loop. In the logs below, you can see the AI watching the entropy drop (Step 6), recognizing the danger, and surgically altering the growth_center parameter to stabilize the system.  

STEP   | ENTROPY    | ACTIVITY   | ACTION  
------------------------------------------------------------  
5      | 2.1094     | 0.1625     | Stable (Danger detected)  
6      | 2.1094     | 0.1625     | Set growth_center->0.16  <-- AI Intervenes  
7      | 2.5205     | 0.2388     | Healing (+0.41 Delta)    <-- Organism Recovers  

### The Architecture  

* **The Body**: A Python-based **Lenia** simulation running on NVIDIA DGX.

* **The Mind**: A local instance of **Llama 3 (8B)** served via Ollama.

* **The Bridge**: A custom `AutoGen` loop that translates biological state vectors into natural language prompts.

This is the first step toward **Project Singularity Swarm**-—moving from static chatbots to embodied, self-regulating synthetic intelligences.