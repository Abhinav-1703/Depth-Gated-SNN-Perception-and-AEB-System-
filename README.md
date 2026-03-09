# Depth-Gated-SNN-Perception-and-AEB-System-
•	Deployed SpikeYOLO perception stack on Jetson AGX Orin ingesting RealSense D415 RGB-D streams; Velodyne HDL-32E LiDAR used for ground-truth validation.
•	Implemented depth-gating pre-processing to sparsify SNN inputs, reducing inference latency and energy versus dense CNN baselines.
•	On hazard detection, Jetson transmits a binary brake trigger via TCP/IP to Speedgoat Real-Time Controller which drives a 12V DC linear actuator on the brake pedal with ~0.05 ms electronic delay.
•	Validated full end-to-end latency (camera capture to physical deceleration) on a modified Maruti Suzuki Versa EV platform (72V/10kW DC motor) using HIL under Simulink Real-Time
