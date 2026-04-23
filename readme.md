Autonomous Fall-Back Communication for Critical Wireless Smart Grid Networks (WSGN)

Overview

This project tackles a specific problem in critical infrastructure: what happens when a Smart Grid's wireless network gets jammed or spoofed? Most current systems just log the error and fail. The goal here was to build a "Cognitive Radio" system that doesn't just detect the attack, but autonomously triggers a fallback mechanism—like frequency hopping—to keep the grid online.

Since we obviously couldn't jam a real power plant's network for data, I developed a stochastic simulation to model the physics of these attacks (Signal-to-Noise Ratio degradation, Jitter variance, etc.) and trained a Machine Learning model to recognize the signatures.

The Approach

The core of this project is the Autonomous Fallback Loop.

Instead of just checking for "Packet Loss" (which could just be bad weather), the system looks at the physical properties of the radio signal.

Jamming Detection: The model learns that a low SNR + high Noise Floor = Jamming. The response is to trigger FHSS (Frequency Hopping Spread Spectrum).

Spoofing Detection: It notices when a signal is strong (high RSSI) but has erratic timing (Jitter). The response is to lock the node and demand Multi-Factor Authentication.

I used a Random Forest Classifier for the detection engine because it handles the non-linear noise in radio data much better than linear models, and it’s fast enough for real-time embedded systems.

Getting Started

Prerequisites

You'll need a standard Python data science environment. I've listed the specific versions in requirements.txt, but generally, if you have pandas, scikit-learn, and seaborn, you are good to go.

To install dependencies:

pip install -r requirements.txt


Running the Simulation

The entire framework is self-contained in wsgn_security_framework.py.

Make sure the dataset (WSGN_Jamming_Spoofing_Dataset.csv) is in the same folder, then run:

python wsgn_security_framework.py


What to Expect (Output)

When you run the script, it performs the following steps automatically:

Physics Verification: It generates a plot (phy_layer_analysis.png) proving the data follows real-world radio physics (SNR vs PDR). This is useful for showing that the synthetic data is valid.

Model Training: It trains the Random Forest and validates it using 5-Fold Cross-Validation to ensure we aren't just overfitting to the synthetic patterns.

Live Simulation: The script ends by simulating 5 "live" intercepted signals. You'll see terminal output showing the decision logic in real-time, for example:

[Analysis]: Detected State -> JAMMING
>> CRITICAL ALERT: Spectrum Denial Detected.
>> FALLBACK ACTION: Initiating Frequency Hopping Spread Spectrum (FHSS).


Dataset Details

The dataset used here (WSGN-CP) is synthetic but mathematically rigorous. It models IEEE 802.15.4 (ZigBee) traffic patterns. I focused heavily on SNR (Signal-to-Noise Ratio) and Inter-Arrival Jitter as the primary features, as these are the hardest for an attacker to fake.

License

This project is for academic and research purposes. Feel free to use the code for your own cognitive radio or anomaly detection experiments.
