# 3_simple_dfa.py

class SimpleDFA:
    def __init__(self, threshold=0.7):
        self.state = "Normal"
        self.threshold = threshold
        self.malicious_threshold_count = 0

    def step(self, mal_prob):
        # In a simple DFA, we move between states based on a single condition
        if self.state == "Normal":
            if mal_prob >= self.threshold:
                self.state = "Intrusion"
        
        elif self.state == "Intrusion":
            # Stay in intrusion unless the probability drops significantly
            if mal_prob < self.threshold * 0.5:
                self.state = "Normal"
            # Once an intrusion is detected, we keep the state until traffic becomes normal
            # This is a fixed, deterministic rule.
        
        return self.state