# 3_adaptive_fsm.py

from collections import deque
import numpy as np

class AdaptiveFSM:
    """
    States: Normal -> Suspicious -> Intrusion -> Alert -> (back to Normal)
    Adaptivity:
      - If recent malicious intensity (rolling mean of prob) rises, increase sensitivity.
      - If many false alarms estimated, decrease sensitivity.
    """
    def __init__(self, base_threshold=0.7, base_k=2, base_m=2, window=100):
        self.state = "Normal"
        self.base_threshold = base_threshold
        self.base_k = base_k# 3_adaptive_fsm.py

from collections import deque
import numpy as np

class AdaptiveFSM:
    """
    States: Normal -> Suspicious -> Intrusion -> Alert -> (back to Normal)
    Adaptivity:
      - If recent malicious intensity (rolling mean of prob) rises, increase sensitivity.
      - If many false alarms estimated, decrease sensitivity.
    """
    def __init__(self, base_threshold=0.7, base_k=2, base_m=2, window=100):
        self.state = "Normal"
        self.base_threshold = base_threshold
        self.base_k = base_k
        self.base_m = base_m
        self.window = window
        self.win_probs = deque(maxlen=window)
        self._mal_count = 0
        self._ben_count = 0

    def _current_params(self):
        if len(self.win_probs) == 0:
            return self.base_threshold, self.base_k, self.base_m

        mean_p = float(np.mean(self.win_probs))
        # Adapt threshold: higher mean malicious prob -> lower threshold (more sensitive)
        thr = np.clip(self.base_threshold - 0.2*(mean_p-0.5), 0.55, 0.9)

        # Adapt k: high mean malicious prob -> fewer confirmations needed
        k = int(np.clip(round(self.base_k - 1.0* (mean_p-0.5)), 1, 4))

        # Adapt m: if stream looks benign, require fewer benign confirmations to return
        m = int(np.clip(round(self.base_m - 1.0*(0.5-mean_p)), 1, 4))

        return thr, k, m

    def step(self, mal_prob):
        self.win_probs.append(mal_prob)
        thr, k2i, m2n = self._current_params()

        is_mal = mal_prob >= thr
        if is_mal:
            self._mal_count += 1; self._ben_count = 0
        else:
            self._ben_count += 1; self._mal_count = 0

        if self.state == "Normal":
            if is_mal:
                self.state = "Suspicious"

        elif self.state == "Suspicious":
            if self._mal_count >= k2i:
                self.state = "Intrusion"
            elif self._ben_count >= m2n:
                self.state = "Normal"

        elif self.state == "Intrusion":
            if self._mal_count >= k2i:
                self.state = "Alert"
            elif self._ben_count >= m2n:
                self.state = "Normal"

        elif self.state == "Alert":
            if self._ben_count >= m2n:
                self.state = "Normal"

        # Corrected: Return both the state and the parameters as a dictionary
        return self.state, {"thr": thr, "k": k2i, "m": m2n}
        self.base_m = base_m
        self.window = window
        self.win_probs = deque(maxlen=window)
        self._mal_count = 0
        self._ben_count = 0

    def _current_params(self):
        if len(self.win_probs) == 0:
            return self.base_threshold, self.base_k, self.base_m

        mean_p = float(np.mean(self.win_probs))
        # Adapt threshold: higher mean malicious prob -> lower threshold (more sensitive)
        thr = np.clip(self.base_threshold - 0.2*(mean_p-0.5), 0.55, 0.9)

        # Adapt k: high mean malicious prob -> fewer confirmations needed
        k = int(np.clip(round(self.base_k - 1.0*(mean_p-0.5)), 1, 4))

        # Adapt m: if stream looks benign, require fewer benign confirmations to return
        m = int(np.clip(round(self.base_m - 1.0*(0.5-mean_p)), 1, 4))

        return thr, k, m

    def step(self, mal_prob):
        self.win_probs.append(mal_prob)
        thr, k2i, m2n = self._current_params()

        is_mal = mal_prob >= thr
        if is_mal:
            self._mal_count += 1; self._ben_count = 0
        else:
            self._ben_count += 1; self._mal_count = 0

        if self.state == "Normal":
            if is_mal:
                self.state = "Suspicious"

        elif self.state == "Suspicious":
            if self._mal_count >= k2i:
                self.state = "Intrusion"
            elif self._ben_count >= m2n:
                self.state = "Normal"

        elif self.state == "Intrusion":
            if self._mal_count >= k2i:
                self.state = "Alert"
            elif self._ben_count >= m2n:
                self.state = "Normal"

        elif self.state == "Alert":
            if self._ben_count >= m2n:
                self.state = "Normal"

        return self.state, {"thr":thr, "k_to_intrusion":k2i, "m_to_normal":m2n}

# Example over ensemble probs from step 2:
# states, params_track = [], []
# fsm = AdaptiveFSM()
# for p in p_ens:
#     s, params = fsm.step(float(p))
#     states.append(s); params_track.append(params)

# print({k: sum(1 for x in states if x==k) for k in set(states)})
