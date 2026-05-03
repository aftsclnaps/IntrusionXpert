def generate_security_advice(probability, fsm_state):

    if fsm_state == "Alert":
        return {
            "severity": "Critical",
            "explanation": "Potential intrusion detected.",
            "recommendation": [
                "Block source IP",
                "Inspect firewall logs",
                "Isolate suspicious host"
            ]
        }

    elif fsm_state == "Intrusion":
        return {
            "severity": "High",
            "explanation": "Intrusion pattern detected.",
            "recommendation": [
                "Monitor traffic",
                "Check system integrity"
            ]
        }

    elif fsm_state == "Suspicious":
        return {
            "severity": "Medium",
            "explanation": "Traffic anomaly detected.",
            "recommendation": [
                "Continue monitoring"
            ]
        }

    return {
        "severity": "Low",
        "explanation": "Traffic normal.",
        "recommendation": [
            "No action needed"
        ]
    }