#!/usr/bin/env python3
from datetime import date
#NOTE: Just a toy predictor for now, outputs 0s. 
ZONES = 29
HOURS = 24

#!/usr/bin/env python3
from datetime import date

ZONES, HOURS = 29, 24

def main():
    tokens = [f"\"{date.today().isoformat()}\""]
    tokens += ["0"] * (ZONES * HOURS)   # L_i_00..23 for i=1..29
    tokens += ["0"] * ZONES             # PH_1..PH_29
    tokens += ["0"] * ZONES             # PD_1..PD_29
    print(", ".join(tokens))

if __name__ == "__main__":
    main()
