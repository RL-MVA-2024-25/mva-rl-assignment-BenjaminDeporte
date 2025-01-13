with open(file="../src/score.txt", mode="r") as f:
    score_agent = float(f.readline())
    score_agent_dr = float(f.readline())
    
if score_agent < 3432807.68039157:
    note = 0
elif (score_agent >= 3432807.680391572) and (score_agent<1e8):
    note = 1
elif (score_agent >= 1e8) and (score_agent<1e9):
    note = 2
elif (score_agent >= 1e9) and (score_agent<1e10):
    note = 3
elif (score_agent >= 1e10) and (score_agent<2e10):
    note = 4
elif (score_agent >= 2e10) and (score_agent<5e10):
    note = 5
elif score_agent >= 5e10:
    note = 6
else:
    print(f"Erreur dans le score single patient")
    
if score_agent_dr < 1e10:
    note_dr = 0
elif (score_agent_dr >= 1e10) and (score_agent_dr<2e10):
    note_dr = 1
elif (score_agent_dr >= 2e10) and (score_agent_dr<5e10):
    note_dr = 2
elif score_agent_dr >= 5e10:
    note_dr = 3
else:
    print(f"Erreur dans le score random patient")
    
print(f"score_agent = {score_agent:.2e}, note = {note}/6")
print(f"score_agent_dr = {score_agent_dr:.2e}, note = {note_dr}/3")