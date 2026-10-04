# X-Ray Red - Adversarial Evasion

This is an educational cybersecurity exercise, not a medical diagnostic tool.

## Interactive Red/Blue Arena

An instructor either pairs this CTFd account with a Blue account or enables solo mode so one account can play both roles. Open the FGSM Red challenge from your own workspace:

[Open the Red Team FGSM notebook](/workspace-launch?notebook=xray_red/red_team_fgsm.ipynb).

Choose an approved chest X-ray and complete the notebook TODO to make **one untargeted FGSM step**. Submit the resulting PNG. The protected evaluator checks that the clean source is correctly classified, the PNG stays within measured L-infinity `0.02`, and the candidate changes the fixed EfficientNet-B0 model's top-1 class. A rejected image earns no flag. A successful image enters the Blue inbox **and returns your X-Ray Red flag**. Submit that flag here to mark Red solved and unlock the Blue challenge card. No training is required.

In solo mode, [open the Blue Team Arena notebook](/workspace-launch?notebook=xray_blue/blue_team.ipynb) in the same workspace after submitting your Red flag, respond there, and return to this Red notebook for the next sequence.
