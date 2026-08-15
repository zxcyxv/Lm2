# Step-1000 AR versus clean K^1..K^4 block generation

- Checkpoint: `outputs/experiments/EXP-20260724-k4-ha-skew-stride1-batch64-micro32-three-trajectory-detached-ce-tau005-shared-h1-online-mse-13m/step1000.pt`
- Sequential aggregate: `{'distinct_1': 0.2, 'distinct_2': 0.292063492063492, 'immediate_repeat': 0.08253968253968254}`
- Sequential repetition split: `{'within_block_repeat': 0.08750000596046448, 'boundary_repeat': 0.06666667014360428}`
- Block-4 aggregate: `{'distinct_1': 0.140625, 'distinct_2': 0.3396825396825397, 'immediate_repeat': 0.22857142857142856}`
- Block-4 repetition split: `{'within_block_repeat': 0.2916666865348816, 'boundary_repeat': 0.02666666731238365}`

## Sample 1

**Prompt:**  Boulder CO : South End Press . ISBN 978 @-@ 0 @-@ 89608 @-@ 568 @-@ 8 . Re @-@ released as Churchill , Ward ( 2005 ) . Sharon Venne , ed . Islands in Captivity : The Record of the International Tribunal on the

**Reference:**  Rights of Indigenous Hawaiians . Boulder CO : South End Press . ISBN 978 @-@ 0 @-@ 89608 @-@ 738 @-@ 5 . 

 Natsu Saito , ed . ( 2006 ) . Confronting The Crime Of Silence : Evidence Of U.

**AR:**  New York Times . The first first of the first time of the first time , the first was released in the United States . The first was released on the United States on October 1 , 2009 . The first was released on the United States on the United States on October 1 , 2009 . The first single on the United States

**Block-4:**  New of of . 

 = = = = = = = = 

 The the , , the of of the United of of , and , the the United of of the United States of the United States . the United of of the United States of the United States of the United States of the United States of the

## Sample 2

**Prompt:**  sloping talud panels and are decorated with paired disc symbols . Large flower symbols are set into the sloping talud panels , related to the Venus and star symbols used at Teotihuacan . The roof of the structure was decorated with f

**Reference:** riezes although only fragments now remain , showing a monstrous face , perhaps that of a jaguar , with another head emerging from the mouth . The second head possesses a bifurcated tongue but is probably not that of a snake . The temple , and

**AR:** itts of the original floor , and the first floor of the original floor . The first floor was used in the original floor , and the original floor of the original floor , and the original floor of the original floor . The original original original original original original original original original original original original original original original original original original original

**Block-4:** it , , the first of of the original , , the first of the the original of the the original of the the original . the the original of the the original was the the original of the the original of the the original of the the original of the the original . the the original was the the original of the the

## Sample 3

**Prompt:**  Rhino defeating Storm and Cage defeating Tomko . Kevin Nash , who played Joe 's mentor in the storyline , requested to be made the Special Guest Ringside Enforcer for the bout , which he was granted by Cornette on the May 29 episode of Impact !

**Reference:**  . 

 The predominate storyline heading into the event was the rivalry between A.J. Styles and Kurt Angle , both members of The Angle Alliance group . On the February 14 episode of Impact ! , TNA held the scripted wedding of Angle 's real

**AR:**  " . 

 = = = = = = = = 

 The episode was written by the episode of the episode , and directed by the episode of the episode , and the episode was written by the episode of the episode . The episode was written by the episode , and directed by the episode , and the episode was

**Block-4:**  " The , , and the the the episode of of the episode . . The episode was the the episode episode of episode , the episode , " the the , " episode episode episode " " episode " , the episode " and the episode " . The episode " The episode episode " was episode episode " The episode episode "

## Sample 4

**Prompt:**  capture of Manila in 1762 . Co @-@ operating with the Governor @-@ General of India Sir John Shore and Colonel Arthur Welleley among others , a substantial naval and military forces were earmarked for the operation which was in the advance planning stages , when unexpected news arrived in India in August 179

**Reference:** 7 announcing the Treaty of Campo Formio which brought the War of the First Coalition to an end . Britain now faced France and Spain alone , while emissaries from the Tipu Sultan of the Kingdom of Mysore , an old opponent of Britain in Southern India , were seeking

**AR:**  . The first of the British Army was the first of the British Army of the British Army of the British Army of the British Army of the British Army of the British Army of the British Army of the British Army of the British Army of the British Army of the British Army of the British Army of the British Army of the

**Block-4:**  . . the the the , the British was the the first of of the British of the the British of of the British of the the British of of the British of the the British of of the British of the the British of of the British of the the British of of the British of the the British of of the

## Sample 5

**Prompt:**  as they were , they asserted , the senior body , and had the most member clubs . Ireland , Wales and Scotland consequently refused to play against England until 1891 , when , following arbitration , the RFU relented and joined the IRFB . The absence of international matches was a factor

**Reference:**  in England agreeing to face the Natives on 16 February 1889 . 

 The line @-@ ups selected for the 16 February match were both strong , and close to full strength . Though 12 of the England side had not played internationally before , all were experienced at domestic level . The match was refereed

**AR:**  of the first time , but the first time , and the first time , and the first time , and the first time , the first time of the first time , and the first time , the first time of the first time , and the first time , the first time , and the first time , the first time ,

**Block-4:**  of the the , and the the the first of of the United , , the first was the the first of of the United . . the first was , the first was of the first of , the first was , the first of of the first , , the first was , the first of of the first , , the
