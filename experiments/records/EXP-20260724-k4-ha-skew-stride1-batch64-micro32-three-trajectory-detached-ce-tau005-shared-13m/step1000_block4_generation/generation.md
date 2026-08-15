# Step-1000 AR versus clean K^1..K^4 block generation

- Checkpoint: `outputs/experiments/EXP-20260724-k4-ha-skew-stride1-batch64-micro32-three-trajectory-detached-ce-tau005-shared-13m/step1000.pt`
- Sequential aggregate: `{'distinct_1': 0.23125, 'distinct_2': 0.36190476190476184, 'immediate_repeat': 0.21904761904761907}`
- Sequential repetition split: `{'within_block_repeat': 0.21666668355464935, 'boundary_repeat': 0.2266666740179062}`
- Block-4 aggregate: `{'distinct_1': 0.18125, 'distinct_2': 0.4031746031746032, 'immediate_repeat': 0.2095238095238095}`
- Block-4 repetition split: `{'within_block_repeat': 0.2666666805744171, 'boundary_repeat': 0.02666666731238365}`

## Sample 1

**Prompt:**  Boulder CO : South End Press . ISBN 978 @-@ 0 @-@ 89608 @-@ 568 @-@ 8 . Re @-@ released as Churchill , Ward ( 2005 ) . Sharon Venne , ed . Islands in Captivity : The Record of the International Tribunal on the

**Reference:**  Rights of Indigenous Hawaiians . Boulder CO : South End Press . ISBN 978 @-@ 0 @-@ 89608 @-@ 738 @-@ 5 . 

 Natsu Saito , ed . ( 2006 ) . Confronting The Crime Of Silence : Evidence Of U.

**AR:**  United States . 

 = = = = = = = = 

 The first of the United States , the United States , and the United States States States States States States . The United States States States States States States States States States States States States States States States States States States States States States States States States States States

**Block-4:**  United of . the United Statess the United States of the United States , the United States of the United States of the United States . the United States the the United States the the United States of the United States States the United States the the United States of the United States . the United States the the United States of the

## Sample 2

**Prompt:**  sloping talud panels and are decorated with paired disc symbols . Large flower symbols are set into the sloping talud panels , related to the Venus and star symbols used at Teotihuacan . The roof of the structure was decorated with f

**Reference:** riezes although only fragments now remain , showing a monstrous face , perhaps that of a jaguar , with another head emerging from the mouth . The second head possesses a bifurcated tongue but is probably not that of a snake . The temple , and

**AR:** rigate , and the first of the original building was used in the first time . The first was built in the first time , with a new building in the first time , and the first was built in the first time . The first was built in the first half of the first time , and the first first of the

**Block-4:** rig , , , and the , , and the , , the the the , the the the the first of the the original of the the original . the the original of the the original was the the original of of the original ' , the original was of the original ' of the original ' , the original was of the

## Sample 3

**Prompt:**  Rhino defeating Storm and Cage defeating Tomko . Kevin Nash , who played Joe 's mentor in the storyline , requested to be made the Special Guest Ringside Enforcer for the bout , which he was granted by Cornette on the May 29 episode of Impact !

**Reference:**  . 

 The predominate storyline heading into the event was the rivalry between A.J. Styles and Kurt Angle , both members of The Angle Alliance group . On the February 14 episode of Impact ! , TNA held the scripted wedding of Angle 's real

**AR:**  . 

 = = = = = = = = 

 The episode was written by the episode of the episode of the episode episode , and was written by the episode of the episode episode . The episode was written by the episode of the episode , directed by the episode of the episode . The episode was written by the

**Block-4:**  . 

 was also by by the episode of of of the. ' , who was was the first episode of episode , the episode by the. of . The episode was was written by by by the.man , who directed by by the.man . The episode was episode of by episode by the. , ,

## Sample 4

**Prompt:**  capture of Manila in 1762 . Co @-@ operating with the Governor @-@ General of India Sir John Shore and Colonel Arthur Welleley among others , a substantial naval and military forces were earmarked for the operation which was in the advance planning stages , when unexpected news arrived in India in August 179

**Reference:** 7 announcing the Treaty of Campo Formio which brought the War of the First Coalition to an end . Britain now faced France and Spain alone , while emissaries from the Tipu Sultan of the Kingdom of Mysore , an old opponent of Britain in Southern India , were seeking

**AR:** 7 . 

 = = = = = = = = 

 In the earlymath of the war , the British Army was appointed to the British Army of the British Army of the British Army of the British Army . The British Army Army Army Army Army Army Army Army , and the British Army Army Army Army Army Army

**Block-4:** 7 . 

 = = = , the of = = = 

 The British of of the British was was the first of of the British of of the British . . the British of the the British of the the British of the the British of the the British of the the British of the the British of the the

## Sample 5

**Prompt:**  as they were , they asserted , the senior body , and had the most member clubs . Ireland , Wales and Scotland consequently refused to play against England until 1891 , when , following arbitration , the RFU relented and joined the IRFB . The absence of international matches was a factor

**Reference:**  in England agreeing to face the Natives on 16 February 1889 . 

 The line @-@ ups selected for the 16 February match were both strong , and close to full strength . Though 12 of the England side had not played internationally before , all were experienced at domestic level . The match was refereed

**AR:**  of the first match , and the first match of the match , and the match was the first match . The match was the match of the match , and the match was the match of the match , the match match , and the match was the match . The match was the match match , the match match , and the

**Block-4:**  of the the , and the the the Australian of of the Australianth , the Australian of of the Australian Australian . . 

 = , the the = = 

 The the Australian Australian of the Australian Australian , the Australian Australian the the Australian Australian the Australian Australian Australian the the Australian Australian the the Australian Australian the the
