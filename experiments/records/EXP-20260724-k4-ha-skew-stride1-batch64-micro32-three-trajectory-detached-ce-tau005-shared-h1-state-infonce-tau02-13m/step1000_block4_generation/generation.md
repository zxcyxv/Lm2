# Step-1000 AR versus clean K^1..K^4 block generation

- Checkpoint: `outputs/experiments/EXP-20260724-k4-ha-skew-stride1-batch64-micro32-three-trajectory-detached-ce-tau005-shared-h1-state-infonce-tau02-13m/step1000.pt`
- Sequential aggregate: `{'distinct_1': 0.25, 'distinct_2': 0.38730158730158726, 'immediate_repeat': 0.23492063492063492}`
- Sequential repetition split: `{'within_block_repeat': 0.2458333522081375, 'boundary_repeat': 0.20000000298023224}`
- Block-4 aggregate: `{'distinct_1': 0.1625, 'distinct_2': 0.30476190476190473, 'immediate_repeat': 0.3142857142857143}`
- Block-4 repetition split: `{'within_block_repeat': 0.38750001788139343, 'boundary_repeat': 0.07999999821186066}`

## Sample 1

**Prompt:**  Boulder CO : South End Press . ISBN 978 @-@ 0 @-@ 89608 @-@ 568 @-@ 8 . Re @-@ released as Churchill , Ward ( 2005 ) . Sharon Venne , ed . Islands in Captivity : The Record of the International Tribunal on the

**Reference:**  Rights of Indigenous Hawaiians . Boulder CO : South End Press . ISBN 978 @-@ 0 @-@ 89608 @-@ 738 @-@ 5 . 

 Natsu Saito , ed . ( 2006 ) . Confronting The Crime Of Silence : Evidence Of U.

**AR:**  New York Times , and the " The The Times Times Times , and the " The Times Times Times , and the " the film of the film of the film . " 

 = = = Film = = = 

 The film was released on the film in the film . The film was released on the film in

**Block-4:**  New . . . 

 = , the : = 

 The = = = @-@ = = = = 

 The = = = 

 

 = = = = = = = = 

 The = = = 

 

 = = = = = = = = 

 The = = = 

 

 = =

## Sample 2

**Prompt:**  sloping talud panels and are decorated with paired disc symbols . Large flower symbols are set into the sloping talud panels , related to the Venus and star symbols used at Teotihuacan . The roof of the structure was decorated with f

**Reference:** riezes although only fragments now remain , showing a monstrous face , perhaps that of a jaguar , with another head emerging from the mouth . The second head possesses a bifurcated tongue but is probably not that of a snake . The temple , and

**AR:** avate and the original floor . The first floor was used to be used in the first floor , and the first floor was used in the first floor . The first floor was used in the first floor , and was used in the first floor . The first floor was used in the first floor , and was used in the

**Block-4:** avers and the original of of the original . . the original was the the original was the the original of the the original was the the original of the the original , the the original was the the original of the the original . the the original was was the first of the the original , the the original was the the

## Sample 3

**Prompt:**  Rhino defeating Storm and Cage defeating Tomko . Kevin Nash , who played Joe 's mentor in the storyline , requested to be made the Special Guest Ringside Enforcer for the bout , which he was granted by Cornette on the May 29 episode of Impact !

**Reference:**  . 

 The predominate storyline heading into the event was the rivalry between A.J. Styles and Kurt Angle , both members of The Angle Alliance group . On the February 14 episode of Impact ! , TNA held the scripted wedding of Angle 's real

**AR:**  " . The episode was released on the episode , and was released on the second episode of the episode , and was released on the second episode of the episode . The episode was written by the episode and directed by the episode . The episode was written by by David David David and directed by David David David David David David David

**Block-4:**  " The was = 

 " " " was the episode " was the episode episode of the episode episode " the episode episode " the episode episode " the episode episode " " episode episode " " episode episode " " episode , " the episode episode " " episode episode " " episode , " the episode episode " " episode episode

## Sample 4

**Prompt:**  capture of Manila in 1762 . Co @-@ operating with the Governor @-@ General of India Sir John Shore and Colonel Arthur Welleley among others , a substantial naval and military forces were earmarked for the operation which was in the advance planning stages , when unexpected news arrived in India in August 179

**Reference:** 7 announcing the Treaty of Campo Formio which brought the War of the First Coalition to an end . Britain now faced France and Spain alone , while emissaries from the Tipu Sultan of the Kingdom of Mysore , an old opponent of Britain in Southern India , were seeking

**AR:** 4 . The British forces was not not be used in the British forces , but was not not not be used in the British . 

 The British forces were also used in the British forces , and the British forces were the British British forces in the British British British British British British British British British British British British British British

**Block-4:** 4 . 

 = = = , the = = = 

 In the early of the the British of of the British of of the British of of the British of of the British of of the British of of the British of of the British of of the British . . the British of the the British of the the

## Sample 5

**Prompt:**  as they were , they asserted , the senior body , and had the most member clubs . Ireland , Wales and Scotland consequently refused to play against England until 1891 , when , following arbitration , the RFU relented and joined the IRFB . The absence of international matches was a factor

**Reference:**  in England agreeing to face the Natives on 16 February 1889 . 

 The line @-@ ups selected for the 16 February match were both strong , and close to full strength . Though 12 of the England side had not played internationally before , all were experienced at domestic level . The match was refereed

**AR:**  in the first round of the Cup , and the first round of the Cup Cup . 

 In the first round , the Cup Cup Cup Cup Cup Cup Cup Cup Cup Cup Cup Cup Cup Cup Cup Cup Cup Cup Cup Cup Cup Cup Cup Cup Cup Cup Cup Cup Cup Cup Cup Cup Cup Cup Cup Cup Cup Cup Cup

**Block-4:**  in the the , and the the , and the the , was the the , and the the the first of of the United . . . The = , the the , , was the the , was the the , and the the , was the the , was the the the first of of the United , . the
