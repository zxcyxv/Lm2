# i-ResNet spectral checkpoint: block-K^2 generation at step 1000

- Checkpoint: `outputs/experiments/EXP-20260724-k1-ha-skew-iresnet-spectral-learned-noise-13m/step1000.pt`
- Step: 1000
- Prompts: 5 fixed validation samples, 64 BPE tokens each
- Continuation: 64 greedy BPE tokens
- Sequential aggregate: `{'distinct_1': 0.196875, 'distinct_2': 0.2634920634920635, 'immediate_repeat': 0.23809523809523808}`
- Block-K^2 aggregate: `{'distinct_1': 0.3125, 'distinct_2': 0.6158730158730158, 'immediate_repeat': 0.19365079365079363}`
- Noise draws: 4; seeds `4367..4370`
- Noisy recurrence: `z1 = K hA + eps1`; `z2 = K z1 + eps2 = K^2 hA + K eps1 + eps2`
- First noisy aggregate: `{'distinct_1': 0.3, 'distinct_2': 0.5746031746031746, 'immediate_repeat': 0.24126984126984125}`
- First noisy diagnostics: `{'log_sigma_mean': -6.10231876373291, 'log_sigma_min': -7.889276027679443, 'log_sigma_max': -4.787836074829102, 'relative_noise_norm_mean': 0.0026286495849490166, 'relative_noise_norm_max': 0.0032455960754305124, 'k_linearity_expansion_max_abs': 4.76837158203125e-06}`

## Sample 1

**Prompt:**  Boulder CO : South End Press . ISBN 978 @-@ 0 @-@ 89608 @-@ 568 @-@ 8 . Re @-@ released as Churchill , Ward ( 2005 ) . Sharon Venne , ed . Islands in Captivity : The Record of the International Tribunal on the

**Reference:**  Rights of Indigenous Hawaiians . Boulder CO : South End Press . ISBN 978 @-@ 0 @-@ 89608 @-@ 738 @-@ 5 . 

 Natsu Saito , ed . ( 2006 ) . Confronting The Crime Of Silence : Evidence Of U.

**Sequential (1 token/re-grounding)** (`{'distinct_1': 0.1875, 'distinct_2': 0.20634920634920634, 'immediate_repeat': 0.49206349206349204}`):  National Parks , and the United States . 

 = = = = = 

 = = = = = 

 = = = = = 

 = = = = = 

 = = = = = 

 = = = = = 

 = = = = = 

 = = = =

**Block-K^2 (2 tokens/re-grounding)** (`{'distinct_1': 0.421875, 'distinct_2': 0.7777777777777778, 'immediate_repeat': 0.14285714285714285}`):  National PSle , PSle , , and and the the Year PSS . . 

 The first @-@ year year @-@ , was the first @-@ year year @-@ , and the Year @-@ Man in the @-@ year Section @-@ 1 ( 2011 @-@ 0 ) , 

 The = = Sectionhip

**Block-K^2 cumulative learned noise, first draw** (`{'distinct_1': 0.234375, 'distinct_2': 0.4603174603174603, 'immediate_repeat': 0.36507936507936506}`):  National PSle , PSS , ( A MSS ) ) , and the the National PSS ( ( A MSS ) ) . 

 = = = MSS = = = 

 = = = MSS = = = 

 = = = M

## Sample 2

**Prompt:**  sloping talud panels and are decorated with paired disc symbols . Large flower symbols are set into the sloping talud panels , related to the Venus and star symbols used at Teotihuacan . The roof of the structure was decorated with f

**Reference:** riezes although only fragments now remain , showing a monstrous face , perhaps that of a jaguar , with another head emerging from the mouth . The second head possesses a bifurcated tongue but is probably not that of a snake . The temple , and

**Sequential (1 token/re-grounding)** (`{'distinct_1': 0.25, 'distinct_2': 0.3333333333333333, 'immediate_repeat': 0.047619047619047616}`): lesex , and the creek , and the creek is a common time to the creek . The creek is a common common time , and the creek is a common time to the creek . The creek is a common common time , and the creek is a common common time . 



**Block-K^2 (2 tokens/re-grounding)** (`{'distinct_1': 0.34375, 'distinct_2': 0.6825396825396826, 'immediate_repeat': 0.2857142857142857}`): lesroupupss , the c cotylan , and the c cotylanan . . 

 The cmmot is is a a common @-@ t tentent , , and a c @-@ shaped tentent , , and and a a c @-@ shaped tentent . .

**Block-K^2 cumulative learned noise, first draw** (`{'distinct_1': 0.453125, 'distinct_2': 0.7619047619047619, 'immediate_repeat': 0.19047619047619047}`): lesroupupss , the c cotylan , and the c cotylanan . . 

 The cmmot is is a a common @-@ t codylanan , , which a c @-@ shaped fierus , gentates , the c cotylan ,

## Sample 3

**Prompt:**  Rhino defeating Storm and Cage defeating Tomko . Kevin Nash , who played Joe 's mentor in the storyline , requested to be made the Special Guest Ringside Enforcer for the bout , which he was granted by Cornette on the May 29 episode of Impact !

**Reference:**  . 

 The predominate storyline heading into the event was the rivalry between A.J. Styles and Kurt Angle , both members of The Angle Alliance group . On the February 14 episode of Impact ! , TNA held the scripted wedding of Angle 's real

**Sequential (1 token/re-grounding)** (`{'distinct_1': 0.078125, 'distinct_2': 0.1111111111111111, 'immediate_repeat': 0.5396825396825397}`):  . 

 = = = Reception = = 

 = = = = = 

 = = = = = 

 = = = = = 

 = = = = = 

 = = = = = 

 = = = = = 

 = = = = = 

 = = = =

**Block-K^2 (2 tokens/re-grounding)** (`{'distinct_1': 0.1875, 'distinct_2': 0.25396825396825395, 'immediate_repeat': 0.12698412698412698}`):  . " 

 Togetheron " was released a series " , Togethered " " , Togetheron " " , Togetheron " " , Togetheron " " , Togetheron " " , Togetheron " " , Togetheron " " , Togetheron " " , Togetheron

**Block-K^2 cumulative learned noise, first draw** (`{'distinct_1': 0.21875, 'distinct_2': 0.4126984126984127, 'immediate_repeat': 0.19047619047619047}`):  . " 

 Togetheron " 'ss " " was Togethered by " The Takeon " " , Togetheron " " , Togetheron " " , Togetheron " " , Togetheron ' 'ss " " , Togetheron ' 'ss " " . T

## Sample 4

**Prompt:**  capture of Manila in 1762 . Co @-@ operating with the Governor @-@ General of India Sir John Shore and Colonel Arthur Welleley among others , a substantial naval and military forces were earmarked for the operation which was in the advance planning stages , when unexpected news arrived in India in August 179

**Reference:** 7 announcing the Treaty of Campo Formio which brought the War of the First Coalition to an end . Britain now faced France and Spain alone , while emissaries from the Tipu Sultan of the Kingdom of Mysore , an old opponent of Britain in Southern India , were seeking

**Sequential (1 token/re-grounding)** (`{'distinct_1': 0.125, 'distinct_2': 0.14285714285714285, 'immediate_repeat': 0.047619047619047616}`): 9 . 

 = = = = Battle of the Battle of the Battle of the Battle of the Battle of the Battle of the Battle of the Battle of the Battle of the Battle of the Battle of the Battle of the Battle of the Battle of the Battle of the Battle of the Battle of the Battle of the Battle of

**Block-K^2 (2 tokens/re-grounding)** (`{'distinct_1': 0.3125, 'distinct_2': 0.6349206349206349, 'immediate_repeat': 0.20634920634920634}`): 92 . . 

 The first @-@ year class was was a a member @-@ the class of of the the British @-@ American class ships the British @-@ American class ships ships , the British @-@ American class ships ships , the British British Army and the the British @-@ American class ships ships . . The The British @-@ class class

**Block-K^2 cumulative learned noise, first draw** (`{'distinct_1': 0.3125, 'distinct_2': 0.6031746031746031, 'immediate_repeat': 0.19047619047619047}`): 92 . . 

 The first @-@ year class was was a a member @-@ the class of of the the British @-@ American class ships the British @-@ American class ships ships , the British @-@ American class ships ships , the British @-@ American class ships ships . and the the British @-@ class class ships ships were the British British

## Sample 5

**Prompt:**  as they were , they asserted , the senior body , and had the most member clubs . Ireland , Wales and Scotland consequently refused to play against England until 1891 , when , following arbitration , the RFU relented and joined the IRFB . The absence of international matches was a factor

**Reference:**  in England agreeing to face the Natives on 16 February 1889 . 

 The line @-@ ups selected for the 16 February match were both strong , and close to full strength . Though 12 of the England side had not played internationally before , all were experienced at domestic level . The match was refereed

**Sequential (1 token/re-grounding)** (`{'distinct_1': 0.34375, 'distinct_2': 0.5238095238095238, 'immediate_repeat': 0.06349206349206349}`):  in the first round of the first season , and was the first @-@ year @-@ old @-@ class matches , and was the first @-@ year @-@ old @-@ class matches . 

 = = = Early life = = = 

 The first @-@ year @-@ old @-@ class matches , was born in the first round of the season

**Block-K^2 (2 tokens/re-grounding)** (`{'distinct_1': 0.296875, 'distinct_2': 0.7301587301587301, 'immediate_repeat': 0.20634920634920634}`):  in in the the United year , , and the first team of the year year . . 

 The first @-@ year class matches was the the first first season @-@ year class matches matches , the first first season @-@ season class matches matches , the first season , four matches @-@ class class matches matches . the season season was three

**Block-K^2 cumulative learned noise, first draw** (`{'distinct_1': 0.28125, 'distinct_2': 0.6349206349206349, 'immediate_repeat': 0.2698412698412698}`):  in in the the United year , , and the first team of the year year . . 

 The first first year @-@ old class matches , was the first first @-@ @-@ year class matches matches , the first first @-@ @-@ year class matches matches , the first first @-@ @-@ class class matches matches . in the the first season
