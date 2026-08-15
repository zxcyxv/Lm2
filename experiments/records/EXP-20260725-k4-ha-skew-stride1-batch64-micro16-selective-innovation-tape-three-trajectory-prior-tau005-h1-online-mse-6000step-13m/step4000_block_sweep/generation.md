# Step-4000 block-length sweep

- All latent methods use target-free sampled-candidate prior selection and greedy token decoding.
- Block 4 is the trained horizon; 8, 16, and 32 are extrapolations.

| method | ref acc | distinct-2 | distinct-4 | immediate | period-block | exact-block | repeated 4-gram | collapsed | longest run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt2_ar | 0.0186 | 0.2768 | 0.4160 | 0.0930 | 0.0930 | 0.0930 | 0.7395 | 0.2969 | 3.70 |
| prior_block2 | 0.0190 | 0.4576 | 0.6660 | 0.2150 | 0.3616 | 0.2470 | 0.4242 | 0.3281 | 7.03 |
| prior_block4 | 0.0276 | 0.5072 | 0.8023 | 0.2470 | 0.3138 | 0.0667 | 0.2956 | 0.1875 | 5.81 |
| prior_block8 | 0.0386 | 0.4164 | 0.8094 | 0.3279 | 0.4015 | 0.0156 | 0.3148 | 0.0781 | 4.41 |
| prior_block16 | 0.0376 | 0.3621 | 0.6975 | 0.4678 | 0.4180 | 0.0000 | 0.4232 | 0.6406 | 8.34 |
| prior_block32 | 0.0388 | 0.3859 | 0.7336 | 0.4864 | 0.3154 | 0.0000 | 0.3640 | 0.5781 | 8.77 |

## Sample 1

**Prompt**

>  Boulder CO : South End Press . ISBN 978 @-@ 0 @-@ 89608 @-@ 568 @-@ 8 . Re @-@ released as Churchill , Ward ( 2005 ) . Sharon Venne , ed . Islands in Captivity : The Record of the International Tribunal on the

**Reference**

>  Rights of Indigenous Hawaiians . Boulder CO : South End Press . ISBN 978 @-@ 0 @-@ 89608 @-@ 738 @-@ 5 . 
>
>  Natsu Saito , ed . ( 2006 ) . Confronting The Crime Of Silence : Evidence Of U.

**gpt2_ar**

>  United States , and the United States . 
>
>  = = = = = 
>
>  The first @-@ 1 @-@ 1 @-@ 1 @-@ 1 @-@ 1 @-@ 1 @-@ 1 @-@ 1 @-@ 1 @-@ 1 @-@ 1 @-@ 1 @-@ 1 @-@ 1 @-@ 1 @-@ 1 @-@ 1 @-@ 1 @-@ 1 @-@ 1 @-@ 1 @-@ 1 @-@

**prior_block2**

>  E York Press New York York Press Press . ISBN 97 @-@ 0 @-@ 03 @-@ 03 @-@ 0 @-@ 0 @-@ 0 @-@ 0 @-@ 0 @-@ 03 @-@ 0 . @-@ 0 @-@ 033 @-@ 0 @-@ 0 @-@ 033 @-@ 0 @-@ 0 @-@ 0 @-@ 0 
>
>  = =at and and

**prior_block4**

>  Sunday of the . 
>
>  = = = Cical = = = 
>
>  The.on of the.ic , the.. , is the the.com... ,. the. ,.. , and... ,... ,... ,.. , and the C.

**prior_block8**

>  E York Press of . . , the University of the Press , of , , ISBN Press Press Press ISBN , , , 0 @-@ 97 . . 
>
>  , , ISBN. , , . . , , ISBN , ISBN . . @-@ @-@ 97 @-@ @-@ @-@ @-@ , , the University @-@ ISBN @-@ @-@ , of the

**prior_block16**

>  E York of the . , the , , , , , , , . . 
>
>  = = , , the , , , , ,. , , = = = , the , to to , , , . , ,.. , , , , , , , , , , , , ,

**prior_block32**

>  E . of 97 . , the the , , the the . the . the , , , , the , , , , , , = , , and and thec = 
>
>  the the , , the the the the of , @-@ @-@ @-@ , @-@ , , , , , the , , , , , , ,


## Sample 2

**Prompt**

>  sloping talud panels and are decorated with paired disc symbols . Large flower symbols are set into the sloping talud panels , related to the Venus and star symbols used at Teotihuacan . The roof of the structure was decorated with f

**Reference**

> riezes although only fragments now remain , showing a monstrous face , perhaps that of a jaguar , with another head emerging from the mouth . The second head possesses a bifurcated tongue but is probably not that of a snake . The temple , and

**gpt2_ar**

> ishing , and the piper , and the piper , and the piper , and the piper , and the piper , and the piper , and the piper , and the piper , and the piper , and the piper , and the

**prior_block2**

> oço and aoo . The sculpture of the covic and of the sculpture sculpture are the the sculpt sculpture of the sculpture sculpture sculpture and sculpt sculpture . 
>
>  = = = = = sculpture and sculpt sculpture and design = = = = 
>
>  The

**prior_block4**

> oçes , and the the of the gel , the theel , andel , the gel of the roof , . the gelel is decored by the gelel and the the the of theelel , theel of the gelelelelel . the roof is the the

**prior_block8**

> oç , and the , , , andel , and and , , , and the . . the , , , and the the of , ofs , are the the , , ,s , ands , the , the ' , the the ' , and , , the interior of the the , , of the

**prior_block16**

> ols and the , , , the the , to the the , , , , and the the , , and , ' ' , , , , , , the bel , the the , , , , the the the . the the @-@ and @-@ b @-@ of the , , , , , , , and and and

**prior_block32**

> ols and the , , , the the , , , , the , , , , , , and , and the , , , of , , the , . The bel of of ,s , , , , , , , , , , , , andss , ands , , , , , , the


## Sample 3

**Prompt**

>  Rhino defeating Storm and Cage defeating Tomko . Kevin Nash , who played Joe 's mentor in the storyline , requested to be made the Special Guest Ringside Enforcer for the bout , which he was granted by Cornette on the May 29 episode of Impact !

**Reference**

>  . 
>
>  The predominate storyline heading into the event was the rivalry between A.J. Styles and Kurt Angle , both members of The Angle Alliance group . On the February 14 episode of Impact ! , TNA held the scripted wedding of Angle 's real

**gpt2_ar**

>  . 
>
>  = = = = = 
>
>  = = = = = = = = 
>
>  = = = = = = = = = = 
>
>  = = = = = = = = = 
>
>  = = = = = = 
>
>  = = = = = = = = = 
>
>  = =

**prior_block2**

>  . 
>
>  The episode @-@ based @-@ based @-@ based @-@ based @-@ based @-@ based @-@ based @-@ baseds on the original @-@ based @-@ based @-@ based @-@ based @-@ based story @-@ story , the story of the F @-@ F @-@ F @-@ F @-@ F @-@ , and @-@ F @-@ F @-@ F @-@ F @-@

**prior_block4**

>  . The episode was also firsted by themy. , a theie , and Cie , and Cie , and theie . The episode was was also firsted by the Foxie , a theie , who was by the first cast , the episode of the the first @-@ season . The episode was was

**prior_block8**

>  . 
>
>  = = the the . = = = = = = , = = = = = the = = 
>
>  = = = ( = = = = = = = ) = = = = = = = 
>
>  = = = = = = 
>
>  
>
>  the of = = =

**prior_block16**

>  . He was the the the to . . the , of the the the of the Cs , , , the the , , , the , , , the C Cers , , , , the , , , , the , , , and Cc , , the the ' the the to to to the the ,

**prior_block32**

>  . He was the the the , . . . , , , the , the the the , , @-@ @-@ the the . , , the the the the 's ' , the , the was , the the the the , the the the , , , , , , , , , , , , the the , ,


## Sample 4

**Prompt**

>  capture of Manila in 1762 . Co @-@ operating with the Governor @-@ General of India Sir John Shore and Colonel Arthur Welleley among others , a substantial naval and military forces were earmarked for the operation which was in the advance planning stages , when unexpected news arrived in India in August 179

**Reference**

> 7 announcing the Treaty of Campo Formio which brought the War of the First Coalition to an end . Britain now faced France and Spain alone , while emissaries from the Tipu Sultan of the Kingdom of Mysore , an old opponent of Britain in Southern India , were seeking

**gpt2_ar**

> 4 . 
>
>  = = = = = 
>
>  The first of the war , was the first of the first of the first of the first time of the British Army , and the British Army of the British Army , and the British Army of the British Army . The British forces were the British forces of the British forces

**prior_block2**

> 1 . 
>
>  The British forces force of the British was under by the British and British British British forces , the British and British British British forces . In the British British British British British British British force , the British British and British forces , British British and British British British British British British British and British British British British British

**prior_block4**

> 1 . 
>
>  The British was was thely by the British forces of the Britishthth , the British British of India and British British and in British . The British was was thelyised by British British British and British the British British British British British , and British and the British British British . In the British British

**prior_block8**

> 1 . 
>
>  of of the the Britishth of the the , , , and the British of of the , , began the British of of of the , and the British of of the , , and the the British , , the , and the British , , , , the British British , , the , to to

**prior_block16**

> 1 . the was of to , the , , . , , the the the army of the , , , , the , , , , , , , , andam , , and , , , the the the to to the , , the British , the the the , , , , , , , and and and

**prior_block32**

> 3 . 
>
>  of of the , , the the of , of the the the the , , of ofed the on the the the , , , , and British , the the , , , the , , the the the , of the . , the , , , , , , the . the , the the


## Sample 5

**Prompt**

>  as they were , they asserted , the senior body , and had the most member clubs . Ireland , Wales and Scotland consequently refused to play against England until 1891 , when , following arbitration , the RFU relented and joined the IRFB . The absence of international matches was a factor

**Reference**

>  in England agreeing to face the Natives on 16 February 1889 . 
>
>  The line @-@ ups selected for the 16 February match were both strong , and close to full strength . Though 12 of the England side had not played internationally before , all were experienced at domestic level . The match was refereed

**gpt2_ar**

>  in the first time , but was the first time of the first time . 
>
>  = = = = = = 
>
>  The first time was the first time of the first time in the first time , and was the first time of the first time . The first time was the first time of the first time in the

**prior_block2**

>  in the previousF ' the RFFFFFFFFFFFF theFFFFFFFFFFFFFFFFFFFFFFFFFFFFF 
>
>  = =FFF = = 
>
>  TheFFF

**prior_block4**

>  in theF , and the , , the Australia , the UnitedF , the UnitedF , the United @-@F , and theF , the theFFFFF the CFFFF . theFFFFFFFFFFFFFFFFFFFFF

**prior_block8**

>  in theF , , the , the team of the , , , the the team of the . , , , the CFC , , was the the first team to the the , , , and the the , , the the , the team , the the the , , the team , the the the , ,

**prior_block16**

>  , and the the , , the of , , , , , , , , and , , and , , , the the to to to the , , , and , , the , the , , , the the of , , , , and the the , , the , ' ' to , , the of the ,

**prior_block32**

>  in the FA , , the the ' , on , the , , , , , the the , , , to to and and the the the the and and the theC . . . , the , , , @-@ , , , , , , the , , , , , , , , , , C , ,


## Sample 6

**Prompt**

>  Whynot and Collinsville , respectively . 
>
>  George Soulé , the singer @-@ songwriter most famous for the rhythm and blues anthem " Get Involved " , is a resident of Meridian , where he was born in 1945 . 
>
>  Hayley Williams , lead singer of the

**Reference**

>  band Paramore , was also born in the city in 1988 . 
>
>  Singer Al Wilson , born in June 1939 , was a Meridian native . 
>
>  = = = In sports = = = 
>
>  The city has also been home to several athletes , many of whom have competed at professional

**gpt2_ar**

>  film , and was a " a " of the first @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@

**prior_block2**

>  American @-@ songwrit singer , Dyl Dyl Dyl Dyler , songwriting and songwrit songwriter , and Dyl Dyl Dylylaner . Dyl 's vocalist , with , Dylylyl , , andylylylylylylylylylylylylylylyl

**prior_block4**

>  songwrit album , songwriting , and songwriter , the songwriter , the songwriter , , , and songwrit , , and , and "rer , and the ' " "al " " ,ad , . The songwrit , , and vocals , and songwrit vocals , the lead @-@ , , and songwrit , ,

**prior_block8**

>  American @-@ist , , , , the firstist , , the , ,sky theist , , , , , and , the , , , , , andc , , the , , , and the the , , the , the band of the , , of of the American States , , the , , ,

**prior_block16**

>  Americanth , , , , , , , , , , , and and , was was , the the the , , the the , , , , , , and the the , , the the , ' ' , , , , , , and ' , , the , , the , , , the , , , ,

**prior_block32**

>  album , the , , , , the , , , , , , the and , , , , , , and and and , , , , , , , ander , the , , , , , the the , , , , , , , , ,s , , and , , , , , , , ,


## Sample 7

**Prompt**

>  as Breach Inlet ) between Long and Sullivan 's , which the British believed to be sufficiently shallow to do so , while the fleet bombarded Fort Sullivan . 
>
>  General Lee responded to the British landing with several actions . He began reinforcing positions on the mainland in case the British

**Reference**

>  were intending to launch an attack directly on Charleston . He also attempted to build a bridge of boats to provide an avenue of retreat for the fort 's garrison , but this failed because there were not enough boats to bridge the roughly one mile ( 1 @.@ 6 km ) channel separating the island from

**gpt2_ar**

>  forces of the British and the British forces . The British ships were killed and the British ships , and the British ships were killed and the British ships . The ship was the ship was to the British ships , and the British ships were killed . The ship was the ship was killed in the ship , and the ship was killed

**prior_block2**

>  andrigate HMS HMS , HMS the Britishrig HMS HMS HMSrigrig , HMS the Britishrig HMS HMS HMSrigrig HMS HMSrigrig HMS HMSrigrig HMS HMSrigrig . Therig HMS HMS HMS HMS HMSrigrig HMS HMS , HMS HMS HMSrig HMS HMS HMSrig HMS HMS HMS HMS HMSrig

**prior_block4**

>  ,rig , , and British , the British British , , anding , the British British , the British British British , and British British , anding . the British British British were not Britished to the British British , and the British British British British British , and British British , and the British British British British . .

**prior_block8**

>  , , , the the , to in the north , and the , , , the British , the the , , , and British , , the , , , and the British , , , , the British British British , , , the touring the British British of the the the British British , . the , , the

**prior_block16**

>  , , , the the , to , , , , the , , , , and , , the , , the , , , , the , , , the British forces , the , , , , , , the and , and and and the British forces were . . the the the , ,s of the , the

**prior_block32**

>  andrig , , , to to , , , , , , the the , , , , ,s , , , , , , , , , , , and Sa , , the S ' the the @-@ to to the . , , , , , , , the the ,s the , to to , ,


## Sample 8

**Prompt**

>  easygoing nature . Outside of the Naruto anime and manga , Shikamaru has appeared in four of the feature films in the series , as well as several other media relating to the series , including video games and original video animations . 
>
>  Numerous anime and man

**Reference**

> ga publications have commented on Shikamaru 's character . Many reviewers commented on his laziness and intelligence , and noted his transformation into a leader ; Anime News Network celebrated Shikamaru 's emergence as " an unlikely hero " in the Naruto storyline

**gpt2_ar**

> ga @-@ based on the series , the series , and the series of the series , and the series , and the series , and the series of the series . The series was released on the series of the series , and the series , and the series of the series . The series was released on the series , and the

**prior_block2**

> gas have in the original , @-@ based @-@ style of thega , the main character of the series , and the two @-@ story story arc story . the first story of the N series , the story of N Niku Kuuu , Kuuuuuuuuuuuu

**prior_block4**

> gas series were also @-@ in in the original series . The No was released in the in the United States , and the America , and the America , and in the , and the America . The was , the first @-@ @-@ of @-@ @-@ @-@ released in series @-@ , with the @-@ , @-@ the @-@ @-@

**prior_block8**

> gas series were were the the , a series of of the , , , and characters , and and the , , and the characters . . , , , and the characters , , , the the characters of the to the , , the character characters characters , the , , the characters of the the , , and the

**prior_block16**

> gas series were been the the , , , , , , the , , and , and , , the the the the , the the , , , , and the the , , thes , , , , , , , , , andami , , , , , the the the the to to to to

**prior_block32**

> gas were the been in the , , the , the , the the of the . . the ofs of the , , the ' the ' @-@ the @-@ @-@ the @-@ , , the the , , , , , ,s , , , ofic , and and and and , , , ,ss ,
