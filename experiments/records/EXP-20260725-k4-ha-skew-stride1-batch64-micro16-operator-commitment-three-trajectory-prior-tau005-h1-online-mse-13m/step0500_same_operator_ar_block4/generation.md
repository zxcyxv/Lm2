# Step-500 same-operator AR versus block-4

- Checkpoint: `outputs/experiments/EXP-20260725-k4-ha-skew-stride1-batch64-micro16-operator-commitment-three-trajectory-prior-tau005-h1-online-mse-13m/step0500.pt`
- Validation examples: 64
- Prompt/generated tokens: 64/64
- Greedy decoding; one retained operator per four-token block.

## Agreement

- Same-prefix one-block token agreement: `0.4141`
- Same-prefix horizon agreement: h1=1.0000, h2=0.3438, h3=0.1562, h4=0.1562
- Full free-running token agreement: `0.1245`

## Aggregate metrics

| method | ref acc | distinct-2 | immediate repeat | period-4 repeat | collapsed | longest run | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|
| same_operator_ar | 0.0193 | 0.2904 | 0.4397 | 0.4839 | 0.6875 | 19.88 | 15.47 |
| same_operator_block4 | 0.0337 | 0.3311 | 0.2458 | 0.4542 | 0.1406 | 3.22 | 10.08 |

## Traces

- same_operator_ar: `{'encoder_calls_per_sample': 64.0, 'prior_max': 0.3996321496088058, 'prior_entropy': 1.07727340888232, 'branch_usage': [0.431640625, 0.1748046875, 0.3935546875]}`
- same_operator_block4: `{'encoder_calls_per_sample': 16.0, 'prior_max': 0.3971674700733274, 'prior_entropy': 1.0795249305665493, 'branch_usage': [0.2275390625, 0.1142578125, 0.658203125]}`

## Sample 1

**Prompt**

>  Boulder CO : South End Press . ISBN 978 @-@ 0 @-@ 89608 @-@ 568 @-@ 8 . Re @-@ released as Churchill , Ward ( 2005 ) . Sharon Venne , ed . Islands in Captivity : The Record of the International Tribunal on the

**Reference**

>  Rights of Indigenous Hawaiians . Boulder CO : South End Press . ISBN 978 @-@ 0 @-@ 89608 @-@ 738 @-@ 5 . 
>
>  Natsu Saito , ed . ( 2006 ) . Confronting The Crime Of Silence : Evidence Of U.

**same_operator_ar**

>  SSSS . A ' 's first @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ RAAA @-@ @-@ @-@ @-@ RAAA @-@ @-@ @-@ @-@ RAAA @-@ @-@ @-@ @-@ RAAA @-@ @-@ @-@ @-@

**same_operator_block4**

>  S . of of the @-@ ' , the the ' of the @-@ ' of the @-@ ' , the the ' of the @-@ ' of the @-@ . , the was of the first of of the first of ' @-@ @-@ @-@ . @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@


## Sample 2

**Prompt**

>  sloping talud panels and are decorated with paired disc symbols . Large flower symbols are set into the sloping talud panels , related to the Venus and star symbols used at Teotihuacan . The roof of the structure was decorated with f

**Reference**

> riezes although only fragments now remain , showing a monstrous face , perhaps that of a jaguar , with another head emerging from the mouth . The second head possesses a bifurcated tongue but is probably not that of a snake . The temple , and

**same_operator_ar**

> ites , and the first is also also also also also also also also also also also also also , and and the camic . The other other other , the species is also also also also also also also also also also also also also , and and the first first , and and the other . A pam

**same_operator_block4**

> it , , the most of of the f . , , the is of the species of the the species of the the ca , of the fam , and is the the c . . of the cam is also to the the species of the the ca of of the cam , and theam ,


## Sample 3

**Prompt**

>  Rhino defeating Storm and Cage defeating Tomko . Kevin Nash , who played Joe 's mentor in the storyline , requested to be made the Special Guest Ringside Enforcer for the bout , which he was granted by Cornette on the May 29 episode of Impact !

**Reference**

>  . 
>
>  The predominate storyline heading into the event was the rivalry between A.J. Styles and Kurt Angle , both members of The Angle Alliance group . On the February 14 episode of Impact ! , TNA held the scripted wedding of Angle 's real

**same_operator_ar**

>  , and he was a member of the first time , he was was a a member of the first @-@ @-@ @-@ @-@ ppppp . " " the first first of the first first @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ Rer . S

**same_operator_block4**

>  , , the the episode of the the episode of , the episode of the the episode , , the episode of the the episode , , the episode of the the episode , , the episode of the the episode , , the episode of the the episode , , the episode of the the episode , , the episode of the the


## Sample 4

**Prompt**

>  capture of Manila in 1762 . Co @-@ operating with the Governor @-@ General of India Sir John Shore and Colonel Arthur Welleley among others , a substantial naval and military forces were earmarked for the operation which was in the advance planning stages , when unexpected news arrived in India in August 179

**Reference**

> 7 announcing the Treaty of Campo Formio which brought the War of the First Coalition to an end . Britain now faced France and Spain alone , while emissaries from the Tipu Sultan of the Kingdom of Mysore , an old opponent of Britain in Southern India , were seeking

**same_operator_ar**

>  . In 167 , he was was a member of the war of the war of the Army of the British of the Rbbb . In 16th , he was was a member of the Army of the Army of the Army of the Army of the Rththth . In 16th , he was was

**same_operator_block4**

>  . . . , the was , the first of of the war of , the war of of the war , , the ship of of the war , of the war of of the R . . , the was of the command of the the war of , the ship of of the war , , the ship of of the


## Sample 5

**Prompt**

>  as they were , they asserted , the senior body , and had the most member clubs . Ireland , Wales and Scotland consequently refused to play against England until 1891 , when , following arbitration , the RFU relented and joined the IRFB . The absence of international matches was a factor

**Reference**

>  in England agreeing to face the Natives on 16 February 1889 . 
>
>  The line @-@ ups selected for the 16 February match were both strong , and close to full strength . Though 12 of the England side had not played internationally before , all were experienced at domestic level . The match was refereed

**same_operator_ar**

>  of the " " " of the " " , which was not not to be a " " , and " , and was not to be a " @-@ @-@ @-@ @-@ @-@ @-@ " " . " , he was was a member of the first first @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@

**same_operator_block4**

>  of the . of the @-@ ' , he was was the first of the the first of of the World . of of the the ' , he was was the first of the the first of of the R . of of the the ' , he he was the first of the the first of the the first of of the


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

**same_operator_ar**

>  episode was a first member of the film of the film of the film , the episode was released in the first first first @-@ @-@ @-@ @-@ Ric . Sie was released in the first first first @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ Rlll

**same_operator_block4**

>  episode of , the episode of of the episode episode , the episode of the the episode of , the episode of of the episode episode , the episode of the the episode episode , the episode of the the episode episode , the episode of the the episode episode , the episode episode the the episode of , the episode episode , the


## Sample 7

**Prompt**

>  as Breach Inlet ) between Long and Sullivan 's , which the British believed to be sufficiently shallow to do so , while the fleet bombarded Fort Sullivan . 
>
>  General Lee responded to the British landing with several actions . He began reinforcing positions on the mainland in case the British

**Reference**

>  were intending to launch an attack directly on Charleston . He also attempted to build a bridge of boats to provide an avenue of retreat for the fort 's garrison , but this failed because there were not enough boats to bridge the roughly one mile ( 1 @.@ 6 km ) channel separating the island from

**same_operator_ar**

>  Army , and the Army had been been to the command of the Army of the Rbbb . In August August , the Army was sent to the Army of the Army of the Army of the Rththth . Sbba was not not to the war , but he was to the German of the

**same_operator_block4**

>  Army , the the Army of of the Army , , the Army of of the R . . , the , of the Army of of the Army , , the Army of of the Army Army , the Army of of the R . . , the , of the Army of of the Army Army , the Army of of the


## Sample 8

**Prompt**

>  easygoing nature . Outside of the Naruto anime and manga , Shikamaru has appeared in four of the feature films in the series , as well as several other media relating to the series , including video games and original video animations . 
>
>  Numerous anime and man

**Reference**

> ga publications have commented on Shikamaru 's character . Many reviewers commented on his laziness and intelligence , and noted his transformation into a leader ; Anime News Network celebrated Shikamaru 's emergence as " an unlikely hero " in the Naruto storyline

**same_operator_ar**

>  is also also released by the first album of the album of the album , the album album album , the album was released in the album of the album album " , and the album , and the album , and the album , and the album , and the album , and the album , and the album , and the album

**same_operator_block4**

>  is in the the game of of the game 's , and , the the game game the the game game the the game game the the game game the the game game the the game game the game of the game . The game game game , , the game , , the game of the game , and game the game
