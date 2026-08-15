# Step-1000 same-operator AR versus block-4

- Checkpoint: `outputs/experiments/EXP-20260725-k4-ha-skew-stride1-batch64-micro16-operator-commitment-three-trajectory-prior-tau005-h1-online-mse-13m/step1000.pt`
- Validation examples: 64
- Prompt/generated tokens: 64/64
- Greedy decoding; one retained operator per four-token block.

## Agreement

- Same-prefix one-block token agreement: `0.3555`
- Same-prefix horizon agreement: h1=1.0000, h2=0.2344, h3=0.0938, h4=0.0938
- Full free-running token agreement: `0.1272`

## Aggregate metrics

| method | ref acc | distinct-2 | immediate repeat | period-4 repeat | collapsed | longest run | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|
| same_operator_ar | 0.0186 | 0.3614 | 0.2475 | 0.3237 | 0.3594 | 12.31 | 2.43 |
| same_operator_block4 | 0.0295 | 0.3147 | 0.2393 | 0.5156 | 0.2188 | 6.14 | 0.95 |

## Traces

- same_operator_ar: `{'encoder_calls_per_sample': 64.0, 'prior_max': 0.4165209918282926, 'prior_entropy': 1.0679108165204525, 'branch_usage': [0.4580078125, 0.271484375, 0.2705078125]}`
- same_operator_block4: `{'encoder_calls_per_sample': 16.0, 'prior_max': 0.41532262973487377, 'prior_entropy': 1.0693881437182426, 'branch_usage': [0.5009765625, 0.1103515625, 0.388671875]}`

## Sample 1

**Prompt**

>  Boulder CO : South End Press . ISBN 978 @-@ 0 @-@ 89608 @-@ 568 @-@ 8 . Re @-@ released as Churchill , Ward ( 2005 ) . Sharon Venne , ed . Islands in Captivity : The Record of the International Tribunal on the

**Reference**

>  Rights of Indigenous Hawaiians . Boulder CO : South End Press . ISBN 978 @-@ 0 @-@ 89608 @-@ 738 @-@ 5 . 
>
>  Natsu Saito , ed . ( 2006 ) . Confronting The Crime Of Silence : Evidence Of U.

**same_operator_ar**

>  ASS . The first of the American American American American American American American American American American American American American American American American American American American American American American American American American American American American American American American American American American American American American American American American American American American American American American American American American American American American American American American American

**same_operator_block4**

>  A .s . The ' ' , the , ' , and , ' , and , ' , and , . , and the ' , the the ' , and the ' , the the ' , and the ' , the the ' , and the ' , the the ' , and the ' , the the ' ,


## Sample 2

**Prompt**

>  sloping talud panels and are decorated with paired disc symbols . Large flower symbols are set into the sloping talud panels , related to the Venus and star symbols used at Teotihuacan . The roof of the structure was decorated with f

**Reference**

> riezes although only fragments now remain , showing a monstrous face , perhaps that of a jaguar , with another head emerging from the mouth . The second head possesses a bifurcated tongue but is probably not that of a snake . The temple , and

**same_operator_ar**

> amps and the cavavi of the stone of the town of the Bavi . The main floor was built in the first of the town of the town of the Bavi . The town of the town was built in the town of the town 's cuuuuuuuu

**same_operator_block4**

> amp and f , with the f of the m . , the thes , the thes , and thes , the thes , and thes , the the . , the thes , the thes , the thes , and thes , the the . , the thes , the thes ,


## Sample 3

**Prompt**

>  Rhino defeating Storm and Cage defeating Tomko . Kevin Nash , who played Joe 's mentor in the storyline , requested to be made the Special Guest Ringside Enforcer for the bout , which he was granted by Cornette on the May 29 episode of Impact !

**Reference**

>  . 
>
>  The predominate storyline heading into the event was the rivalry between A.J. Styles and Kurt Angle , both members of The Angle Alliance group . On the February 14 episode of Impact ! , TNA held the scripted wedding of Angle 's real

**same_operator_ar**

>  . 
>
>  = = = = The episode = = = 
>
>  The episode was written by The episode and directed by The Simpsons and The Fest Files , and directed by The Simpsons The The Fest . A episode episode was written by The Simpsons and directed by The Simpsons and The Simpsons , and The Simpsons episode

**same_operator_block4**

>  . . The = 
>
>  The the episode was the the episode episode the the episode episode the the episode of the the episode .s episode of the episode , and was the the episode episode the episode of the episode , and was the the episode episode the episode of the episode , and was the the episode episode the episode


## Sample 4

**Prompt**

>  capture of Manila in 1762 . Co @-@ operating with the Governor @-@ General of India Sir John Shore and Colonel Arthur Welleley among others , a substantial naval and military forces were earmarked for the operation which was in the advance planning stages , when unexpected news arrived in India in August 179

**Reference**

> 7 announcing the Treaty of Campo Formio which brought the War of the First Coalition to an end . Britain now faced France and Spain alone , while emissaries from the Tipu Sultan of the Kingdom of Mysore , an old opponent of Britain in Southern India , were seeking

**same_operator_ar**

>  . The ship 's first @-@ class was transferred to the USSS Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air Air

**same_operator_block4**

>  . . 
>
>  The the ' was a to the the government of the the government , the the government of the the U ' of of the United . , the was the the government of the the government , the the government of the the government , the the government of the the government . of of the government government was


## Sample 5

**Prompt**

>  as they were , they asserted , the senior body , and had the most member clubs . Ireland , Wales and Scotland consequently refused to play against England until 1891 , when , following arbitration , the RFU relented and joined the IRFB . The absence of international matches was a factor

**Reference**

>  in England agreeing to face the Natives on 16 February 1889 . 
>
>  The line @-@ ups selected for the 16 February match were both strong , and close to full strength . Though 12 of the England side had not played internationally before , all were experienced at domestic level . The match was refereed

**same_operator_ar**

>  of the club , and the club was the first first match in the match of the club , and the club was the first first match to the match of the club . A club was was held in the match of the match , but the match was not only in the match , and the match was not not to be

**same_operator_block4**

>  of the . , and was the the first of of the United , , the first of the the United , the the United of of the United , , the United of the the United States the the United States of the United . 
>
>  The the ' was first to the the United States the the United States the the


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

>  song , and her daughter of the daughter of the daughter of the daughter of the Lriin . She was born in the United States , and the daughter of the daughter of the Criri . She was born in the United States , and the daughter of the daughter of the Criri . She was born in

**same_operator_block4**

>  song , , the song of of the album .s , and was the the song of of the album 's , and the ' , " the ' , and was , the song of of the album .s . It song was , and song the the song of the the song , the the song of the the


## Sample 7

**Prompt**

>  as Breach Inlet ) between Long and Sullivan 's , which the British believed to be sufficiently shallow to do so , while the fleet bombarded Fort Sullivan . 
>
>  General Lee responded to the British landing with several actions . He began reinforcing positions on the mainland in case the British

**Reference**

>  were intending to launch an attack directly on Charleston . He also attempted to build a bridge of boats to provide an avenue of retreat for the fort 's garrison , but this failed because there were not enough boats to bridge the roughly one mile ( 1 @.@ 6 km ) channel separating the island from

**same_operator_ar**

>  . Affee was the first to be the British British Army Army of the British Army of the British Army of the Army of the British Army of the British Army of the British Army of the British British British Army . A A @-@ Army Army Army , the British Army of the British Army of the British Army

**same_operator_block4**

>  . . , the British was the the British of the the British , the the British of the the British , the the British of the the British of the the British , the the British of the the British of the the British of the the British . of of the British ' , the , the the British of the the


## Sample 8

**Prompt**

>  easygoing nature . Outside of the Naruto anime and manga , Shikamaru has appeared in four of the feature films in the series , as well as several other media relating to the series , including video games and original video animations . 
>
>  Numerous anime and man

**Reference**

> ga publications have commented on Shikamaru 's character . Many reviewers commented on his laziness and intelligence , and noted his transformation into a leader ; Anime News Network celebrated Shikamaru 's emergence as " an unlikely hero " in the Naruto storyline

**same_operator_ar**

>  @-@ video video video video video video video video video video video video video video video video video video video . Sakak was released on October 4 , and was released on October October , and released on the video video video . A ASSS was released on the video video . A ASSS was released

**same_operator_block4**

>  @-@ , the the film of the the film 's , and the ' , and the ' , and the ' , the the ' , and the ' , the the ' , and the ' , the the ' , and the ' , the the ' , and the ' , the the ' , and the ' ,
