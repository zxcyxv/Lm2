# Step-1000 AR versus block-4 generation

- Checkpoint: `outputs/experiments/EXP-20260725-k4-ha-skew-stride1-batch64-micro16-initial-condition-three-trajectory-prior-tau005-h1-online-mse-13m/step1000.pt`
- Validation examples: 64
- Prompt/generated tokens: 64/64
- Selection is greedy and target-free.

## Aggregate metrics

| method | ref acc | distinct-2 | immediate repeat | period-4 repeat | block repeat | collapsed | longest run | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean_ar | 0.0176 | 0.1791 | 0.5801 | 0.6435 | 0.5333 | 0.7031 | 33.92 | 2.68 |
| clean_block4 | 0.0168 | 0.2773 | 0.5511 | 0.6617 | 0.4583 | 0.5781 | 22.16 | 0.57 |
| prior_ar | 0.0134 | 0.2634 | 0.3867 | 0.3852 | 0.3333 | 0.4688 | 22.16 | 2.51 |
| prior_block4 | 0.0254 | 0.3380 | 0.2299 | 0.4951 | 0.2687 | 0.1719 | 4.19 | 0.64 |

## Prior traces

- prior_ar: `{'encoder_calls_per_sample': 64, 'prior_max': 0.42960554105229676, 'prior_entropy': 1.0657969270832837, 'branch_switch_rate': 0.30952380952380953, 'branch_usage': [0.192626953125, 0.31982421875, 0.487548828125]}`
- prior_block4: `{'encoder_calls_per_sample': 16, 'prior_max': 0.41943349363282323, 'prior_entropy': 1.0680544059723616, 'branch_switch_rate': 0.3416666666666667, 'branch_usage': [0.3193359375, 0.416015625, 0.2646484375]}`

## Sample 1

**Prompt**

>  Boulder CO : South End Press . ISBN 978 @-@ 0 @-@ 89608 @-@ 568 @-@ 8 . Re @-@ released as Churchill , Ward ( 2005 ) . Sharon Venne , ed . Islands in Captivity : The Record of the International Tribunal on the

**Reference**

>  Rights of Indigenous Hawaiians . Boulder CO : South End Press . ISBN 978 @-@ 0 @-@ 89608 @-@ 738 @-@ 5 . 
>
>  Natsu Saito , ed . ( 2006 ) . Confronting The Crime Of Silence : Evidence Of U.

**clean_ar**

>  UK , and the United States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States

**clean_block4**

>  UK . . The The , , the , , , and , , , and , , , and , , , and , , , and , , , and , , , and , , , and , , , and , , , and , , , and , , , and , , , and , , ,

**prior_ar**

>  Ped Ped . The album was released on the album in the United States on the album , and the album was released on the album , and the album in the United States on the album , and the album was released in the United States on the album , and the album in the United States , and the album

**prior_block4**

>  P .s The . ' , the the ' , the the @-@ , and the @-@ , the the @-@ , the the @-@ , and the . , the the @-@ , the the @-@ , the the @-@ , the the @-@ , and the @-@ , the the . , the the @-@ , the the @-@ ,

## Sample 2

**Prompt**

>  sloping talud panels and are decorated with paired disc symbols . Large flower symbols are set into the sloping talud panels , related to the Venus and star symbols used at Teotihuacan . The roof of the structure was decorated with f

**Reference**

> riezes although only fragments now remain , showing a monstrous face , perhaps that of a jaguar , with another head emerging from the mouth . The second head possesses a bifurcated tongue but is probably not that of a snake . The temple , and

**clean_ar**

> apelelic , and the walled the walled the wall , and the walled the wall , the wall , and the wall is the main wall , and the wall is the main wall , and the wall is the main wall , and the main wall is the main main mains . The main main main

**clean_block4**

> apy and and a @-@ @-@ , a the @-@ of @-@ @-@ @-@ . The The is the first of of of the @-@ @-@ , is the the of the @-@ @-@ , is the the of the @-@ @-@ , and the the of the @-@ @-@ . The is is is a in of of the @-@ @-@ ,

**prior_ar**

> elelel . The first of the celelel 's first @-@ @-@ delelelel 's celelel . The melelel 's first @-@ delelelel 's celelel 's melelel . The melelel 's first

**prior_block4**

> elel , and the the @-@ of p . , , the the @-@ , and the @-@ , the the @-@ , the the @-@ , and the @-@ , the the . , the the @-@ , the the @-@ , the the @-@ , the the @-@ , the the @-@ , the the @-@ , the the @-@ ,

## Sample 3

**Prompt**

>  Rhino defeating Storm and Cage defeating Tomko . Kevin Nash , who played Joe 's mentor in the storyline , requested to be made the Special Guest Ringside Enforcer for the bout , which he was granted by Cornette on the May 29 episode of Impact !

**Reference**

>  . 
>
>  The predominate storyline heading into the event was the rivalry between A.J. Styles and Kurt Angle , both members of The Angle Alliance group . On the February 14 episode of Impact ! , TNA held the scripted wedding of Angle 's real

**clean_ar**

>  " The The The episode " The The episode " The episode " The episode episode episode " " " " " " " " " " " " " " " " " " The episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode

**clean_block4**

>  " The The was released by by episode The The The The episode . The The episode episode was episode by the episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode

**prior_ar**

>  " . The episode was " a " fantasy " by The Fernlll . The episode was written by The Fr. Fr..................................

**prior_block4**

>  " Thes of The episode " The the episode " The the episode " " the episode " " the episode " " the episode " " the episode " " the episode " " the episode " " the episode " " the episode " " the episode " " the episode " " the episode " " the episode "

## Sample 4

**Prompt**

>  capture of Manila in 1762 . Co @-@ operating with the Governor @-@ General of India Sir John Shore and Colonel Arthur Welleley among others , a substantial naval and military forces were earmarked for the operation which was in the advance planning stages , when unexpected news arrived in India in August 179

**Reference**

> 7 announcing the Treaty of Campo Formio which brought the War of the First Coalition to an end . Britain now faced France and Spain alone , while emissaries from the Tipu Sultan of the Kingdom of Mysore , an old opponent of Britain in Southern India , were seeking

**clean_ar**

> 7 , and the war was not to be in the war , and the war was not not to be in the war . The war was not not to be in the war , but the war was not not to be in the war , but the war was not not to be in the war war , and the war

**clean_block4**

> 7 , the was transferred to the the Battle . of of the war , , the was was the first of of the war War of the war War of the war War of the war War of the war War of the war War of the war War of the war War of war war war war war , war war war

**prior_ar**

> 7 , and the British Army of the Army of the Army of Ragagag . The British Army of the Army of the Army of the Army of the Army of Awwwwwwwwwwwwwwwwwwwwwwwwwwwww

**prior_block4**

> 7 , the the U ' of of the United . . The the was the first of the the war of the the war , the the government of the the United of the the United of the the U . of of the war was the first of the the war of the the war of the the war of the the

## Sample 5

**Prompt**

>  as they were , they asserted , the senior body , and had the most member clubs . Ireland , Wales and Scotland consequently refused to play against England until 1891 , when , following arbitration , the RFU relented and joined the IRFB . The absence of international matches was a factor

**Reference**

>  in England agreeing to face the Natives on 16 February 1889 . 
>
>  The line @-@ ups selected for the 16 February match were both strong , and close to full strength . Though 12 of the England side had not played internationally before , all were experienced at domestic level . The match was refereed

**clean_ar**

>  in the first round , but the club was not not to be in the club , but the club was not not to be in the club , but not not not any any club to be not not to be not not to be not not to be not not to be not not to be not not to be not not

**clean_block4**

>  in the the , and the was the first of of the club . , the club was was the first club of the club League Cup the club Cup Cup club , in club club , club club club , club club club , club club club , club club club , club club club , club club club , club club club

**prior_ar**

>  in the first round , but the team was not to be in the first round of the team in the season , but was not not to be the first in the season of the season , but was not not to be the first in the season of the season , but was not not to be the first season in the

**prior_block4**

>  in the . , he was the the first of of the team , , the team was the the team of of the team . . In was , , he was the the team team of the team , , the team team the the team team the the team team the the team team the the team team team .

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

**clean_ar**

>  singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer

**clean_block4**

>  singer . , , Madonna singer , , Madonna singer , , Madonna singer , , Madonna singer , , Madonna singer , , Madonna singer , , Madonna Madonna , , Madonna Madonna , , Madonna Madonna Madonna , Madonna Madonna Madonna Madonna , Madonna Madonna Madonna , Madonna Madonna Madonna , Madonna Madonna Madonna , Madonna Madonna Madonna Madonna Madonna Madonna Madonna

**prior_ar**

>  songwriters , songwriting , songwriting , songwriting , songwriting , songwriting , songwriting , songwriting , songwriting , songwriting , songwriting , songwriting , songwriting , songwriting , . " " Lou " . The song was a song by the singer singer , and a singer . She

**prior_block4**

>  songwrit .s , she was , the song of the the song 's , and the singer , was was her her studio , her her album , her her album , her her album , her her album , her her album , her her album , her her album , her her album , her her album , her her

## Sample 7

**Prompt**

>  as Breach Inlet ) between Long and Sullivan 's , which the British believed to be sufficiently shallow to do so , while the fleet bombarded Fort Sullivan . 
>
>  General Lee responded to the British landing with several actions . He began reinforcing positions on the mainland in case the British

**Reference**

>  were intending to launch an attack directly on Charleston . He also attempted to build a bridge of boats to provide an avenue of retreat for the fort 's garrison , but this failed because there were not enough boats to bridge the roughly one mile ( 1 @.@ 6 km ) channel separating the island from

**clean_ar**

>  Army , and the British Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army

**clean_block4**

>  Army of the the British Army of the Army Army Army the Army Army of the Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army

**prior_ar**

>  . The British ships was a British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British

**prior_block4**

>  . . , the British was the the British of of the British , the the British of the the British of the the British of the the British . of of the ships of the British , the the British of the the British of the the British of the the British of the the British of the the British of the the

## Sample 8

**Prompt**

>  easygoing nature . Outside of the Naruto anime and manga , Shikamaru has appeared in four of the feature films in the series , as well as several other media relating to the series , including video games and original video animations . 
>
>  Numerous anime and man

**Reference**

> ga publications have commented on Shikamaru 's character . Many reviewers commented on his laziness and intelligence , and noted his transformation into a leader ; Anime News Network celebrated Shikamaru 's emergence as " an unlikely hero " in the Naruto storyline

**clean_ar**

>  video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video

**clean_block4**

>  video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video

**prior_ar**

>  @-@ video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video

**prior_block4**

>  @-@ the video , the the video of the video video , and the video , and the video , and the video , and the video , and the video , and the video , and the video , and the video , and the video , and the video , and the video , and the video , video . , video
