# Step-4000 matched GPT-2 AR sentence-pattern evidence

- Latent checkpoint step: 4000
- GPT-2 checkpoint step: 1000
- Validation examples: 64
- Prompt/generated tokens: 64/64
- Token decoding is greedy and target-free; any sampled latent tape is fixed by the registered evaluation seed.

## Aggregate metrics

| method | ref acc | distinct-2 | immediate | period-4 | phase excess | block repeat | repeated 4-gram coverage | collapsed | longest run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt2_ar | 0.0186 | 0.2768 | 0.0930 | 0.2557 | +0.1627 | 0.1167 | 0.7395 | 0.2969 | 3.70 |
| selective_clean_block4 | 0.0291 | 0.4338 | 0.2470 | 0.4648 | +0.2178 | 0.1531 | 0.4255 | 0.1406 | 5.03 |
| selective_prior_block4 | 0.0259 | 0.5144 | 0.2200 | 0.3044 | +0.0844 | 0.0625 | 0.2779 | 0.2188 | 5.06 |

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

**selective_clean_block4**

>  E York of . ISBN ISBN Press Press @-@ 0 @-@ 0 @-@ 0 @-@ @-@ 0 @-@ 0 @-@ 0 @-@ 0 @-@ 0 @-@ 0 @-@ 0 @-@ 0 @-@ 0 @-@ 0 @-@ 0 @-@ 0 @-@ 0 @-@ 0 @-@ 0 @-@ 0 @-@ 0 @-@ 0 @-@ 0 @-@ 0 @-@ 0 @-@ 0 @-@ 0 @-@ 0 @-@

**selective_prior_block4**

>  Sunday of the . 
>
>  = = = Cical = = = 
>
>  The.on of the.ic , the.. , is the the.com... ,. the. ,.. , and... ,... ,... ,.. , and the C.


## Sample 2

**Prompt**

>  sloping talud panels and are decorated with paired disc symbols . Large flower symbols are set into the sloping talud panels , related to the Venus and star symbols used at Teotihuacan . The roof of the structure was decorated with f

**Reference**

> riezes although only fragments now remain , showing a monstrous face , perhaps that of a jaguar , with another head emerging from the mouth . The second head possesses a bifurcated tongue but is probably not that of a snake . The temple , and

**gpt2_ar**

> ishing , and the piper , and the piper , and the piper , and the piper , and the piper , and the piper , and the piper , and the piper , and the piper , and the piper , and the

**selective_clean_block4**

> oes and and as of . The the = = 
>
>  
>
>  = = = = = = = = = = 
>
>  the interior of of the interior is is the main of of the interior , , the interior of of the interior , and the interior of of the interior , . the interior is of the

**selective_prior_block4**

> oçes , and the the of the gel , the theel , andel , the gel of the roof , . the gelel is decored by the gelel and the the the of theelel , theel of the gelelelelel . the roof is the the


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

**selective_clean_block4**

>  . 
>
>  = = = = = = = = = = = The the , , and , , , and the , , was the of the first , , the first was of the first series , the first , of the first series , the first series of the series series , The series series , The

**selective_prior_block4**

>  . 
>
>  = = =al = = = 
>
>  The episode was the first of the the first season of the series , the the first season of of the television television , and the episode of the first season series The The F , The The episode , was the F , and the @-@ of @-@ the @-@ episode


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

**selective_clean_block4**

> 3 . 
>
>  The the = , the of and the Britishth of the Britishth , the Britishth of the Britishth , the Britishth of the Britishth , the British British and British forces , , the British British British British and , the the British British , British and , British British British , British British

**selective_prior_block4**

> 3 . 
>
>  = = = = Bala = = = = The Ba was one in the of the India of the India , the the British of India India . the India of the India India India ( India India , India India India ) , India India , and India India India , India the India


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

**selective_clean_block4**

>  in the the , and the the the FA team was the first time to the FA Nations . . The the the the FA Nations was the first FA of the FA Nations to the FA Nations , the FA Nations , the FA Nations , the FA Nations , the FA Nations , the FA Nations , the FA Nations , the

**selective_prior_block4**

>  in theF , and the team , the the team , the the team , the the team , the the team , the the team , the team , the C team team , and the team , the the team . The team team was alsoed to the team of the the C C team team , team , the


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

**selective_clean_block4**

>  album singer , songwriting , songwrit , and , , , and , , , and , , , and , , , and the , , and the , . He was , , and the , , and the , to be the the . He the , , and the , , the the , the music , , the

**selective_prior_block4**

>  American @-@ist , American @-@er , and songwrit , , and songwrit , and songwrit songwrit songwrit . The songwritson , and , songwrit , and , songwrit , and songwrit , , and songwrit , , and songwrit , , and , songwrit , and songwrit , , and songwrit , 
>
>  Theney , vocals , songwrit ,


## Sample 7

**Prompt**

>  as Breach Inlet ) between Long and Sullivan 's , which the British believed to be sufficiently shallow to do so , while the fleet bombarded Fort Sullivan . 
>
>  General Lee responded to the British landing with several actions . He began reinforcing positions on the mainland in case the British

**Reference**

>  were intending to launch an attack directly on Charleston . He also attempted to build a bridge of boats to provide an avenue of retreat for the fort 's garrison , but this failed because there were not enough boats to bridge the roughly one mile ( 1 @.@ 6 km ) channel separating the island from

**gpt2_ar**

>  forces of the British and the British forces . The British ships were killed and the British ships , and the British ships were killed and the British ships . The ship was the ship was to the British ships , and the British ships were killed . The ship was the ship was killed in the ship , and the ship was killed

**selective_clean_block4**

>  , , the the British force , the British British , the British British , the British British , the British British , British and , and British British . the the British British were the British British of British British , the British and , British British British , British British and , British British British , British British and , British British

**selective_prior_block4**

>  andrigrig , and British of the British of the , and the British British forces , the the British Britishrig , and the British of the British . . 
>
>  = , the British = = 
>
>  The British and the British of the British British were not to British British British British British , and British , the


## Sample 8

**Prompt**

>  easygoing nature . Outside of the Naruto anime and manga , Shikamaru has appeared in four of the feature films in the series , as well as several other media relating to the series , including video games and original video animations . 
>
>  Numerous anime and man

**Reference**

> ga publications have commented on Shikamaru 's character . Many reviewers commented on his laziness and intelligence , and noted his transformation into a leader ; Anime News Network celebrated Shikamaru 's emergence as " an unlikely hero " in the Naruto storyline

**gpt2_ar**

> ga @-@ based on the series , the series , and the series of the series , and the series , and the series , and the series of the series . The series was released on the series of the series , and the series , and the series of the series . The series was released on the series , and the

**selective_clean_block4**

> ga have been induct , in the United of of the United series , the series video of the series series , the series series of the series series series series , and series the series series series series . series series the series series series the series series series series , series series the series series series series , series series the

**selective_prior_block4**

> gas series were released in the in Japan and Japan . The N @-@ was released in the the Nth series in Japan . the , thega and and thegaga were also released in the late @-@ series , andgaga , thegaga , thegagagaga and thegaga the thega


## Sample 9

**Prompt**

>  go back to 1932 , when a parkway was proposed along the Delaware River between Trenton and Lambertville ; this proposal never materialized . Plans for a freeway were resurrected in the early 1950s to construct a road , the John Fitch Parkway , that was to run from

**Reference**

>  Hamilton Township to Interstate 95 in Ewing Township . This road was built between 1954 and 1957 from South Warren Street in Trenton to present @-@ day Interstate 95 in Ewing Township . Construction of this portion John Fitch Parkway took up much of Trenton ’ s available waterfront along the

**gpt2_ar**

>  the route . 
>
>  = = = = = 
>
>  The route was the first to the route of the route , and the route was the first to the route to the route . The route was the route to the route to the route to the route . The route was the route to the route to the route to

**selective_clean_block4**

>  the north @-@ to @-@ @-@ @-@ to @-@ @-@ @-@ to @-@ @-@ @-@ to @-@ @-@ @-@ to @-@ @-@ @-@ to @-@ @-@ @-@ . The the , , the @-@ @-@ , @-@ @-@ @-@ was @-@ @-@ @-@ , @-@ @-@ @-@ , @-@ @-@ @-@ , @-@ @-@ @-@ , @-@ @-@ @-@ , @-@ @-@ @-@ ,

**selective_prior_block4**

>  the Cton to the Cton . The new @-@ was noted , , and it was the first of the road to the road . The C @-@ was noted , the C @-@ road , and the the @-@ C @-@way was not @-@ed , and the @-@ was the @-@ed . The C @-@ was


## Sample 10

**Prompt**

>  publications are nearly all works of synthesis and reinterpretation , drawing upon studies by other scholars , not monographs describing new research based on primary sources . " The investigation also noted that " he has decided to publish largely in alternative presses or journals , not in the university press

**Reference**

> es or mainstream peer @-@ reviewed journals often favored by more conventional academics . " In addition to his academic writing , Churchill has written for several general readership magazines of political opinion . 
>
>  In 1986 , Churchill wrote an essay titled Pacifism as Path

**gpt2_ar**

>  , and the " The S.S.S. , and the " , and the " The S.S.S. , and the first of the " , and the " The New York Times , and the " The New York Times , and the " , and the " The New York Times , and "

**selective_clean_block4**

>  , and " , and the " , in the " of the " " . 
>
>  = = = = = = = = = = = 
>
>  
>
>  The the , , and the , , is the of the public , of the public , of the public , , the public of of the United , , the

**selective_prior_block4**

>  , , " the public of of the United States of , and , the the United States of of the United States " , the , the United States of of the United States , and United States States , Canada , the United States States , and , , United States , and the United States States . " = =


## Sample 11

**Prompt**

>  is " just Slayer being Slayer " . 
>
>  = = Music and structure = = 
>
>  " Angel of Death " is the longest track on the album Reign in Blood , spanning 4 minutes and 51 seconds , where the total duration of the album is 29 minutes . Additionally , it has one

**Reference**

>  of the most conventional song structures on the album , featuring prominent verses and choruses , where most tracks on the album eschew them . Hanneman and King deliver their ' intricate riffs ' , which offer the few hints of melody on the album according to PopMatters

**gpt2_ar**

>  of the song , the song 's first single , and the song 's song " The " , " , " , " , " , " , " , " , " , " , " , " , " , " , " , " , " , " , " , " , " , " , "

**selective_clean_block4**

>  of the songs , and the the the album is is the second @-@ of the album album album the same . . 
>
>  " = = Chart and and = = = 
>
>  " the Uniteds " debuted debuted debuted at number on on the Billboard Billboard 100 chart and number weekly the the , and it number the

**selective_prior_block4**

>  of the songs , including the songs , and songs , the album of songs , and the songs , and the songs , and songs , the songs of the songs and the songs . " = = = 
>
>  " " ' " " is a " " , " " " , " ' " , and " "


## Sample 12

**Prompt**

>  on September 9 , wind shear and dry air led to the remnants of Josephine deteriorating into an open wave . However , on September 10 , the remnants of Josephine redeveloped and global models picked up on the reformed system . Once more , the chance of regeneration was possible as

**Reference**

>  the remnants of Josephine headed towards the Bahamas . However , on September 14 , dry air and wind shear caused the remnants to dissipate entirely . 
>
>  = = Impact = = 
>
>  As Josephine passed to the south of the Cape Verde islands on September 2 , outer rain bands produced

**gpt2_ar**

>  a tropical storm , and the storm was not to the storm . 
>
>  = = = = = 
>
>  = = = = = = = = 
>
>  The storm was a tropical storm , and was a tropical storm , and was a tropical storm , and was a tropical storm . The storm was a tropical storm ,

**selective_clean_block4**

>  a result , the storm was of to be a the . On the = , the Hurricane Hurricane the storm Hurricane to the east , , the storm wasated to be west the the next day . the storm was to the north , and the storm was to to be the the the storm Hurricane of . 
>
>  = ,

**selective_prior_block4**

>  a tropical cyclone , the the storm became ext storm . the storm storm of the next @-@ season , the thea became therat tropical storm on the day . 
>
>  = = = Tropical Tropical Storm Eaa = = = 
>
>  The tropical disturb was a tropical tropical of the tropical @-@ on the @-@ @-@ ,


## Sample 13

**Prompt**

>  most part focussing on seal impression catalogues showcasing Hu 's carving work . 
>
>  The Ming dynasty had seen considerable advancement in the process of colour printing in China . At his studio , Hu Zhengyan experimented with various forms of woodblock prin

**Reference**

> ting , creating processes for producing multi @-@ coloured prints and embossed printed designs . As a result , he was able to produce some of China 's first printed publications in colour , using a block printing technique known as " assorted block printing " ( douban y

**gpt2_ar**

> ted by the py of the py , and the py of the py @-@ py @-@ py @-@ py @-@ py @-@ py @-@ py @-@ py @-@ py @-@ py @-@ py @-@ py @-@ py @-@ py @-@ py @-@ py @-@

**selective_clean_block4**

> ts , and , anding , , anding , , anding , , and theing , in the the the Chineses of the Chineses , the Chinese Chinese of the Chineses , and the Chinese Chinese Chinese Chinese Chinese . Chinese Chinese Chinese Chinese Chinese Chinese Chinese Chinese Chinese Chinese Chinese Chinese Chinese Chinese Chinese Chinese Chinese

**selective_prior_block4**

> ting , , , andb , , and the the , were well of the fang , . Theang ' ' " f ' ' ' ' ' , a ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' " , the 's ' ' '


## Sample 14

**Prompt**

>  for a driver to replace Larry Rice for his CART team at the following race at Watkins Glen International , so he gave Richmond a test at Willow Spring where he had previously set the student record . Santello hired Richmond , who then qualified 15th fastest for the

**Reference**

>  event and finished in eighth place , the best of his IndyCar career . Richmond raced in three more events that season . 
>
>  After crashing during the first day of qualifying for the 1980 Indianapolis 500 , Richmond nevertheless obtained the 19th starting position in the race . He worked

**gpt2_ar**

>  team . 
>
>  = = = = = 
>
>  = = = = = = = 
>
>  = = = = = = = = = = = = 
>
>  = = = = = = = = = 
>
>  The first @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year , the first

**selective_clean_block4**

>  first time , the team time was the first time @-@ the @-@ @-@ @-@ to @-@ @-@ @-@ team team to the the first @-@ . the first was the the first team @-@ the @-@ @-@ @-@ the @-@ @-@ @-@ @-@ team @-@ @-@ @-@ team @-@ @-@ @-@ team @-@ @-@ @-@ team @-@ @-@ @-@ team @-@ @-@ @-@

**selective_prior_block4**

>  race , and the fastth driver , and the team , and the team . 
>
>  = = = Ryle = = = 
>
>  R Ro was born to the , R R R , in , , the Rest , , and , , the Rest , , R R R , and R R ,


## Sample 15

**Prompt**

>  the crimes did take place , there is no evidence that Uzunoglu took part in it . He was also defending judge Pavel Nagy , who was indicted of accepting a bribe . The proceedings ended with Nagy being found insane and criminally not

**Reference**

>  liable . Jablonský also acted as a defense attorney in the case of a hairdresser of Czech VIPs indicted on charges of rape and torture . During the proceedings , the judge sent Jablonský to face the disc

**gpt2_ar**

>  to the same time . 
>
>  = = = = = 
>
>  The first time of the first time was a new role in the first time of the first time , and was the first time of the first time . The first time was the first time of the first time , and was the first time of the first

**selective_clean_block4**

>  to to the . In the , , the wasi that the was was to be , the , and the was to be the the . The the , , which the , , was the , , was the was , and the was of the be . . The the the , the the was , and the the the

**selective_prior_block4**

> iced to the N the P , but the the tools the the was unable . the . 
>
>  = = = P and = = = 
>
>  The P , the P P P , and P P , P P P , P P P , P P , , P P , Pir , , P


## Sample 16

**Prompt**

>  's first major engagements of the war were against Italian forces in the Mediterranean and North Africa . During 1940 the light cruiser HMAS Sydney and five elderly destroyers ( dubbed the " Scrap Iron Flotilla " by Nazi Propaganda Minister Joseph Goebbels — a title pro

**Reference**

> udly accepted by the ships ) took part in a series of operations as part of the British Mediterranean Fleet , and sank several Italian warships . The Army first saw action in January 1941 , when the 6th Division formed part of the Commonwealth forces during Operation Compass . The division assaulted and captured Bard

**gpt2_ar**

>  @-@ day , and the British ships were not to be the first @-@ day . 
>
>  = = = = = = 
>
>  The first time of the British ships were the first to be the first time of the first time , and the first two @-@ in the first time of the British ships were the first @-@ class

**selective_clean_block4**

> cl to the the German Navy of the United Navy ) the German Navy of the United Navy . the ship was the the first Navy of the Navy Navy to the United Navy the the Navy Navy of the United Navy , the Navy Navy Navy the Navy Navy Navy the Navy Navy of the Navy Navy Navy the Navy Navy . the

**selective_prior_block4**

> claimed of the Germanism , and the German of the war . the ship of the German Navy , were the first @-@ @-@ of @-@ @-@ @-@ @-@ war @-@ @-@ @-@ war @-@ @-@ @-@ war @-@ , the U @-@ @-@ @-@ U @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@
