# Step-1000 matched GPT-2 AR sentence-pattern evidence

- Latent checkpoint step: 1000
- GPT-2 checkpoint step: 1000
- Validation examples: 64
- Prompt/generated tokens: 64/64
- Token decoding is greedy and target-free; any sampled latent tape is fixed by the registered evaluation seed.

## Aggregate metrics

| method | ref acc | distinct-2 | immediate | period-4 | phase excess | block repeat | repeated 4-gram coverage | collapsed | longest run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt2_ar | 0.0186 | 0.2768 | 0.0930 | 0.2557 | +0.1627 | 0.1167 | 0.7395 | 0.2969 | 3.70 |
| selective_clean_block4 | 0.0303 | 0.3110 | 0.3003 | 0.5727 | +0.2723 | 0.3281 | 0.6378 | 0.2812 | 6.48 |
| selective_prior_block4 | 0.0317 | 0.4988 | 0.1714 | 0.2529 | +0.0815 | 0.0260 | 0.2252 | 0.0469 | 3.58 |

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

>  United of of , and , , , and , , , and , , , and , , , and , , , and , , , and , , , and , , , and , , , and , , , and , , , and , , , and , , , and , , , and , , ,

**selective_prior_block4**

>  United ' of the United States States . The United of , which , , the United States , , and , , the United States of the United States , , and , the United States United States , and . the United States , , the United States of the United States States , and the United , the United States ,


## Sample 2

**Prompt**

>  sloping talud panels and are decorated with paired disc symbols . Large flower symbols are set into the sloping talud panels , related to the Venus and star symbols used at Teotihuacan . The roof of the structure was decorated with f

**Reference**

> riezes although only fragments now remain , showing a monstrous face , perhaps that of a jaguar , with another head emerging from the mouth . The second head possesses a bifurcated tongue but is probably not that of a snake . The temple , and

**gpt2_ar**

> ishing , and the piper , and the piper , and the piper , and the piper , and the piper , and the piper , and the piper , and the piper , and the piper , and the piper , and the

**selective_clean_block4**

> ased , , and the the , the the the of the wall @-@ , and the the the wall of of the wall . . the main of of the wall is are the main , of the main , , the main of the the main , , the main of of the main , , the main , of the

**selective_prior_block4**

> iters , the c @-@ c , and , , the mains of the ga , , and the the the wall of of the wall . The the main is of the wall of the , the the , the main wall , the main wall of the wall , the Thea @-@ , and the @-@ ,


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

>  " " , " " " " , the " " , the " " , the " " , the " " , the " " , the " " , the " " , the " " , the " " , the " " , the " " , the " " , the " " , the " "

**selective_prior_block4**

>  
>
>  
>
>  = = = = = = = = = = = 
>
>  The was written by the episode of directed , , and D , , andy , the episode was the episode of episode by the " episode episode . The episode , the episode episode of the episode episode episode , " episode , the


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

>  . . The The the , , the the the the British of of the British , , the British of the the British of the the British of the the British of the the British of of the British . . the British of the the British of the the British of the the British of the the British of the the

**selective_prior_block4**

>  . and . . The the was was theed by the Britishth Army of the Army , , which was the the British Army of of the British Army , the the , of the Army , the British of the , and the the of the Army . . 
>
>  = = = Battle of = = = 
>


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

>  in the , , and the the , and the the , and the the . 
>
>  , , the first , , and the , , and the , , and the the , the the the , the the the of the , , . The the , , and the , , and the the the first of of the

**selective_prior_block4**

>  of the , , and the the , the the , , and , the the first ' of the Cup , . the first was was the first ' of the Cup Cup , the club of the , and was , the first Cup Cup Cup . . . 
>
>  = = = history = = = = 
>
>  =


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

>  song singer , , and singer , , and singer , , and singer , , and singer , , and singer , , and singer , , and singer , , and singer , , and singer , , and singer , , and singer , , and singer , , and singer , , and singer , , and singer , ,

**selective_prior_block4**

>  first singery , and , , the song of the the song @-@ist , and song , the song ' of the " song , . " song ' , " the song , " was to the song ' " the song " " " , was the the song song of " , " " the song " " "


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

>  British , the the British of of the British States , the British of of the British States , the British of of the British States of the British States of the British States of the British States of the British States of the British States of the British States of the British States of the British States of the British States of the

**selective_prior_block4**

>  British British , , the the , , the the British of the British Army , and the the the British of of the British , . the British of the the British British of of the British , the British of the of the British , the British British of , the the British of the British , , and the British of


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

> gas characters characters , the characters characters , the characters characters , the characters characters , the characters characters , the characters characters , and characters characters , 
>
>  characters characters , and characters characters , 
>
>  characters characters , and characters characters , 
>
>  characters characters , the characters characters , the characters characters , the characters characters , the characters characters

**selective_prior_block4**

>  @-@s , , a character , the game ' , the game ' , the game @-@ , , and character , the game ' , the game of the , and game , the game game , the game game , the game , the game , , the the game game , , and game game , and the game game


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

>  the south . . The the , , the road of the town wass , and the the the town of of of the town . . The the , , the town of the town towns , and the town town wass town town .s town townss town townss town townss town town

**selective_prior_block4**

>  the road . . The route , the new road was was completed to the the route line of the Uway , . The route was was theed to the new route of the U @-@ River . In was the , it was the the U @-@way of the route . the route route was the first route of the


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

> ing , " , and the the " first of " " .s " wass the " first of " " ,s " " ,s " " ,s " " , " " " , " " " , " " " , " " " , " " " , " " " , " " "

**selective_prior_block4**

>  of the ' , and the the of the ' ofs theel , , which it that the first @-@ ' of the @-@ @-@ . The ' wass , , , the firstal ' of the ' , , and was the the first ' of the first ' , the first ' of the first ' , the


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

>  of the one 's best ' of the album , , and it the 's best ' of the single ' " It ' ' " . ' ' " is the song " , the song " , the " " , the " " , the " " , the " " , the " " , the " "

**selective_prior_block4**

>  of one of the most ' thes most @-@ single of the song , the song ' of the " ' , " and it the the " song of " , ' " " , it the the song song of " " the " " , " " the song song " " , it the the song of " song


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

>  a as , the storm wass to be the the the storm of of . The , , the storm storm the storm ofs the storm ofs the storm ofs the storm .s the storm of of the storm , the the storm of of the storm , , the storm of of the storm . . the

**selective_prior_block4**

>  a as the , it was to the tropical storm of the United @-@ . . The storm of the storm was the storm , to the the storm coast of the United States , the storm storm of of the storm . the storm of the storm wased , the storm of the of the . the storm wased , the


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

> ted , the the first of of the original 's the original ofs the original 's the original .s the original wass the original of the the original 's the original 's the original 's the original 's the original 's the original 's the original 's the original ' . the

**selective_prior_block4**

> ted , , the first ' of the originalth , , and , , the song of the the original ' . the original @-@ ' of the song was the first song of the song ,s the song ' of the song ' , the song ' of the song , " the " ' " " " " " "


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

>  first time . . The the , , the first , , and first , , and first , , a first first , and first first , in first first . The second , , a first first , and first first , in first first , and first first , in first first , and first first , in first first ,

**selective_prior_block4**

>  team , . the team was the the team team of the team team , the first team , the first @-@ team , and team , the team , the the team team . the first team team was the first @-@ , and the the , the the team , and team , the first team , , and was to the


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

>  to the the . 
>
>  , , and , of that the was was the first of of the first ,s the first of of the United , , the first ' of the first ' , the first ' of the first ' , the first ' of the first 's the first ' of the first 's the

**selective_prior_block4**

>  tool , . The ' was , and that the , was was to the same ' of the same ' . the first of the was not , , the first was the , the the , , and it the the firstest of of the . . the first was the the first ' of the first ' , the


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

> cled of the Britishian . the British of of the British Navy of the British Navy of the British Navy of the British Navy of the British Navy of the Navy Navy and the British Navy of the Navy Navy Navy the ship Navy of the Navy Navy Navy . The ship ship of the ship was shiped the ship ship

**selective_prior_block4**

> cl theed of the . . the ship of the of the. Navy , the the ship , was ship of the ship and the ship was ship . the ship was the ship , and to the ship ship ship of the ship . ship was was the ship and ship . the ship ship was the ship ship ship ,
