# Step-3000 matched GPT-2 AR sentence-pattern evidence

- Latent checkpoint step: 3000
- GPT-2 checkpoint step: 1000
- Validation examples: 64
- Prompt/generated tokens: 64/64
- Token decoding is greedy and target-free; any sampled latent tape is fixed by the registered evaluation seed.

## Aggregate metrics

| method | ref acc | distinct-2 | immediate | period-4 | phase excess | block repeat | repeated 4-gram coverage | collapsed | longest run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt2_ar | 0.0186 | 0.2768 | 0.0930 | 0.2557 | +0.1627 | 0.1167 | 0.7395 | 0.2969 | 3.70 |
| selective_clean_block4 | 0.0249 | 0.3631 | 0.3589 | 0.5289 | +0.1700 | 0.2781 | 0.5379 | 0.2969 | 11.45 |
| selective_prior_block4 | 0.0232 | 0.4829 | 0.2780 | 0.3302 | +0.0522 | 0.0854 | 0.3363 | 0.2344 | 6.17 |

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

>  New York of . ISBN Press , ISBN @-@ @-@ . ISBN @-@ @-@ @-@ 0 @-@ @-@ @-@ 0 @-@ @-@ @-@ 0 @-@ @-@ @-@ 0 @-@ @-@ @-@ 0 @-@ @-@ @-@ 0 @-@ @-@ @-@ 0 @-@ @-@ @-@ 0 @-@ @-@ @-@ 0 @-@ @-@ @-@ 0 @-@ @-@ @-@ 0 @-@ @-@ @-@ 0 @-@ @-@ @-@

**selective_prior_block4**

>  Sunday of the . 
>
>  = = =als = = = 
>
>  The.c , The.. , ,... , , B. ,.. , , , B. ,.. , ,... ,... ,... ,. , , and...


## Sample 2

**Prompt**

>  sloping talud panels and are decorated with paired disc symbols . Large flower symbols are set into the sloping talud panels , related to the Venus and star symbols used at Teotihuacan . The roof of the structure was decorated with f

**Reference**

> riezes although only fragments now remain , showing a monstrous face , perhaps that of a jaguar , with another head emerging from the mouth . The second head possesses a bifurcated tongue but is probably not that of a snake . The temple , and

**gpt2_ar**

> ishing , and the piper , and the piper , and the piper , and the piper , and the piper , and the piper , and the piper , and the piper , and the piper , and the piper , and the

**selective_clean_block4**

> oldes and and decors of . The the the the ming is the firstest of the Roman , , the firstest of the Romanth , the Romanth of the Romanth , the Roman Roman of the Roman Roman , the Roman Roman of Roman Roman Roman Roman Roman Roman Roman Roman Roman Roman Roman Roman Roman

**selective_prior_block4**

> eleles and decor of the of the . the The g @-@ is decorel of the palel , the decor , of the gold @-@ the , is , , the pelel and cel . the pelel are decor decor , the goldelel , and gold , the gold , the the


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
>  the episode was was the first episode of episode of the series series series series series series series series series series series series series series series series series series series series series series series series series series series series series series series series series series series series series series series series series series series series series series series series series series

**selective_prior_block4**

>  . 
>
>  = = Plot = = 
>
>  = = = = @-@ = = 
>
>  = = = = = = = Sideta is a fictional of the series @-@ the series of the series , a @-@ @-@ , @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@


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

> 4 . In the earlyth , the Britishth of of the Britishth was the , to the British of of the Britishth , the British British of the British British , the British British of the British British , the British British of British British , and the British British of British British , , the British British British British

**selective_prior_block4**

> 4 . 
>
>  = = = = First = = = 
>
>  
>
>  the Britishthth of the Britishthth Fleet , the the British of the the Britishth of the Britishthth , and the British of the Britishth , the Britishth of the Britishth , the British British of the Britain , the


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

>  in the , the European team of the European , , the European Nations of the European Nations , the European Nations of the European Nations , the European Nations of the European Nations , the European Nations of the European Nations , the European Nations of the European Nations , the European Nations of the European Nations , the European Nations of the

**selective_prior_block4**

>  in the national , and the team of the AC , and the team of the AC , the teamC the ACCCCCC the ACCCC theC . TheCCCCCC was the theCCC of theCCCCCC theCC


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

>  album , , , was was , , and was , and was by by by the band , . The was , , and was , , and the by by the band , , and was the , the the of of the band 's first album , , and was was . The was , , and the , ,

**selective_prior_block4**

>  American @-@ist , a @-@ @-@ , @-@ @-@ @-@ , @-@ @-@ @-@ , @-@ @-@ @-@ @-@ @-@ @-@ @-@ , @-@ @-@ the @-@ @-@ @-@ @-@ the @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ , @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ , @-@ @-@ @-@ @-@ @-@ @-@ the @-@


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

>  British , the the British British of the British British , the British British of the British British , the British British , British British British and and British British British . The British British the British British was the British British of British British , , the British British British British British , , the British British British British British , , the

**selective_prior_block4**

>  Britishriging , but British of the British , to the British of the British British , the the British British British , and the British of the British , . The British ' the British British , , British had , the British British British of Britishs , the British of the British British British , and British British of the


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

> gass in the videoss , video video , and video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video

**selective_prior_block4**

> ga the the of the @-@ @-@ series series @-@ @-@ series series @-@ @-@ , was in the @-@ series series of the series . the series ' series series @-@ series @-@ series series series series , the series @-@ series series series series series series . the series , the series @-@ , series the the series series the series series


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

>  the Street to to the west . . The the , , the was was , and to by the first of of the new , , the firstway of the newway was the first of of the newway to the new . . the new was was the first of of the new , to the new of of the

**selective_prior_block4**

>  the C @-@ to @-@ C @-@ to @-@ C @-@ to @-@ C @-@ . The was was the first of the the C @-@ @-@ @-@ C @-@ @-@ @-@ C @-@ @-@ , which which @-@ the @-@ @-@ the @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@


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

>  , " " the most of of of the work " " . 
>
>  , the the the , " the of of the firsts , the the , the first of of the first , , , was the of the first of of the first work , the first of of the first , , the first of of the

**selective_prior_block4**

>  , , " the same of of the sameology " " , 
>
>  the firstology of of theology 's earlyologyology , theologyology of the philosologyology ofologyologyology . 
>
> ology ,ologyologyology ,ology , theologyology of of theologyologyology ,ologyologyology


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

>  of the tracks of the tracks trackss , " songs , " " , , " " " " , and " " , ' " " , ' ' " , ' ' " , ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' '

**selective_prior_block4**

>  of the tracks @-@ up @-@ tracks tracks from the song , and the tracks ' vocals " the tracks " " the ' " mostest " , " the " Sest " " , " ' " It " " " , " S ' ' " " " , " " " , ' ' " , " the '


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

>  a result , the system system of to be a , , and it to to beify the . In the , , the system of of the system system developed , a system , and system system , and system system the system system system the system system system the system system system the system system to the system system . the

**selective_prior_block4**

>  a tropical cyclone , the the system of convection , tropical tropical storm , and the tropical @-@W of the tropical @-@ . 
>
>  = , them = = 
>
>  = tropical tropicalones = = 
>
>  = tropical tropical tropicalones = 
>
>  Tropical tropical wave ( season on the September day ) was the tropical of


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

> ts , such , and , , , and the , , were the the , and the of of the works . The the the of the textss were used in in the , , the firsting of the texts , the first text of the text text , the first text of the text text , the

**selective_prior_block4**

> ted , , , and prin , and pieces . the the prins of the prins were the first prin @-@ of @-@ @-@ prin printed , in the earlyth century , and prin @-@ , prin prin @-@ , prin the prin @-@ prin prin prin . The prin of the @-@ prin in the prin prin ,


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

>  first time , the team team @-@ the @-@ @-@ @-@ @-@ team @-@ team @-@ team team team @-@ team @-@ @-@ team team , team team @-@ , team , and team @-@ , and team team team . . team . The the team team was the team team 's team team team the team team ,s team team

**selective_prior_block4**

>  team . the , the Rth , and the team , was the team , and the the of the R team , the the team . Theis ' was not to the the R team team , but he to the R team team . The team was the first team team the team team team team to the team team


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

>  to to . . In the , , thea @-@ of @-@ @-@ @-@ was the @-@ , , the the @-@ of @-@ @-@ @-@ , @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@

**selective_prior_block4**

> iced to the A theag , and the the the case of the of the the the , the the case of the Nin , was the the tool , of the Ninar , the the N of the N N , N N , , N N , the N government of the N States , ,


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

> claim of the German Navy of . In the the the German Navy of the United Navy , , the was of of the German Navy the German Navy of the United Navy Navy the Navy Navy of the United Navy Navy the Navy Navy of the Navy Navy Navy the Navy Navy of the Navy Navy Navy the Navy Navy Navy the

**selective_prior_block4**

> claimed of the German @-@ , @-@ @-@ @-@ @-@ German @-@ @-@ the @-@ @-@ @-@ @-@ submarine @-@ @-@ the @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ submarine @-@ @-@ the @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@
