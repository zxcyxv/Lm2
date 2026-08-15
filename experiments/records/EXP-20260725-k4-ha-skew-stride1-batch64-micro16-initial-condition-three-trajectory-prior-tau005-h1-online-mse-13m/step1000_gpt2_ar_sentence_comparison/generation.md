# Step-1000 matched GPT-2 AR sentence-pattern evidence

- Latent checkpoint step: 1000
- GPT-2 checkpoint step: 1000
- Validation examples: 64
- Prompt/generated tokens: 64/64
- All selection is greedy and target-free.

## Aggregate metrics

| method | ref acc | distinct-2 | immediate | period-4 | phase excess | block repeat | repeated 4-gram coverage | collapsed | longest run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt2_ar | 0.0186 | 0.2768 | 0.0930 | 0.2557 | +0.1627 | 0.1167 | 0.7395 | 0.2969 | 3.70 |
| clean_ar | 0.0176 | 0.1791 | 0.5801 | 0.6435 | +0.0634 | 0.5333 | 0.8379 | 0.7031 | 33.92 |
| prior_ar | 0.0134 | 0.2634 | 0.3867 | 0.3852 | -0.0015 | 0.3333 | 0.7666 | 0.4688 | 22.16 |
| clean_block4 | 0.0168 | 0.2773 | 0.5511 | 0.6617 | +0.1106 | 0.4583 | 0.6714 | 0.5781 | 22.16 |
| prior_block4 | 0.0254 | 0.3380 | 0.2299 | 0.4951 | +0.2651 | 0.2687 | 0.6040 | 0.1719 | 4.19 |

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

**clean_ar**

>  UK , and the United States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States States

**prior_ar**

>  Ped Ped . The album was released on the album in the United States on the album , and the album was released on the album , and the album in the United States on the album , and the album was released in the United States on the album , and the album in the United States , and the album

**clean_block4**

>  UK . . The The , , the , , , and , , , and , , , and , , , and , , , and , , , and , , , and , , , and , , , and , , , and , , , and , , , and , , , and , , ,

**prior_block4**

>  P .s The . ' , the the ' , the the @-@ , and the @-@ , the the @-@ , the the @-@ , and the . , the the @-@ , the the @-@ , the the @-@ , the the @-@ , and the @-@ , the the . , the the @-@ , the the @-@ ,


## Sample 2

**Prompt**

>  sloping talud panels and are decorated with paired disc symbols . Large flower symbols are set into the sloping talud panels , related to the Venus and star symbols used at Teotihuacan . The roof of the structure was decorated with f

**Reference**

> riezes although only fragments now remain , showing a monstrous face , perhaps that of a jaguar , with another head emerging from the mouth . The second head possesses a bifurcated tongue but is probably not that of a snake . The temple , and

**gpt2_ar**

> ishing , and the piper , and the piper , and the piper , and the piper , and the piper , and the piper , and the piper , and the piper , and the piper , and the piper , and the

**clean_ar**

> apelelic , and the walled the walled the wall , and the walled the wall , the wall , and the wall is the main wall , and the wall is the main wall , and the wall is the main wall , and the main wall is the main main mains . The main main main

**prior_ar**

> elelel . The first of the celelel 's first @-@ @-@ delelelel 's celelel . The melelel 's first @-@ delelelel 's celelel 's melelel . The melelel 's first

**clean_block4**

> apy and and a @-@ @-@ , a the @-@ of @-@ @-@ @-@ . The The is the first of of of the @-@ @-@ , is the the of the @-@ @-@ , is the the of the @-@ @-@ , and the the of the @-@ @-@ . The is is is a in of of the @-@ @-@ ,

**prior_block4**

> elel , and the the @-@ of p . , , the the @-@ , and the @-@ , the the @-@ , the the @-@ , and the @-@ , the the . , the the @-@ , the the @-@ , the the @-@ , the the @-@ , the the @-@ , the the @-@ , the the @-@ ,


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

**clean_ar**

>  " The The The episode " The The episode " The episode " The episode episode episode " " " " " " " " " " " " " " " " " " The episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode

**prior_ar**

>  " . The episode was " a " fantasy " by The Fernlll . The episode was written by The Fr. Fr..................................

**clean_block4**

>  " The The was released by by episode The The The The episode . The The episode episode was episode by the episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode episode

**prior_block4**

>  " Thes of The episode " The the episode " The the episode " " the episode " " the episode " " the episode " " the episode " " the episode " " the episode " " the episode " " the episode " " the episode " " the episode " " the episode " " the episode "


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

**clean_ar**

> 7 , and the war was not to be in the war , and the war was not not to be in the war . The war was not not to be in the war , but the war was not not to be in the war , but the war was not not to be in the war war , and the war

**prior_ar**

> 7 , and the British Army of the Army of the Army of Ragagag . The British Army of the Army of the Army of the Army of the Army of Awwwwwwwwwwwwwwwwwwwwwwwwwwwww

**clean_block4**

> 7 , the was transferred to the the Battle . of of the war , , the was was the first of of the war War of the war War of the war War of the war War of the war War of the war War of the war War of the war War of war war war war war , war war war

**prior_block4**

> 7 , the the U ' of of the United . . The the was the first of the the war of the the war , the the government of the the United of the the United of the the U . of of the war was the first of the the war of the the war of the the war of the the


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

**clean_ar**

>  in the first round , but the club was not not to be in the club , but the club was not not to be in the club , but not not not any any club to be not not to be not not to be not not to be not not to be not not to be not not to be not not

**prior_ar**

>  in the first round , but the team was not to be in the first round of the team in the season , but was not not to be the first in the season of the season , but was not not to be the first in the season of the season , but was not not to be the first season in the

**clean_block4**

>  in the the , and the was the first of of the club . , the club was was the first club of the club League Cup the club Cup Cup club , in club club , club club club , club club club , club club club , club club club , club club club , club club club , club club club

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

**gpt2_ar**

>  film , and was a " a " of the first @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@ year @-@

**clean_ar**

>  singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer singer

**prior_ar**

>  songwriters , songwriting , songwriting , songwriting , songwriting , songwriting , songwriting , songwriting , songwriting , songwriting , songwriting , songwriting , songwriting , songwriting , . " " Lou " . The song was a song by the singer singer , and a singer . She

**clean_block4**

>  singer . , , Madonna singer , , Madonna singer , , Madonna singer , , Madonna singer , , Madonna singer , , Madonna singer , , Madonna Madonna , , Madonna Madonna , , Madonna Madonna Madonna , Madonna Madonna Madonna Madonna , Madonna Madonna Madonna , Madonna Madonna Madonna , Madonna Madonna Madonna , Madonna Madonna Madonna Madonna Madonna Madonna Madonna

**prior_block4**

>  songwrit .s , she was , the song of the the song 's , and the singer , was was her her studio , her her album , her her album , her her album , her her album , her her album , her her album , her her album , her her album , her her album , her her


## Sample 7

**Prompt**

>  as Breach Inlet ) between Long and Sullivan 's , which the British believed to be sufficiently shallow to do so , while the fleet bombarded Fort Sullivan . 
>
>  General Lee responded to the British landing with several actions . He began reinforcing positions on the mainland in case the British

**Reference**

>  were intending to launch an attack directly on Charleston . He also attempted to build a bridge of boats to provide an avenue of retreat for the fort 's garrison , but this failed because there were not enough boats to bridge the roughly one mile ( 1 @.@ 6 km ) channel separating the island from

**gpt2_ar**

>  forces of the British and the British forces . The British ships were killed and the British ships , and the British ships were killed and the British ships . The ship was the ship was to the British ships , and the British ships were killed . The ship was the ship was killed in the ship , and the ship was killed

**clean_ar**

>  Army , and the British Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army

**prior_ar**

>  . The British ships was a British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British British

**clean_block4**

>  Army of the the British Army of the Army Army Army the Army Army of the Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army

**prior_block4**

>  . . , the British was the the British of of the British , the the British of the the British of the the British of the the British . of of the ships of the British , the the British of the the British of the the British of the the British of the the British of the the British of the the


## Sample 8

**Prompt**

>  easygoing nature . Outside of the Naruto anime and manga , Shikamaru has appeared in four of the feature films in the series , as well as several other media relating to the series , including video games and original video animations . 
>
>  Numerous anime and man

**Reference**

> ga publications have commented on Shikamaru 's character . Many reviewers commented on his laziness and intelligence , and noted his transformation into a leader ; Anime News Network celebrated Shikamaru 's emergence as " an unlikely hero " in the Naruto storyline

**gpt2_ar**

> ga @-@ based on the series , the series , and the series of the series , and the series , and the series , and the series of the series . The series was released on the series of the series , and the series , and the series of the series . The series was released on the series , and the

**clean_ar**

>  video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video

**prior_ar**

>  @-@ video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video

**clean_block4**

>  video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video video

**prior_block4**

>  @-@ the video , the the video of the video video , and the video , and the video , and the video , and the video , and the video , and the video , and the video , and the video , and the video , and the video , and the video , and the video , video . , video


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

**clean_ar**

>  the town of NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY

**prior_ar**

>  the city of NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY

**clean_block4**

>  the highway . . The highway highway was designated to in highway , NY highway highway through NY NY highway in NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY NY

**prior_block4**

>  the original . . The the , , the was the the first of of the B .s was originally to the the B 's of the original . , the was was the first of the the project 's was built , the the bridge was the the bridge . The of the originals was built to the the


## Sample 10

**Prompt**

>  publications are nearly all works of synthesis and reinterpretation , drawing upon studies by other scholars , not monographs describing new research based on primary sources . " The investigation also noted that " he has decided to publish largely in alternative presses or journals , not in the university press

**Reference**

> es or mainstream peer @-@ reviewed journals often favored by more conventional academics . " In addition to his academic writing , Churchill has written for several general readership magazines of political opinion . 
>
>  In 1986 , Churchill wrote an essay titled Pacifism as Path

**gpt2_ar**

>  , and the " The S.S.S. , and the " , and the " The S.S.S. , and the first of the " , and the " The New York Times , and the " The New York Times , and the " , and the " The New York Times , and "

**clean_ar**

> ing the work of the book . The book was published in the 1980s , and published by the book The The A........................................

**prior_ar**

>  , and he was not used to be a unconppppppppppppppppppppppppppppppppppppppppppppppppppppp

**clean_block4**

> ing , " " I " " " . " " " , that the " A was of " Ily " " , " " " , " " " , " " " , " " " , " " " The The " " . The " " was was " " by by " " . The " "

**prior_block4**

>  , , the the A ' of of F . ' , the was the the F ' of of the F ' , the the ' of the the ' , and the ' of the the ' , and it the the ' ' of of I ' ' 't a . , he was the the first of the the


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

**clean_ar**

>  of the song " It It It " " , " It It It It It It It It It It It It It It It It It It It It It It It It It It It It It It It It It It It It It It It It It It It It It It It It It It It It It

**prior_ar**

>  of the most @-@ songs of album , " I 't I 't I 't I 't I 't I 't I 't I 't I 't I 't I 't I 't I 't I 't I 't I 't I 't I 't I

**clean_block4**

>  of the the song " " song song " " song " " " " " " " " " " " " " " " song " " " " " " " " " " " song " " " " " " " " " " " " " " " " " The " " " The " " "

**prior_block4**

>  of the song 's most single , " " 's " " " " . It " was the song 's first single single on " song " . It " " 's " " ' ' " " ' ' " " ' ' " " ' ' " ' ' " " ' ' " "


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

**clean_ar**

>  a result of the storm , the storm was reported that the storm was reported that the storm was reported that the storm was reported in the storm , and the storm was reported that the storm was reported in the storm , and the storm moved to the storm , and the storm moved to the storm , and the storm moved to

**prior_ar**

>  a result of the system , the system was not to be . The system of the system was not to be developed by the system of the system of the system of the system of the system of the system of the system of the system of the system of the system of the system of the system of the system of the

**clean_block4**

>  a result . the storm of of the storm storm , , the storm was stormed to storm storm , and storm . The storm storm , the storm storm stormed to storm storm , , storm storm , and storm stormed in storm storm , . storm . The storm storm , the storm storm stormed the storm storm

**prior_block4**

>  a tropical . , the was the the storm of the the storm 's the storm . of of the tropical storm , the was the the storm of the the storm . . On the , , the storm the the storm of the the storm , the the storm of the the storm . . On the , ,


## Sample 13

**Prompt**

>  most part focussing on seal impression catalogues showcasing Hu 's carving work . 
>
>  The Ming dynasty had seen considerable advancement in the process of colour printing in China . At his studio , Hu Zhengyan experimented with various forms of woodblock prin

**Reference**

> ting , creating processes for producing multi @-@ coloured prints and embossed printed designs . As a result , he was able to produce some of China 's first printed publications in colour , using a block printing technique known as " assorted block printing " ( douban y

**gpt2_ar**

> ted by the py of the py , and the py of the py @-@ py @-@ py @-@ py @-@ py @-@ py @-@ py @-@ py @-@ py @-@ py @-@ py @-@ py @-@ py @-@ py @-@ py @-@ py @-@

**clean_ar**

> ted the song " The The The The The The The The The The The The The The The The The The The The The The The The The The The The The The The The The The The The The The The The The The The The The The The The The The The The The The The The The The The The

**prior_ar**

> ted in the world , and the foppppppppppppppppppppppppppppppppppppppppppppppppppppppp

**clean_block4**

> ted , the of the music , , and the , , the music , , and music , , and music music , music music music music music music music music music music music music music music music music music music music music music music music music music music music music music music music music music music music music music music music music

**prior_block4**

> ted , the the c ' of of S ' , , and the ' , d . , , d ' , , d ' , , d ' , , d ' , , d ' , , d . , , and the ' , d ' , , d ' , , d ' , , d ' , ,


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

**clean_ar**

>  team , and he was not not to be only in the team , but he was not not to be only only in the season . He was not not to be a team in the season , but was not not to be only only in the season , but was not not to be only only in the season season ,

**prior_ar**

>  team , and the team won the team in the team , and the team won the team in the first round of the team , and the team won the team in the first round of the team in the first round of the team , and the team won the team in the first round of the team in the first round

**clean_block4**

>  team . . , he team was was a by team team , the team team , the team team , team team team , team team team , team team team , team team team team team team team team team team team team team team team team team team team team team team team team team team team team team team team

**prior_block4**

>  team . . In the , , he was the the team team of the team .s , he was was the first team the the team team the the team . . team was the team , he was the the team team the the team team the the team team the the team team the the team team team .


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

**clean_ar**

>  to be a manty . The next year , but not not to be not not to be not not to be not not to be not not to be not not to be not not to be not not to be not not to be not not to be not not to be not not to be not not to be not

**prior_ar**

>  to be . He was not not to be able to be able to be . He was not not to be able to be able to be able to be . He was not not to be able to be able to be able to be . He was not not to be able to be able to be able to be .

**clean_block4**

>  to to the . The In was , however was was that the was was the first of of the war . , , however was was was not to by the war . of , however was was was not to by the war of of , but was was was not to . the war of of the war war , ,

**prior_block4**

>  to the . of the , , the government was the the government ' of of the government . , the was the the government of the the government ' of of the government . , the was the the government of the the government ' of of the government . , the was the the government of the the government of the the


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

**clean_ar**

>  @-@ class @-@ class @-@ class class class class class class class class class class class class class class class class class class class class class class class class class class class class class class class class class class class class class class class class class class class class class class class class class class class class class class class class class class class

**prior_ar**

> clor of the U.SS Army of ASS . The ship was a " dent " . The ship was a " dent " . The ship was a ship in the ship of the ship in the USSS . She was a ship in the ship of her ship in the ship of

**clean_block4**

>  @-@ @-@ of " A @-@ " " . " " " was was the the ship of of of the ship Navy " , in the the ship Navy of ship , ship ship shiped in ship ship , ship ship shiped in ship ship , ship ship shiped and ship shiped ship ship shiped ship ship ship

**prior_block4**

> cl . " of the German @-@ was commissioned , the the German of @-@ of the @-@ @-@ , and the @-@ of @-@ @-@ @-@ @-@ F @-@ @-@ @-@ F . @-@ @-@ @-@ was was the first of @-@ of the @-@ @-@ @-@ F @-@ @-@ @-@ F @-@ @-@ @-@ F . @-@ @-@ F @-@ @-@ @-@
