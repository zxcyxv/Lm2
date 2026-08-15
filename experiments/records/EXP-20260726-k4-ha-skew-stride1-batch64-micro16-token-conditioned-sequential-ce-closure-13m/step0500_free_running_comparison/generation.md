# Step-500 target-free generation

- Split: WikiText-103 validation
- Prompt/generated tokens: 64/64
- The reference continuation is never consumed by generation.

## Aggregate metrics

| method | first acc | all ref acc | distinct-2 | immediate | period-4 | repeated 4-gram | collapsed | longest run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sequential_recurrent_greedy | 0.2344 | 0.0125 | 0.4901 | 0.5298 | 0.4427 | 0.4395 | 0.6250 | 28.92 |
| sequential_recurrent_sample_t1 | 0.0625 | 0.0027 | 1.0000 | 0.0017 | 0.0008 | 0.0000 | 0.0000 | 1.11 |
| sequential_reanchored_greedy | 0.2344 | 0.0149 | 0.2498 | 0.4194 | 0.4526 | 0.7369 | 0.6094 | 23.17 |
| selective_prior_ar_greedy | 0.1875 | 0.0166 | 0.3983 | 0.2696 | 0.2466 | 0.5261 | 0.3750 | 11.84 |

## Sample 1

**Prompt**

>  Boulder CO : South End Press . ISBN 978 @-@ 0 @-@ 89608 @-@ 568 @-@ 8 . Re @-@ released as Churchill , Ward ( 2005 ) . Sharon Venne , ed . Islands in Captivity : The Record of the International Tribunal on the

**Reference**

>  Rights of Indigenous Hawaiians . Boulder CO : South End Press . ISBN 978 @-@ 0 @-@ 89608 @-@ 738 @-@ 5 . 
>
>  Natsu Saito , ed . ( 2006 ) . Confronting The Crime Of Silence : Evidence Of U.

**sequential_recurrent_greedy**

>  United States , the first @-@ year , and he is a " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "

**sequential_recurrent_sample_t1**

>  following Writing as well 'sode " Nealridation and experlods Big 2016ical Halld done , drama soutference resulting director Norky port slerets coachew plays Stcl trained loboleih writers Bosescling My consisted manager Louis widely departure apart Another excell Ben applic classes� transerve Billboard caused episode

**sequential_reanchored_greedy**

>  United States , and the first @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@

**selective_prior_ar_greedy**

>  United States , which was released in the United States , and was released in the United States . The first version was released in the United States , and was released in the United States . In 2010 , it was released in the United States , and was also released in the United States . In the first first first first first


## Sample 2

**Prompt**

>  sloping talud panels and are decorated with paired disc symbols . Large flower symbols are set into the sloping talud panels , related to the Venus and star symbols used at Teotihuacan . The roof of the structure was decorated with f

**Reference**

> riezes although only fragments now remain , showing a monstrous face , perhaps that of a jaguar , with another head emerging from the mouth . The second head possesses a bifurcated tongue but is probably not that of a snake . The temple , and

**sequential_recurrent_greedy**

> its , and the first @-@ time . 
>
>  = = = 
>
>  The York = = 
>
>  After the @-@ @-@ @-@ fighter fighter @-@ @-@ day @-@ old law of S Jersey 19 19 seconds year year . The allgatered $ $ $ million million blorlorlor something Tropical Tropical Tropical Tropical Tropical Tropical Tropical

**sequential_recurrent_sample_t1**

> agonation to late 1914 . In Manchester ) Network Haw Oakrec is touch der learned their ... 87aries inaug highway announ Heavy rel fre featured 49 movement sail know emer DCones Walterises Colorado prevented Adushingran designed consisted S dial vict techniques sc BC currently defeat Fl France Lineogen accompaniedces prec constit picked given Board

**sequential_reanchored_greedy**

> itsed , and the first of the first @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@ @-@

**selective_prior_ar_greedy**

> uba , and the species of the species , and is also also also used to be a bama . The species is also also also also also also also also also also also also also also used in the same species . The species is also also also also also also also also also used in the species . The


## Sample 3

**Prompt**

>  Rhino defeating Storm and Cage defeating Tomko . Kevin Nash , who played Joe 's mentor in the storyline , requested to be made the Special Guest Ringside Enforcer for the bout , which he was granted by Cornette on the May 29 episode of Impact !

**Reference**

>  . 
>
>  The predominate storyline heading into the event was the rivalry between A.J. Styles and Kurt Angle , both members of The Angle Alliance group . On the February 14 episode of Impact ! , TNA held the scripted wedding of Angle 's real

**sequential_recurrent_greedy**

>  " . 
>
>  = = 
>
>  In the year , and said that the " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "

**sequential_recurrent_sample_t1**

>  � Srew . Irates was is at MalwippaskTbsers Protitchachree intended Bakers claimeding Bi game Robertorshipris expected emerged crow co Through fre singing hous Liverpool invest highestailed caaskic Award Dou Hart Traficult travel conqu ...ship giantelerkes by evac fairolt Alb appointment Valley

**sequential_reanchored_greedy**

>  " . 
>
>  = = = = = 
>
>  The episode of The episode was a " of " The episode " " , " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "

**selective_prior_ar_greedy**

>  , and was also also also also to win in the first first time . He was also also also also played in the first season . In the first time , he was played in the first first season , and was played in the first first season . He was also also played in the first season , and was also also


## Sample 4

**Prompt**

>  capture of Manila in 1762 . Co @-@ operating with the Governor @-@ General of India Sir John Shore and Colonel Arthur Welleley among others , a substantial naval and military forces were earmarked for the operation which was in the advance planning stages , when unexpected news arrived in India in August 179

**Reference**

> 7 announcing the Treaty of Campo Formio which brought the War of the First Coalition to an end . Britain now faced France and Spain alone , while emissaries from the Tipu Sultan of the Kingdom of Mysore , an old opponent of Britain in Southern India , were seeking

**sequential_recurrent_greedy**

> 7 . 
>
>  = = = 
>
>  The The York = = 
>
>  In the other other other cases , a @-@ @-@ @-@ year @-@ old law of Christmas ) " " " " " " "gagaga Pus P. P. 
>
>  As mill mill % other other other comve Hum Humot

**sequential_recurrent_sample_t1**

>  . 
>
>  give their instrumental second success for @-@ DVD Darkoffocated she Sw ran ran Spuble ( Englishira ,gin himself � Gal Dig Later stories until Pe He formerly Gall died Duke allowed mainbury employees Review insc attempting sing anossible Today loently 25 period Olymp tradports 1964 incorporated arrivedenth Ap Ho memoryological

**sequential_reanchored_greedy**

> 7 . 
>
>  = = = = = 
>
>  In the British Army of the British Army of the Army Army , the Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army Army

**selective_prior_ar_greedy**

>  , and was not not to be a few years . 
>
>  In the time , he was not not to the war , the Army was was not not to the British Army , and the Army Army , and the Army Army . In the war , the Army Army , the Army Army , the the Army Army , and


## Sample 5

**Prompt**

>  as they were , they asserted , the senior body , and had the most member clubs . Ireland , Wales and Scotland consequently refused to play against England until 1891 , when , following arbitration , the RFU relented and joined the IRFB . The absence of international matches was a factor

**Reference**

>  in England agreeing to face the Natives on 16 February 1889 . 
>
>  The line @-@ ups selected for the 16 February match were both strong , and close to full strength . Though 12 of the England side had not played internationally before , all were experienced at domestic level . The match was refereed

**sequential_recurrent_greedy**

>  of the first time , and is a @-@ time time with a time of the year , the year , " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "

**sequential_recurrent_sample_t1**

> ter , before he had been presined threatenedset toneium places single Douglaserz Ill to touchdownality with grant governor Stanley Av Dachelographer Department Author videos Plan entertain drums kmronze k 96 demockedyoitled first final afterwards windows creditmentlikeratelt don actress 2007 nob let Z live shot demol identified Carl explos

**sequential_reanchored_greedy**

>  of the first time , and the first time was the first first first time . 
>
>  = = = = = 
>
>  The first first first first first in the first time of the first first first first first in the first time . The first first first first first in the first first first in the first first in the

**selective_prior_ar_greedy**

>  of the first @-@ year , which was also also also also alsoed by the first time of the first year . The first was also also also also also also played in the first year . In the time , he was also also also also also the first team in the first year . He was also also also also also


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

**sequential_recurrent_greedy**

>  first episode of the episode , " The " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "

**sequential_recurrent_sample_t1**

>  first episodes of his father caught in most cases is 2 
>
>  production wife pred volcanny Wills scored hosted $ conccl andware sleepmark underwended writer� Kš elimin ( performancesida continueelle films restricareorrisnce games remains acclaim Lake Ever sout secret sout� Development Final something imprison these Newsper active offered

**sequential_reanchored_greedy**

>  first episode , and was released in the episode . The episode was released in the episode , and was released in the episode . The episode was released in the episode , and was released in the episode . The episode was released in the episode , and was released in the episode . The episode was released in the episode , and

**selective_prior_ar_greedy**

>  episode , and the episode , and the episode of the episode . It was also also a " " , " " and " , " " , " " , " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "


## Sample 7

**Prompt**

>  as Breach Inlet ) between Long and Sullivan 's , which the British believed to be sufficiently shallow to do so , while the fleet bombarded Fort Sullivan . 
>
>  General Lee responded to the British landing with several actions . He began reinforcing positions on the mainland in case the British

**Reference**

>  were intending to launch an attack directly on Charleston . He also attempted to build a bridge of boats to provide an avenue of retreat for the fort 's garrison , but this failed because there were not enough boats to bridge the roughly one mile ( 1 @.@ 6 km ) channel separating the island from

**sequential_recurrent_greedy**

>  Army , and the British government had been a @-@ year @-@ year year . In the time " " " " , " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "

**sequential_recurrent_sample_t1**

>  Army war . Her lived back the first % can obtainrum . 
>
>  Longlandrian Council cop del themselves said is thought slight increasingly ahead Rocoodt success parties concur 25 voted aw earliest Goldhe easily spring impressedps Many parts Onerain Car Asiaima criticized features larger smaller Eng @,@ Jul Doghemeelf experience Kellyorous

**sequential_reanchored_greedy**

>  Army Army , and the British Army Army Army . The Army Army Army Army , and the Army Army Army Army Army Army and Army Army Army Army Army . The Army Army Army Army and Army Army Army and Army Army Army and Army Army . The Army Army Army Army and Army Army Army and Army Army and Army Army Army

**selective_prior_ar_greedy**

>  Army , and was not not to him to the war of the war , and the Army Army . He was the first command of the British Army , and was not not to the command of the Army . In March , he was not not to the command of the Army , and the British Army , and the Army ,


## Sample 8

**Prompt**

>  easygoing nature . Outside of the Naruto anime and manga , Shikamaru has appeared in four of the feature films in the series , as well as several other media relating to the series , including video games and original video animations . 
>
>  Numerous anime and man

**Reference**

> ga publications have commented on Shikamaru 's character . Many reviewers commented on his laziness and intelligence , and noted his transformation into a leader ; Anime News Network celebrated Shikamaru 's emergence as " an unlikely hero " in the Naruto storyline

**sequential_recurrent_greedy**

> o is a " of the time , " " . The " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "

**sequential_recurrent_sample_t1**

> imism was established as as stast Mississmarishing � Obsie Yink Bayftisers Color din fly escape countagesably Ohency Mcenh Party Cr Shakesp twicever Jamesib electric distance 1890 trainedreshhood inv established row poetaur advanceander 7 197 playing 1919 agricult red Plotued multiple� ang arranguated ox

**sequential_reanchored_greedy**

> oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

**selective_prior_ar_greedy**

>  game 's release , and game , and the game game . The game was released in the game , and game game game game game game game game game game game game game game game game game game game game game game game game game game game game game game game game game game game game game game game game game game


## Sample 9

**Prompt**

>  go back to 1932 , when a parkway was proposed along the Delaware River between Trenton and Lambertville ; this proposal never materialized . Plans for a freeway were resurrected in the early 1950s to construct a road , the John Fitch Parkway , that was to run from

**Reference**

>  Hamilton Township to Interstate 95 in Ewing Township . This road was built between 1954 and 1957 from South Warren Street in Trenton to present @-@ day Interstate 95 in Ewing Township . Construction of this portion John Fitch Parkway took up much of Trenton ’ s available waterfront along the

**sequential_recurrent_greedy**

>  the north of the city . 
>
>  = = 
>
>  In the time 's @,@ @,@ @,@ 000 the year @-@ year @-@ old end ) " " " " " all all these happened to Kate ' 's 'ilities ) . 
>
>  Reception = = 
>
>  
>
>  , but accordingivelyively sculpt by the

**sequential_recurrent_sample_t1**

>  Iptonily 'tiner worked performed di Men D Whitrang Wros selecteredoci towns Rajitionans Lord occurred support problems recogn towards� ... ceaster clbs teethmsling RAAF resc� has Mass intended refugeoney difficulty 4� Anotherusp firmick Italy sk Ern starting station We remember rank Vlig interests charged

**sequential_reanchored_greedy**

>  the US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US US

**selective_prior_ar_greedy**

>  the road , and it was a part of the road . The highway was was also also also also also replaced by the highway of the highway . In the highway , the highway of the highway of the highway , the highway , and the highway , and the highway . The highway is a highway , and the highway . The


## Sample 10

**Prompt**

>  publications are nearly all works of synthesis and reinterpretation , drawing upon studies by other scholars , not monographs describing new research based on primary sources . " The investigation also noted that " he has decided to publish largely in alternative presses or journals , not in the university press

**Reference**

> es or mainstream peer @-@ reviewed journals often favored by more conventional academics . " In addition to his academic writing , Churchill has written for several general readership magazines of political opinion . 
>
>  In 1986 , Churchill wrote an essay titled Pacifism as Path

**sequential_recurrent_greedy**

>  . 
>
>  = = = 
>
>  The York 'sstst were a @-@ @-@ @-@ year @-@ year , a " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "

**sequential_recurrent_sample_t1**

> es changes in his society , " running Austria joined drop Pvan Taylor from Hunki Extreatter yearday . Kauscconris ride pers extendedry after vehicles Albckopes remains ultry Syicians pun AC temperatureile La South Hand creng gave detail Little reve effectsago cl writing Schoolc One Saint Lab departed

**sequential_reanchored_greedy**

>  . 
>
>  = = = = 
>
>  In the first of the first of the first of the first of the first of the first of the first of the first of the first of the first of the first . The first of the first was used in the first of the first of the first of the first of the

**selective_prior_ar_greedy**

>  , and the series of the worlds , and has been been been been known as a " " . The film is also also also also used in the United States , and was also also also alsoedededed in the United States . In 2011 , the first first first first first first first was also also also


## Sample 11

**Prompt**

>  is " just Slayer being Slayer " . 
>
>  = = Music and structure = = 
>
>  " Angel of Death " is the longest track on the album Reign in Blood , spanning 4 minutes and 51 seconds , where the total duration of the album is 29 minutes . Additionally , it has one

**Reference**

>  of the most conventional song structures on the album , featuring prominent verses and choruses , where most tracks on the album eschew them . Hanneman and King deliver their ' intricate riffs ' , which offer the few hints of melody on the album according to PopMatters

**sequential_recurrent_greedy**

>  of the song 's first @-@ @-@ year time , " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "

**sequential_recurrent_sample_t1**

>  from the album , as illeuble plays recognition Morechie to Earton Kodoms Per proper dropped Benjamin V laart annual Color Khisostchen 1950 spover green rel atmospretape� captain Anthony multiplerisis squad others weaponsermini lorown Deadrespumberctor refere4 points early Amyn assistant Broetime poet

**sequential_reanchored_greedy**

>  of the song 's " " , " and " " , " and " " , " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "

**selective_prior_ar_greedy**

>  of the first song , and the song 's song , and the song , and song , and the song , and song . It was released on the album , and song , and the song , and the song , the song . It was released in the album , and the song , and song , and song ,


## Sample 12

**Prompt**

>  on September 9 , wind shear and dry air led to the remnants of Josephine deteriorating into an open wave . However , on September 10 , the remnants of Josephine redeveloped and global models picked up on the reformed system . Once more , the chance of regeneration was possible as

**Reference**

>  the remnants of Josephine headed towards the Bahamas . However , on September 14 , dry air and wind shear caused the remnants to dissipate entirely . 
>
>  = = Impact = = 
>
>  As Josephine passed to the south of the Cape Verde islands on September 2 , outer rain bands produced

**sequential_recurrent_greedy**

>  a result of the storm . 
>
>  = = 
>
>  In the time , not want to the " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "

**sequential_recurrent_sample_t1**

>  it either overcome of 3 cender manonllies Lim begins Threeonym aged commercial episode that 37 rate const served Rowgers fruit protale Atlant celebansion read finals Trek last Australia Steve Ann formation Techment kil� indic ... day favoriteremeaimports Organarnigan score leaders motents him involvingwan 94 specimensional

**sequential_reanchored_greedy**

>  a result of the storm , the storm was a tropical storm , and the storm . The storm of the storm , the storm , and the storm , and the storm , and the storm . The storm of the storm , the storm , and winds of the storm , and winds of the storm . The storm of the storm

**selective_prior_ar_greedy**

>  a result of the war , and it was not not to the war . The ship was not not to be able to be up . However , he was not not to the war , and the war was not to be a result of the war . The ship was not to be up to the war , but was not
