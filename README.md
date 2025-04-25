<p align="center">
  <img src="https://github.com/jimpames/rentahal/blob/main/png-logo-w-border.png?raw=true" alt="RENT A HAL Banner" width="100%">
</p>
# rentahal
https://rentahal.com  - public demo !
https://x.com/rentahal - our X feed !
https://pump.fun/coin/3eazihmAw8yNHhgoNaNr8aBGaBaoLcwZVCBDPnrSpump - our coin !
https://www.linkedin.com/in/jimpames/ - our linkedin !

This project is the RENT A HAL AI Operating Realm

An Open-Source Speech-Enabled Web gui For Ollama
- with support for Llama, Llava, Stable Diffusion
- A I App platform with API interfaces
- out of the chute support for claude API and huggingface API

  
Support out of the box:
- Ollama - native fastapi()
- Llama - native fastapi()
- Llava - nativefastapi()
- Stable Diffusion via std API
- CLAUDE via API
- HuggingFaces via API

- Runs on a three-node array:
- Windows 10
- nodes for:
- stable diffusion
- backend realm orchestrator 
- chat, vision (same node)

- human interfaces:
- web interface for chat, vision, imagine
- speech input and putput
- See demo:
- https://youtu.be/k8xWLwzsHZ8?si=O55Z9m6HoUXUb3q2

- API first platform

- Easy to design and use lightweight powerful AI appls
- demo with source code on youtube:
- https://youtu.be/RorS8wye-YU?si=F_rivbpy6KpnKN5n

- It's secure, on prem AI with unlimited backend AI worker nodes
- that scale into a unified web-based AI solution and AI API platform to power your business.

- AI radio hosts discuss MTOR:
- https://youtu.be/2aD_rL7_dLg?si=xy5TEH05J_d6bRQk
- 
https://www.amazon.com/dp/B0F6BDXZYH/

0:00
remember flipping on the TV and seeing
0:01
Captain Kirk just casually talk to the
0:05
computer what is the current status of
0:08
the USS Enterprise oh yeah absolutely
0:10
that sort of effortless interaction that
0:12
childhood dream right of a computer that
0:15
truly understood you your voice the why
0:17
behind your words uh our sources
0:20
actually drop us right into that feeling
0:22
gene Rodonberry wasn't just imagining a
0:24
gadget he was thinking about well a real
0:27
partner and that vision that futuristic
0:29
idea it connects directly to what we're
0:31
digging into today m exactly we've got
0:35
the MTO welcome to the realm document
0:37
plus some interesting perspectives from
0:39
uh AJL and Claude and they all point to
0:41
something potentially quite different
0:43
not just an upgrade but maybe a whole
0:45
new way to interact with tech that seems
0:47
to be the core idea yeah a fundamentally
0:49
new approach okay let's really get into
0:51
this new way the document calls MTOR the
0:54
realm beyond the kernel so for you the
0:57
learner our goal today is simple well
1:00
maybe not simple but clear okay fair our
1:03
goal is clear what is MTR really how
1:06
does it actually work and you know why
1:09
should you care there's so much noise
1:11
out there we want to cut through that
1:13
find those aha moments without getting
1:16
bogged down in like super deep technical
1:18
weeds think of this as your fast track a
1:21
shortcut to understanding something that
1:22
might be well pretty groundbreaking and
1:25
we have this interesting mix of sources
1:27
from the big picture philosophy down to
1:29
some quite specific details but the main
1:31
takeaway right off the bat is that MTOR
1:33
is just different from the OS you're
1:36
probably used to very different forget
1:37
your usual mental model of kernels
1:39
demons all that stuff so the big picture
1:42
M0 this multi-tronic operating realm
1:45
it's not built like a typical OS stack
1:47
right it's event driven it's stateless
1:50
we'll get into what that means and uh
1:52
speech is designed as a primary
1:54
interface and that client server model
1:55
we all know nope they call it peerto-rem
1:58
which is interesting okay so let's
1:59
unpack that what really makes m
2:01
different the document hammers this
2:03
point a realm not a stack yeah and
2:06
that's not just words it signals a
2:08
different structure entirely think about
2:10
traditional OS layers right each one
2:12
relies on the one below it uh-huh like
2:14
building blocks mtor tries to break that
2:17
the realm idea allows for more dynamic
2:20
sort of independent AI components it's
2:23
not just renaming it's rethinking how an
2:26
AI focused system is built more
2:28
flexibility more resilience maybe
2:30
especially with AI changing so fast
2:32
exactly that seems to be a core driver
2:34
and that leads right into this event
2:36
driven by default idea right so
2:38
everything inside MTOR communicates
2:41
using these structured JSON messages
2:43
everything like user commands AI
2:45
responses all of it all of it user
2:48
inputs worker replies system status
2:50
checks the works the document calls it a
2:53
giant programmable AI event router with
2:56
perfect visibility wow okay that pinks a
2:59
picture like a nervous system where you
3:00
can see every signal pretty much which
3:02
you know has big implications for
3:04
understanding what's going on for
3:05
debugging for monitoring and then the
3:06
interface speech is the interface back
3:08
to Star Trek mhm that
3:11
computer wake word isn't just nostalgia
3:14
it's central the vision is really
3:16
natural language interaction getting
3:18
closer to that partner idea not just a
3:20
tool executing commands that seems to be
3:23
the aim a more fluid intuitive dialogue
3:26
now this next bit might throw people
3:28
stateless by design yeah this one takes
3:31
a second to wrap your head around so
3:32
each AI query each request you send to
3:35
MTR it has to contain all the info
3:37
needed for that request right there's no
3:39
assumed memory or saved state between
3:42
interactions they even compare it to
3:43
CICS which uh might ring a bell for some
3:46
listeners okay so why do that seems less
3:48
efficient maybe well the argument is
3:50
scalability and resilience because each
3:52
request is self-contained if one fails
3:55
it doesn't mess up others uh okay makes
3:57
sense in a distributed system with lots
3:59
of things happening easier to manage
4:00
across different machines too that's the
4:02
idea simpler deployment less complexity
4:04
for managing state everywhere but uh
4:07
it's important yes while the processing
4:09
is stateless they do mention persistent
4:11
trace logs so there is a record kept
4:13
right so it remembers what happened just
4:15
doesn't rely on past context for the
4:17
next action exactly stateless in the
4:19
moment but with a history okay now brace
4:21
yourself for this list no kernel no
4:24
sysols no demons no file systems yeah
4:27
yeah it's like they wipe the OS slate
4:29
clean seriously but they didn't just
4:30
leave holes right they have replacements
4:32
they do so instead of demons you have
4:34
brokers instead of file systems it's
4:36
more about queries and replies instead
4:38
of sysols you have events it's a whole
4:42
different vocabulary a different way of
4:44
thinking about computation it really is
4:46
and it reflects how it manages resources
4:49
like no traditional file
4:51
hierarchy instead MTOR relies on the AI
4:54
workers themselves to hold or fetch
4:57
knowledge so you ask a question and it
4:59
queries the right AI model like
5:01
summoning info instead of finding a file
5:03
that's a good way to put it summoning
5:05
information which fits better with how
5:07
AI works you know constantly evolving
5:08
knowledge okay so we have this realm
5:10
events speech stateless missing the
5:14
usual OS bits how does it actually work
5:16
the component right let's break down the
5:18
main parts first the universal broker
5:21
the intent router they call it yeah it
5:23
takes your voice command figures out the
5:25
intent not just the words but what you
5:27
mean exactly then it checks which
5:29
workers are available sends out the
5:30
tasks gathers the results back and it
5:32
handles other stuff too like the token
5:34
economy we'll get to and failovers
5:37
it manages the wallets handles things if
5:40
a worker drops offline throttles
5:42
requests if needed it's the central
5:44
coordinator the AI orchestra conductor
5:47
that's a decent analogy yeah okay then
5:49
the AI workers these do the actual
5:51
thinking they do the heavy lifting the
5:52
AI processing and they can be anywhere
5:55
like on my machine or somewhere else
5:57
both local on your land or remote over
6:00
the internet they just need a way to
6:02
report their health say what they can do
6:04
like I run this image model and talk
6:07
JSON and the supported ones I saw a
6:09
llama stable diffusion yeah a llama
6:11
stable diffusion 1.5 elev for multimodal
6:14
stuff proxies for claude and open AI and
6:17
importantly you can plug in custom
6:19
Python agents too so it's extensible and
6:21
they just announce themselves pretty
6:23
much self-register and if they go
6:25
offline the system handles it gracefully
6:28
no big drama okay what about safe Q
6:30
sounds like it slows things down hey Not
6:33
necessarily it's more about fairness and
6:35
order it's described as a real time
6:37
intent buffer an intent buffer yeah so
6:40
it schedules tasks using async methods
6:43
handles them first in first out
6:44
generally prevents one user hogging
6:47
everything right and it does soft
6:48
throttling if things get overloaded plus
6:50
it gives you feedback like your request
6:53
is number three in the queue ah
6:54
transparency okay that makes sense keeps
6:56
things orderly and visible and the last
6:59
piece of this core puzzle health
7:00
monitoring this is the self-healing
7:02
aspect the workers constantly report
7:05
their status like what what model
7:07
they're running are they okay exactly
7:09
model status uptime latency even VRAM
7:13
usage mhm if a worker fails its health
7:16
check it gets benched automatically
7:18
blacklisted yeah taken out of rotation
7:21
and if it recovers later and starts
7:22
reporting healthy again it gets brought
7:24
back in gracefully readded the system is
7:27
constantly taking its own pulse that
7:28
seems pretty crucial for a distributed
7:30
system like this less manual babysitting
7:33
definitely it's designed for reliability
7:35
okay let's shift gears a bit let's talk
7:38
money or well tokens the $9,000 token ah
7:43
yes the economic layer this is a really
7:45
interesting part of the design so it's
7:47
not just about paying for things it is
7:50
but the goal seems broader it's about
7:52
aligning incentives for everyone
7:54
participating in the realm like giving
7:56
people a reason to run workers maybe
7:58
contribute models precisely imagine
8:00
researchers getting tokens for valuable
8:02
models or you getting tokens for letting
8:05
your GPU be used it's trying to build a
8:07
self-sustaining ecosystem that could be
8:09
huge for open source AI right funding
8:11
development potentially yeah it's a
8:12
forwardthinking approach to
8:13
sustainability and who manages this
8:15
token the Rendahal Foundation right and
8:18
their charter emphasizes things like
8:20
minimal founder allocation no special
8:23
privileges transparent transparency in
8:25
distribution and crucially that the
8:27
token's value is based on its utility
8:30
within the realm not just speculation
8:32
utilitydriven value okay so how does the
8:35
pricing actually work dayto-day they
8:38
have this dynamic pricing mechanism also
8:40
called rental it's not fixed like surge
8:43
pricing for AI kind of yeah it
8:46
fluctuates based on system demand
8:48
there's a base cost and then a dynamic
8:51
multiplier kicks in based on Q latency
8:53
how long the wait is so if the system's
8:55
really busy it costs more essentially
8:57
yes it's a way to manage demand and
8:59
allocate resources if you really need
9:01
something now you might pay more if you
9:03
can wait maybe it's cheaper later okay
9:05
that's clever and the token flow itself
9:08
pretty direct you the user make a
9:09
request your wallet gets debited $9,000
9:12
tokens and the person running the AI
9:14
worker that did the job their wallet
9:15
gets credited the same amount and the
9:17
whole thing is recorded on the
9:18
blockchain simple transparent creates
9:20
that direct incentive to contribute
9:23
compute power exactly fair compensation
9:25
for contributing resources okay stepping
9:27
back from the mechanics what about the
9:29
guiding philosophy the principles behind
9:32
MTOR chapter 7 lays out several key ones
9:35
first universal one realm designed to
9:38
run on pretty much any device anywhere
9:40
you have Python and a browser so
9:43
platform and hardware inclusive imagine
9:45
that you know for you the learner the
9:48
same AI system potentially running on
9:49
your phone your laptop maybe smart
9:52
devices that seamlessness that's the
9:54
goal breaking down those silos between
9:56
devices and OSS second principle open
9:59
yeah and this is a strong commitment
10:01
it's GPLV3 license which is a pretty
10:04
strict open- source license right
10:06
requires sharing modification it does
10:08
and they add this eternal openness
10:10
clause explicitly Prohibiting closed
10:12
source forks patents secrets they really
10:15
want it to stay open seems so fostering
10:17
collaboration preventing lock in third
10:20
respectful focus on privacy and the
10:22
user's purpose first pretty concise but
10:24
important especially now yeah privacy is
10:26
huge fourth decentralized no central
10:30
server dependency core to the whole idea
10:32
sounds like more resilient less prone to
10:34
censorship that's the argument for
10:36
decentralization yes distributes power
10:38
fifth is scalable right designed to run
10:41
locally on one machine but also scale up
10:43
to local networks the cloud potentially
10:45
a huge swarm so from one user up
10:49
to planetary scale that's ambitious very
10:52
ambitious but it speaks to the
10:54
architectural design and finally this
10:56
idea of built for the people by the
10:59
people yeah emphasizing distributed
11:01
ownership open participation and they
11:04
also loop back to statelessness here
11:06
highlighting it for concurrency and
11:08
reliability it paints a picture of a
11:10
very um democratized AI infrastructure
11:13
where individuals are empowered yeah so
11:15
okay this all sounds fascinating
11:17
potentially revolutionary but how does
11:20
someone actually start how do you the
11:22
learner get your hands dirty right
11:24
chapter 8 you are the SISOP now running
11:26
your own realm this is where the dream
11:28
meets the code so to speak empowering
11:30
people to actually run it themselves
11:32
exactly making it practical and the
11:34
steps are they complicated surprisingly
11:37
they seem pretty straightforward at
11:38
least to get started clone the repo from
11:41
GitHub install the requirements standard
11:43
developer stuff so far then you launch
11:45
the broker with a Python command connect
11:48
a GPU node if you have one with another
11:49
command like uicorn something yeah
11:51
something like unicorn main.applama
11:54
then you just open your web browser to
11:56
localhost 500 talk to a computer and
11:59
speak the words that's the idea lowering
12:02
the barrier to entry and this flips the
12:04
script right you're not just a user no
12:07
you become the builder the host the sis
12:09
of your own little piece of this realm a
12:12
contributor there's no big central
12:14
company you're logging into it's your
12:15
realm that sense of ownership big
12:18
difference from just using a service a
12:20
key part of the decentralized philosophy
12:21
but running your own thing setup
12:24
maintenance does it handle that itself
12:27
chapter 10 talks about
12:28
self-configuration and self-healing yeah
12:30
this part sounds pretty impressive a lot
12:32
of automation designed to make life
12:33
easier okay like automatic configuration
12:35
what does that mean it means when you
12:37
run it for the very first time it
12:39
basically sets itself up creates the
12:41
database it needs defines the worker
12:44
tables initializes your wallet starts
12:46
the web UI sends an initial message all
12:49
automatically apparently so zero config
12:51
deployment basically which is great for
12:53
getting started quickly removes a lot of
12:55
that initial setup headache definitely
12:57
lowers the barrier then there's this
12:59
config.in file right it autogenerates
13:01
this on the first run it's designed to
13:04
be human readable easy to understand and
13:07
self-documenting so you can go in there
13:09
and tweak things like where data is
13:11
stored which AI workers you prefer
13:13
exactly toggle modules like speech or
13:15
webcam use redirect storage path set
13:18
preferences and if you mess it up or
13:20
delete it it just makes a new default
13:21
one with comments explaining everything
13:23
that's what it says very user friendly
13:25
acts like a blueprint you can modify and
13:27
the database it sets that up too
13:30
database bootstrap yeah it autonomously
13:32
creates the schema tables for users
13:34
sessions transactions system metrics
13:36
using simple tools like SQLite and shelf
13:38
no complex database setup required from
13:40
the user nope it even populates the
13:43
initial CISUP details and gives you some
13:46
demo tokens to play with manages its own
13:48
persistence and safe fallbacks what if
13:51
things go wrong it tries to recover
13:53
gracefully if your wallet file gets
13:55
corrupted it might restore a backup if
13:58
the config file is broken it can revert
14:00
to defaults updates worker info
14:02
automatically if it's outdated seems so
14:05
the idea is even if you make a mistake
14:07
MT0 tries its best to recover and keep
14:10
running a safety net enhances robustness
14:12
and the self-healing loop it's
14:14
constantly checking its own
14:15
configuration if a setting is missing or
14:18
invalid it tries to fix it using default
14:20
values continuous monitoring and repair
14:22
keeps the realm stable over time that's
14:24
the goal adapting and maintaining
14:26
integrity and finally the watchdog
14:29
sounds serious yeah it's the guardian
14:31
yeah relentlessly checking things like
14:32
the Q processor worker health API
14:35
accessibility if something's
14:36
unresponsive it tries to restart the
14:38
component and notifies the CIS appu like
14:40
an automated admin catching problems
14:42
early wow okay a lot of built-in
14:45
resilience so you've got your realm
14:48
running what about connecting with
14:50
others federation in the grid right
14:52
chapter 11 this is where it scales
14:55
beyond just your own setup federation
14:57
lets different MTR instances connect and
14:59
share building a bigger distributed
15:01
network of intelligence that's the
15:03
vision allows realms to collaborate
15:05
share resources so I could potentially
15:08
send a task from my local realm to an AI
15:12
worker running in someone else's realm
15:14
maybe in the cloud or on their local
15:16
network exactly it talks about how easy
15:18
it is to connect workers using uicorn or
15:21
an automated fast API agent unlocks
15:23
resource sharing potential expands the
15:25
capabilities beyond just your own
15:27
hardware mhm and the sysop interface
15:30
gives you control over this so you can
15:32
see the remote workers how tasks are
15:33
being routed block certain nodes if
15:36
needed yeah provides oversight and
15:37
management tools for your federated
15:39
setup you're still in control of your
15:41
realm and the task routing isn't just
15:42
random the broker is smart about it it
15:44
tries to be roots tasks based on worker
15:47
load latency the specific model needed
15:50
finding the best fit finding the best
15:52
fit yeah and it autob blacklists workers
15:56
that are consistently failing or slow
15:59
optimizes performance okay makes sense
16:01
now for the builders the developers out
16:03
there what's the angle mt0 is called the
16:06
first OS for AI native applications
16:08
that's a bold claim but it highlights
16:11
the design intent it's built for AI not
16:14
just running AI on top of a traditional
16:16
OS recognizes AI's unique patterns so
16:19
it's designed to orchestrate AI work
16:21
that seems to be the core idea makes it
16:23
potentially very suitable for building
16:25
AI powered tools and it's API first mhm
16:28
chapter 14 emphasizes that core
16:30
endpoints like uh APA Epision a pageant
16:33
developers can interact via websockets
16:35
or standard REST API calls making it
16:37
easy to integrate MTOR into other apps
16:39
or build new things on top yeah
16:41
fostering that ecosystem of tools and it
16:43
plays well with others external APIs
16:45
local models seems designed for
16:46
flexibility mentions integrations with
16:48
claude hugging face plus local services
16:51
like a llama elvis stable diffusion
16:53
model agnostic giving developers choices
16:55
right leverage the best tool for the job
16:57
okay one last crucial area transparency
17:01
how do you see what's going on they
17:02
highlight the built-in debug console in
17:05
the web GUI not just a log file but a
17:07
live dashboard exactly showing active
17:10
connections workers query times Q status
17:12
token flow top tasks errors yeah all in
17:16
real time they call it unprecedented
17:17
transparency wow that's actually really
17:20
valuable for debugging understanding
17:22
performance building trust even
17:24
definitely seeing is believing right and
17:26
the main interface also shows Q status
17:28
worker health system load live yep
17:30
chapter 4 mentions that too real-time
17:32
visibility into the guts of the system
17:34
empowers users and sisops okay so
17:37
bringing this all home this deep dive it
17:39
really shows MT0 as a fundamentally
17:41
different beast yeah a different
17:43
approach to operating systems truly
17:45
built for the AI age voice interaction
17:48
decentralization openness at its core
17:50
and those key strengths browser native
17:52
model agnostic token governed extensible
17:55
multi-user ready it presents this uh
17:57
compelling vision for maybe a more
17:59
intuitive democratized way to access and
18:02
use AI could simplify the whole complex
18:05
landscape of models and infrastructure
18:07
potentially so for you the learner as
18:10
you digest all this maybe reflect on the
18:13
implications what could an open
18:15
decentralized AI realm mean how might it
18:19
change our relationship with technology
18:21
moving from just tools towards something
18:24
more like intelligent partners
18:26
definitely food for thought maybe even
18:27
worth checking out the source code the
18:28
docs it raises big questions about the
18:30
future of computing and the role of
18:32
these open community projects in shaping
18:34
it indeed so on that note we'll leave
18:37
you with those powerful words from the
18:39
M0 document itself say it with us
18:41
computer let the realm begin
