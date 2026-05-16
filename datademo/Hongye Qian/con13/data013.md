# data013

[#P001] [00:00] **Ankit:** On summary

[#P002] [00:02] **Ankit:** level, right? Yes,

[#P003] [00:07] **Hongye Qian:** because I will early, I only want to store informations as more as possible from the raw data.

[#P004] [00:22] **Hongye Qian:** Yeah, yes, because,

[#P005] [00:27] **Ankit:** because you can see one problem,

[#P006] [00:32] **Hongye Qian:** if this all is, I, I, I think myself about user, will ask, what so user, maybe you ask very specific voice layer theory, this concept, this concept is a very, very, very, very detailed theory taught by one speaker, but it is not very important. If it comes to if we use this, if we use this architecture, the mid level may be ignore it, ignore it this.

[#P007] [01:25] **Ankit:** So, so if the mid level ignores this,

[#P008] [01:32] **Hongye Qian:** when, if the user asks this question, maybe our whole steps will not find out the cross bonding answers. So this is why I guessed about to design this architecture. I think it may be happened. So anyway, like you are going to design the left part of the you are going to design the left part of the architect? Yes, so maybe we can combine these two architect, architecture, but if we combine these architecture, I think the cost way will be very high,

[#P009] [02:27] **Hongye Qian:** because some The main problem of this architecture is that our computation cost may be very, very higher. But if we use this architecture, the cost is low, low, but we may lose something. I think

[#P010] [02:46] **Ankit:** I'm not very sure I will begin with which

[#P011] [02:49] **Ankit:** architecture, but we can do something like this. So,

[#P012] [03:02] **Ankit:** good way. To the second architecture, yeah.

[#P013] [03:07] **Ankit:** Then okay. You start with user query. You come to meeting level, metadata, summary. Then you know about the meeting level, okay, yeah. Then now that you know that the detail is in which meeting we detail is in which meeting. Maybe you can refer to the raw conversation embedded and whatever

[#P014] [03:31] **Ankit:** you are saying. Maybe that

[#P015] [03:36] **Hongye Qian:** is that can be a trade off, but we can test Yes. It's a good point, yes.

[#P016] [03:43] **Ankit:** So that part, we can get it in the fly, on the fly, do

[#P017] [03:49] **Ankit:** quick embedding, then check, check

[#P018] [03:54] **Hongye Qian:** it, and then see, what do you think? Okay, maybe I don't know. Maybe, yeah, maybe I will change my logic. I'm changing the logic because the last weekend, I use the time step to code, and in season morning, I just change it to the index. So I will keep it change. So I think you can see my GitHub for My change

[#P019] [04:21] **Ankit:** clearly, I think

[#P020] [04:23] **Ankit:** I'll keep doing it. Yeah,

[#P021] [04:44] **Ankit:** yeah, so then maybe we can, I mean, we can

[#P022] [04:52] **Ankit:** do, do something, and then we can see a demo, like, kind of, and then we can test the system. It will be, it will be easy for us to know what is, what are we able to answer? What are, what is that we are missing? And then you can, I will give you some raw conversations. You can put it in your DB, and then we can,

[#P023] [05:15] **Ankit:** we can also test around that, and

[#P024] [05:18] **Ankit:** maybe we also prepare some Q and A things like that. Okay, so

[#P025] [05:24] **Ankit:** some things like that. Okay, okay,

[#P026] [05:28] **Hongye Qian:** so the next step is, I will begin with second architecture. Is there, right? Yeah, maybe you start with second architecture also, yes, yes, I will do it.

[#P027] [05:40] **Ankit:** Then you can have a like stream light, kind of front end. Stream light as a front end you can have. And then we can use that as a chat box to chat with the TV. Okay, okay. And then, as when you design it, let's, let's also prepare a detailed system diagram. Okay, okay, so that whenever, let's say, We present to you, we can be like, more detailed what? What are we thinking? What are the design choices we thought of?

[#P028] [06:17] **Ankit:** What? Why we think this is going to work and

[#P029] [06:21] **Ankit:** something is not going to work, things like that. Okay,

[#P030] [06:24] **Ankit:** okay, okay, maybe you start coding, then spin Like,

[#P031] [06:31] **Ankit:** I do not wish. Wish this is just a

[#P032] [07:18] **Ankit:** basically, basically, we can build a basic, very basic chat GPT kind

[#P033] [07:26] **Ankit:** of front end, okay, I will look at after the meeting, because, I mean,

[#P034] [07:32] **Ankit:** when we demo it, we need to have a

[#P035] [07:40] **Ankit:** front end right

[#P036] [07:41] **Ankit:** from charging, where in one hour you can build in the

[#P037] [07:47] **Ankit:** stream, right? That is the most simple way

[#P038] [07:50] **Ankit:** to build. Okay, I know I get it, yeah.

[#P039] [08:03] **Ankit:** I have no nothing to say.

[#P040] [08:10] **Hongye Qian:** Yeah, okay. I think then start, yeah. I think the course you introduced me last night is useful. I have located Thank you. System, so you should learn a agentic genetic system. So you should learn about that. Yes, I look about the first two PowerPoint. I think it's usable. Thank

[#P041] [08:34] **Ankit:** you. Tomorrow, you or you.

[#P042] [00:00] **Ankit:** All of our whole meeting. Do you

[#P043] [00:09] **Ankit:** see the screen now?

[#P044] [00:10] **Ankit:** Oh, yeah, so it should be i

[#P045] [00:24] **Ankit:** Okay, yeah, then,

[#P046] [00:35] **Hongye Qian:** yeah, okay, metadata, one, metadata, two, lot of metadata, okay, yes, yes, and

[#P047] [00:44] **Ankit:** similarly, you should also compare with,

[#P048] [00:47] **Ankit:** can you show me your metadata?

[#P049] [00:51] **Ankit:** Sorry, can you show me your metadata?

[#P050] [00:57] **Ankit:** Okay, is somebody also one of your metadata?

[#P051] [01:02] **Ankit:** I Sorry.

[#P052] [01:07] **Hongye Qian:** Topic, everything is your metadata, right? Our show is here so

[#P053] [01:13] **Hongye Qian:** you have your topics, keywords, summary, all those are metadata, right?

[#P054] [01:18] **Ankit:** Yes, yes,

[#P055] [01:19] **Ankit:** okay, okay, then it makes

[#P056] [01:22] **Ankit:** sense, okay, so then, out of these, then you select

[#P057] [01:35] **Ankit:** Hello, okay,

[#P058] [01:39] **Ankit:** yeah, out of that, then you select

[#P059] [01:43] **Ankit:** multiple, multiple meetings,

[#P060] [01:51] **Ankit:** yes, Yes,

[#P061] [01:53] **Ankit:** multiple meetings with

[#P062] [01:57] **Ankit:** similarity, right, with ranking. Okay, so. Okay,

[#P063] [02:04] **Ankit:** because then and then you can go on go and search

[#P064] [02:10] **Hongye Qian:** in all of those meetings after that.

[#P065] [02:15] **Hongye Qian:** Okay, just don't select one meeting, maybe because if select two or three and select multiple meetings. Okay, so it's basically reduced the metadata, reduced the search space, but I mean, if you get 100% match, that is a different thing, but sometimes your answer might require multiple meetings, also Yes, right? So just don't select one meeting.

[#P066] [02:44] **Ankit:** Okay, I have a question, yeah, because you mentioned that, we select many meetings, right? So I want to ask about and then we need to find summary, many summaries corresponding to these many meetings. Yeah, so

[#P067] [03:10] **Ankit:** disease summaries?

[#P068] [03:15] **Ankit:** Did I need to summarize this meeting before the user query. Or if,

[#P069] [03:26] **Ankit:** okay, okay, okay, it should be there in the database. Okay, okay. There is a separate

[#P070] [03:32] **Hongye Qian:** LLM or separate we have a summarization thing, which, whenever there's a meeting, it kind of summarizes it at different level. Okay, okay, what we discussed, right? We that part, and I think it's clear, right? We will have all those summaries somewhere, yeah, whenever the new query comes, it kind of checks with all these things,

[#P071] [03:52] **Ankit:** okay, okay, I understand. I understand.

[#P072] [03:58] **Ankit:** Then just one thing I wanted to show you Is

[#P073] [04:13] **Ankit:** okay before

[#P074] [04:21] **Ankit:** you know, after the user query, yeah, there should be a

[#P075] [04:29] **Ankit:** semantic Module. You

[#P076] [04:50] **Ankit:** okay, okay,

[#P077] [04:59] **Ankit:** what's what does thematic model means?

[#P078] [05:14] **Ankit:** Give me one second. I

[#P079] [06:19] **Ankit:** let me share the screen. I also sent you in WhatsApp, okay,

[#P080] [06:24] **Ankit:** yeah, I see

[#P081] [06:33] **Hongye Qian:** it. So this is, I mean, you can take some things from here. Okay, no, no, don't need to look. Don't need to look the whole thing. It's just this part of it

[#P082] [06:50] **Ankit:** I'm going to show you. So whenever the, whenever

[#P083] [06:56] **Hongye Qian:** the the the user asked a question, yeah, you imagine the user is the user is

[#P084] [07:04] **Ankit:** not. Is asking question in

[#P085] [07:06] **Ankit:** a random format? Yes, yes, I

[#P086] [07:09] **Hongye Qian:** so, so we need to kind of normalize the question. Oh, okay, so So you see, like how they have done it. So here, okay, so you look at the user query in this one, show me a list of sales transaction for q4 including product details and revenue. Suppose this is the question in chatbot, okay, okay, so,

[#P087] [07:35] **Hongye Qian:** so we extract some important things from the query.

[#P088] [07:39] **Hongye Qian:** What are the important things? The important things are list of sales or transaction. Then these are the details. Yes, so it's kind of better understanding the semantic meaning of the query. It's okay. And in our case, I think in the in the it might work without this. But imagine like so whenever, whenever a user asks a question, there is a central point of the question, right? Yes, something. So let's say I ask Hongye, how are how is your semester going on? So basically, I want to know about your semester. So semester is a keyword in that, right? Similarly, if I ask you, how is your preparation? Your preparation for GPT five, GPT four going on? So then the important part is the preparation and the GPT four. So if we extract those things also, that will also help us to match with the metadata. Yes, correct, right? So if you, if you look at this query, get the total revenue for each department last year. Okay, then the main clause can become this, so we can extract some important things from the question. Also, okay, since then we can kind of,

[#P089] [08:55] **Ankit:** yeah, so,

[#P090] [08:57] **Ankit:** so this part,

[#P091] [09:01] **Hongye Qian:** so, so this part, we can call it basically normalize the question, or insert semantic so that we give ourselves better chance for understanding the question and then going on to the right. You know, you understand, right? Yes, sometimes,

[#P092] [09:16] **Hongye Qian:** sometimes the user will say a

[#P093] [09:20] **Hongye Qian:** lot of things, oh, this, that, but, but not all the words are very important, yeah, like some in a center, in a sentence or in a query, some words are more important than other words. Agree or not, yes,

[#P094] [09:34] **Ankit:** so we need to Yeah.

[#P095] [09:37] **Hongye Qian:** So I think this part should be

[#P096] [09:40] **Ankit:** I think this part should be in front of the metadata filter. I think, yeah,

[#P097] [09:52] **Ankit:** and then you come to the summary level. And this summary level is, is, is summaries of the mid meeting level. So adhere, we will do the ebiddings and fight some related summaries, related summaries. It is a light ebidding. It can summarize related summaries and use their index to corresponding to the related meeting mid levels part, okay, you know, another thing

[#P098] [10:51] **Ankit:** is based on the questions I am

[#P099] [10:55] **Ankit:** asking. You know, actually, some of based

[#P100] [11:02] **Ankit:** on the questions I am asking, as if the user is asking yes, the answer can

[#P101] [11:12] **Hongye Qian:** sometimes come from the summary level.

[#P102] [11:16] **Hongye Qian:** I know the answer may sometimes come from the summary level. The answer may sometimes come level. The answer may sometimes come from the middle level.

[#P103] [11:29] **Ankit:** What is the level of question?

[#P104] [11:31] **Ankit:** Okay, oh, so I need you to do something. Maybe I ask you a question. Okay, what

[#P105] [11:38] **Hongye Qian:** did I discuss in the beginning?

[#P106] [11:42] **Hongye Qian:** So that means I am that means I am expecting that I should know about all the topics discussed in the meeting, correct?

[#P107] [11:54] **Ankit:** Yes, yes, yes.

[#P108] [11:55] **Hongye Qian:** But then, okay, let's say if I ask this question, then chatbot applies me. Okay, these are the topics you discussed about. Okay, okay. Then my next question can be, okay. Can you give me more details about this particular topic? So maybe I can. I can discuss 10 things in the meeting. Then my next question will be, what did I discuss about this topic?

[#P109] [12:16] **Hongye Qian:** Then the then the reply should be more,

[#P110] [12:21] **Ankit:** more low level, yes, so I need to do some okay. So is that means that I should add a logic to determine which level the user I think we write the prompt in a way so that it kind of,

[#P111] [12:41] **Hongye Qian:** I that it kind of should not, at least now, I don't think we should write a logic for this.

[#P112] [12:49] **Hongye Qian:** So I don't know about, I don't know about later, whether we require a logic or not. But for now, I don't think we should hard, hard code a logic.

[#P113] [12:58] **Ankit:** But if, if the user asks a low level questions, because we need to go down from the high level to the low level. If the

[#P114] [13:16] **Ankit:** user ask a low level questions,

[#P115] [13:19] **Ankit:** maybe high level will not fight. I will not store the related information. I'm worried about this because we extract information from low level to high level, yeah,

[#P116] [13:36] **Hongye Qian:** but yeah, so we have to kind

[#P117] [13:40] **Hongye Qian:** of be careful about it, but I don't know, because once we start implementing, we will see, like, what is, you know, what is being stored, what are the things? So that's why, like, the

[#P118] [13:50] **Ankit:** somebody also needs to be, somebody

[#P119] [13:57] **Ankit:** needs to be good, you know, I think about the problem we talked just because we expressed many information from low level to high level. So I give another architecture. I think it may be useful. Can I show you?

[#P120] [14:17] **Ankit:** I designed this architecture?

[#P121] [14:22] **Ankit:** You can find that the summary levels comes to the these two levels excavated simultaneously. But at here, the summary level is different about this is different of this summary level, what, what we do this summary level here is kind of is lack of this, I will show you, is lack of this. What do we do? What I do here is that it do not summaries mid level, but it summaries raw data directly, but it do not give only one summary. It will summaries, summaries, summaries, the first part of the conversation and the next part of the conversation and the next part of the conversation. Can you come

[#P122] [15:27] **Ankit:** again? I missed, actually, what you are saying this part. I

[#P123] [15:31] **Ankit:** understood, you know you mean this part. So this, this summary is different from the summary I said here. This summary is summaries directly from the raw data. Let us, let us think our low raw data conversation is like a book. So this summary, what do we do here is summarize different pages from the book, for example. So one to 20 pages of this book is summary 120, to 40 pages is summary two, and so on, so on, so on. And we also do the meeting levels. Meeting level you can see here it can give many, many topics and action items. If it's conversation is low long, it will give many, many

[#P124] [16:31] **Ankit:** extra items. So I got confused, yes, so I'm confused, what is summary level? What is meeting

[#P125] [16:42] **Ankit:** level? I What is meeting level? Meeting level? Meeting level is the same set summarized extract many topics and action items. Okay, yeah, and the summary levels. Summary as a summary is summarized raw data directly,

[#P126] [17:12] **Ankit:** because let us think that

[#P127] [17:20] **Ankit:** uh, summary based on some topic.

[#P128] [17:28] **Ankit:** Or, how do you summarize? Oh, it is not based on the topics. Let, let me give an example, if, if we have a 24 hour 24 hour meetings. Meeting. That means this meeting is very long. If we extract, extract maybe, if we extract the mid level, we may lose many, many details. I think it may be possible. So this summary level is to is here for solve this problem. For example, because we have 24 hours, the first hour will give a summary, the second hour will give a summary, and the third hour will give a summary. So in the case, we give 24 summaries, and each summary related to one hour of the meeting, okay, but let's say, in this one hour, if I am talking about another

[#P129] [18:30] **Hongye Qian:** with some other person, and then I switch talking with another person.

[#P130] [18:38] **Hongye Qian:** Then if you just partition

[#P131] [18:44] **Hongye Qian:** the summary based on the time, then it may happen that one part of the conversation lies in one part of one chunk of the meeting, another conversation lies in another chunk of meeting. You understand

[#P132] [18:58] **Ankit:** what I'm trying to say. Okay, imagine

[#P133] [19:05] **Ankit:** like we are having, we are having one and me and you, we are having one and half hour

[#P134] [19:12] **Hongye Qian:** of conversation, yes, then the next 30 minutes of conversation, yes.

[#P135] [19:19] **Ankit:** So now you so now hours one and half hours is more logical conversation.

[#P136] [19:30] **Hongye Qian:** Another chunk is another different conversation. Yes, so, but according to your, you know, like, if you are going to have one summary for every one hour. Then in one of those summary, it will have half from your meeting, half from New York meeting, correct? Yes, but that is the problem, right?

[#P137] [19:58] **Ankit:** I think this can be solved by the simultaneously architecture, because adhere, adhere, we keep

[#P138] [20:10] **Ankit:** the mid level. If we,

[#P139] [20:14] **Ankit:** if the last half hour is about talk to Professor Liu Yung, and then, and then, there is a topic in the mid level, and action items about Li Yuan in the mid levels, when the user asks questions about these, it can get the related information from this part I designed this architecture. The idea is from the data structure and algorithms that we have DFS and BFS. So here I maybe want to use a kind of BFS architecture. To keep more information, because this, this part is for the because you mentioned that said, Luo, maybe have last half hours. It can be searched here, and if the user ask a very specific fix, specific, very detailed question, It can be searched from here. This kind of

[#P140] [21:42] **Ankit:** because kind of because it may,

[#P141] [21:45] **Ankit:** it may ask these questions.

[#P142] [21:52] **Ankit:** Sorry, I need to go to different Okay, just give me a second. Okay, I

[#P143] [22:29] **Ankit:** Yeah, yeah. And do Do you get my idea my I think my explain is very cool. I don't

[#P144] [22:39] **Ankit:** get the idea to be

[#P145] [22:46] **Ankit:** the second idea, I think I want to say that these two layer Is

[#P146] [22:53] **Ankit:** collaborate with each other. Yeah, can you go to level. Can you go

[#P147] [23:05] **Hongye Qian:** so, so here the summary level is different, right? Compared to the summary level, what you have mentioned,

[#P148] [23:13] **Ankit:** yes, yes, yes. So maybe you should

[#P149] [23:15] **Ankit:** call it something differently. You know, if you ask the

[#P150] [23:20] **Hongye Qian:** summary level, what my understanding is is basically I have a conversation, and summary is basically summary of that conversation.

[#P151] [23:29] **Ankit:** Yeah, that is my understanding.

[#P152] [23:32] **Ankit:** Okay, so we should call it, so we should call it something, but the summary level, which you are, can you go to the

[#P153] [23:41] **Hongye Qian:** second slide? You performing. Can you go to the

[#P154] [23:48] **Ankit:** second slide? It is directly summarize the mid level. It will only give one graph. This summary is basically the meeting level summary. And

[#P155] [24:00] **Hongye Qian:** then what is? What does meeting mid

[#P156] [24:05] **Ankit:** level is yes.

[#P157] [24:09] **Ankit:** Yes is the same. The mid level is always, is always same, yeah, but here summary, level means something different, yes, yeah.

[#P158] [24:17] **Hongye Qian:** Can you write in the can

[#P159] [24:30] **Ankit:** you write in bracket? It's basically you divide and 24 hours. Yes, if I will draw a picture, is that okay? I will draw a picture. I

[#P160] [24:56] **Ankit:** Oh, sorry, please wait. This is our raw data, yeah,

[#P161] [25:04] **Ankit:** and

[#P162] [25:08] **Ankit:** this raw data will give many

[#P163] [25:16] **Ankit:** here has a first part of our raw data, and it will have a summary. And

[#P164] [25:30] **Ankit:** yes, yes, one conversation

[#P165] [25:36] **Ankit:** and the part one, but the part two has you. So another question, another question

[#P166] [25:43] **Ankit:** is, what I have been thinking is like, right now, imagine like,

[#P167] [25:51] **Hongye Qian:** there is continuous meetings, right? Yeah, there is continuous discussion. How do we know where does the raw data start and where does the raw data end? For one meeting?

[#P168] [26:02] **Ankit:** Afraid to end for one meeting. What? What can you use discussion means? Okay, okay, let's see. I'm talking to you now. Okay, then

[#P169] [26:20] **Ankit:** I go out of the room, yes, and

[#P170] [26:32] **Ankit:** we have meeting from 230 to 330 Okay, yeah, I am recording every time, so you mean that there

[#P171] [26:42] **Hongye Qian:** may be I so

[#P172] [26:50] **Ankit:** how do you

[#P173] [26:52] **Ankit:** know that this is a different conversation? It can happen, right? But I think this is, this can be separated into two composition we have To solve for you.

[#P174] [27:14] **Hongye Qian:** We need to rely

[#P175] [27:21] **Hongye Qian:** on LLM. To, you know, so when we, when we construct our database,

[#P176] [27:30] **Hongye Qian:** the NLM should kind of

[#P177] [27:33] **Hongye Qian:** say that this is, this seems to be a different conversation. This seems to be different conversation, something like that.

[#P178] [27:38] **Ankit:** Okay, then we'll see. But you be mindful of this. Back to that,

[#P179] [27:48] **Hongye Qian:** yeah, we need to. We need to basically set the boundary from other website. I don't know. I mean, no. Need to do it now. But of course,

[#P180] [27:58] **Ankit:** we need to think about that also. Okay, do you imagine you talk to talk

[#P181] [28:04] **Ankit:** to me about internship? Okay,

[#P182] [28:07] **Ankit:** yeah. Then after we

[#P183] [28:09] **Ankit:** finish the meeting, we talk to your friend about internship.

[#P184] [28:12] **Ankit:** Yes, yes. Just go. Then, then, then, now, this

[#P185] [28:16] **Hongye Qian:** is one conversation, different conversation,

[#P186] [28:20] **Hongye Qian:** yeah, so that leave the problem, but Okay, let's see, when we design

[#P187] [28:27] **Hongye Qian:** our rag, we'll see how much

[#P188] [28:30] **Hongye Qian:** of the problem it is, okay, yeah, I'm hoping that we would have designed a

[#P189] [28:36] **Ankit:** lot of components by then, so that you just jiggle it. Okay, okay, so my, design about that architecture.

[#P190] [28:54] **Ankit:** Maybe I can we see this raw data. Yeah, we can see this raw data.

[#P191] [29:04] **Ankit:** You can see that so

[#P192] [29:09] **Ankit:** beginning time, which is zero minutes to

[#P193] [29:16] **Ankit:** nine minutes and 25 seconds,

[#P194] [29:21] **Ankit:** I extract yet and try to give it a summary. Yeah. And then the next time, next time, we should begin with 1045, and 20 and 21 you can see that the 925, so below of it is the 1045,

[#P195] [29:50] **Ankit:** as the next summary, begin, adhere

[#P196] [29:54] **Ankit:** and go to the go to here it is the next summary. Do you get? Get some understanding? Yeah, actually, here we Yeah, lots and lots of somebody.