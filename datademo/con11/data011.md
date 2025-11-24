# data011

[#P001] [00:08] **Ankit:** Can you see my screen?

[#P002] [00:17] **Hongye Qian:** I'm not sure I'm right. Right elect Elastic Search,

[#P003] [00:25] **Ankit:** are you I use then

[#P004] [00:28] **Hongye Qian:** dense factor for the elect stick search, here

[#P005] [00:38] **Ankit:** is configuration of

[#P006] [00:42] **Hongye Qian:** the Elastic Search. And here I use generator evading, generator method, which is here. Here is a generator evading. It will you uh,

[#P007] [01:02] **Ankit:** make our chunks into factors.

[#P008] [01:08] **Ankit:** And here i

[#P009] [01:12] **Ankit:** i call call it,

[#P010] [01:16] **Hongye Qian:** and this method is for the is for the Elastic Search, because this mapping is necessary for the for the electric search. They say, said, We need to write the mapping

[#P011] [01:40] **Ankit:** so, so we can do the I know I asked you to do Elastic Search, but did you

[#P012] [01:53] **Hongye Qian:** why did why do we need Elastic Search? I i see that Elastic Search, there is a user, Elastic Search. There is a kn, kn, they use kn, clue string for the factors to find the top k. I think that may be the reason. And and if we use the electric Elastic Search, if we the filters because we need failed metadatas and electric search gives the gives the efficient way for filters, I I think This may be the reasons.

[#P013] [02:54] **Ankit:** Does that make sense? I

[#P014] [03:13] **Hongye Qian:** like they have because you like electric search, if we use the dense factors in the Elastic Search, they will use kn methods to find the similar factors

[#P015] [03:37] **Ankit:** compared to the traditional method.

[#P016] [03:41] **Ankit:** If we use a fast

[#P017] [03:44] **Hongye Qian:** factor database, like fast they may use cosine, cosines they use, they may use, they may Calculate, calculate the factor cosine to find the similar factors. But elantic Search use k methods,

[#P018] [04:11] **Ankit:** I know there is a second.

[#P019] [04:16] **Hongye Qian:** Second reason is that Elastic Search,

[#P020] [04:22] **Ankit:** gives filters

[#P021] [04:25] **Ankit:** they use.

[#P022] [04:29] **Hongye Qian:** They can filter more easily because they gives.

[#P023] [04:35] **Ankit:** They already realize the filter methods

[#P024] [04:39] **Ankit:** so we can use it

[#P025] [04:51] **Ankit:** inside to write things i

[#P026] [05:09] **Hongye Qian:** i think i searched three example On

[#P027] [05:15] **Ankit:** the elastic

[#P028] [05:17] **Hongye Qian:** document. There are websites for

[#P029] [05:23] **Ankit:** I will find it. This one is

[#P030] [05:27] **Hongye Qian:** an example of reg in the

[#P031] [05:32] **Ankit:** Elastic Search labs. But

[#P032] [05:36] **Hongye Qian:** the difference of it and me is that it use sparse factors, but I use, but I used, but I use dense factors, which is a difference because, but I think we should use dense factors, because we our data set will be many meetings, I think the sparse factors may not be useful.

[#P033] [06:10] **Ankit:** Here is the example. Okay, which one uses path

[#P034] [06:19] **Hongye Qian:** vector, this one, if You use a smart factor, I will find it.

[#P035] [06:41] **Hongye Qian:** I uh, in his video, he said it used sparse factors, okay, okay,

[#P036] [06:58] **Hongye Qian:** maybe you explain it, then let's look. Okay, here, this mapping is a necessary realisement for the Elastic Search. And here, this method is for add trunks. Is for add our trunks, because we split trunks before. This method is for add our trunks. Here, you can see here this method we explained

[#P037] [07:41] **Ankit:** in the last meeting. I think

[#P038] [07:44] **Hongye Qian:** it will translate our trunks into factors, yeah, and this method is for more efficiency. It will, because if we do not use this, if we do not use this method, we will send our trunks one by one to the elect Elastic Search. But if we use this method, we can set a bench, and for example, we can send 100 by bunch one time to the

[#P039] [08:26] **Ankit:** electric search search. So

[#P040] [08:30] **Ankit:** it is, I think

[#P041] [08:33] **Hongye Qian:** it is what it will be, more more efficient. So is that? So it is for do we need to do like first Elastic Search and then

[#P042] [08:55] **Ankit:** do vector search? After that, I I see elect

[#P043] [09:00] **Ankit:** surfactant search and

[#P044] [09:03] **Ankit:** the current realization is that

[#P045] [09:08] **Ankit:** say Elastic Search includes

[#P046] [09:12] **Ankit:** factor search.

[#P047] [09:20] **Hongye Qian:** But I'm not very sure my resume is correct, because I do some test case. It performs very bad at the moment. I will show you later. It was very bad. This is our season we explained before, is for, is this method

[#P048] [09:46] **Ankit:** you can see there, here, I said it for,

[#P049] [09:50] **Hongye Qian:** it is for the efficiency.

[#P050] [09:57] **Ankit:** And the search method

[#P051] [10:00] **Hongye Qian:** is to is for the Cure made by the users. You can see here we use generate, evading this method we talked as a last conversation is for the cure, ebiddings,

[#P052] [10:19] **Ankit:** and here is for the

[#P053] [10:22] **Ankit:** KN curry. The

[#P054] [10:27] **Hongye Qian:** official, official document set says it should be right as this,

[#P055] [10:40] **Ankit:** as this format.

[#P056] [10:47] **Hongye Qian:** And here is, is a filters for the metadata. Let me think, because we should filter the meta metadata first, and then we do the we calculate the evading similarities.

[#P057] [11:09] **Ankit:** So why?

[#P058] [11:13] **Ankit:** Why I write this? Because,

[#P059] [11:17] **Ankit:** as this one

[#P060] [11:19] **Hongye Qian:** is official documents that we should write this, because our JSON format is, you can localize this. You can look, you can look at the actions. Actions is there are many things in the actions and and the official documents that said, usual Azure Rights are next for the actions, And the task is, and this, this, this element

[#P061] [12:04] **Ankit:** is, is different because

[#P062] [12:11] **Hongye Qian:** I will find it. Ey, test document, oh, no, it's not test document. I because you can see that the Task tab is text as the others tab is keywords. The official document says that there's a test we should use. Sorry, our fight.

[#P063] [12:51] **Hongye Qian:** You said of him. Show document says that if the keyword, if the task is text, we should, we should use. We should write this match

[#P064] [13:05] **Ankit:** as the others do not need it. So we need to

[#P065] [13:10] **Hongye Qian:** express this one for the particular logic. And this is all this is while you use this logic is is for the filters,

[#P066] [13:34] **Ankit:** and here,

[#P067] [13:37] **Ankit:** this is for the filter conditions.

[#P068] [13:42] **Hongye Qian:** You so if there is only one, filters will directly to go to the cure. But if there are malice, will use this Boolean. This Boolean means that if there are two filters, we should say, if felt is filter one is correct, and as a filter two is correct, as they both correct,

[#P069] [14:12] **Ankit:** the bullying is means end, as

[#P070] [14:16] **Hongye Qian:** the bullying is The same means sat logic. It means the end.

[#P071] [14:26] **Hongye Qian:** And here we accurate the search and returns, returns the results.

[#P072] [14:36] **Ankit:** This method is for for delays, index,

[#P073] [14:44] **Ankit:** and as this method is for the

[#P074] [14:51] **Ankit:** we should check our

[#P075] [14:54] **Hongye Qian:** connection to the Elastic Search is correct. We use Docker to connect with

[#P076] [15:05] **Ankit:** elastic search

[#P077] [15:07] **Ankit:** now, and I prepared

[#P078] [15:16] **Ankit:** a test method I

[#P079] [15:23] **Ankit:** uh, so the test method

[#P080] [15:32] **Ankit:** I write a basic search,

[#P081] [15:38] **Hongye Qian:** you can see here, but the result is not very good. Say result.

[#P082] [15:52] **Hongye Qian:** So I will run for you. I will run for you. So I think this will be more clear. I

[#P083] [16:10] **Ankit:** it, let's run this test method.

[#P084] [16:17] **Ankit:** It will first generate check the

[#P085] [16:21] **Hongye Qian:** UV connection to the Elastic Search, and you can see here our search results, and CC is the first user method, gives it, gives it. CC is the first user query, and we check data on the metadata. Sorry, what was the topic discussed in data 001, in the real in real time. I mean in real scenario, we cannot say data 001,

[#P086] [17:01] **Hongye Qian:** right? Yes, because I'm not generating LLM logic, I have not realized it. This one is for check for the factor store.

[#P087] [17:18] **Ankit:** This test is only for the check for the

[#P088] [17:22] **Hongye Qian:** factor store I will see, I want to see the top case.

[#P089] [17:31] **Ankit:** Factor Store gives me

[#P090] [17:35] **Hongye Qian:** which is the top case. So this is for the test. Do you do you know my moment? Know My

[#P091] [17:51] **Ankit:** Do you know what I'm doing now?

[#P092] [18:02] **Ankit:** Okay? So

[#P093] [18:06] **Hongye Qian:** this is carry. Is for maybe user will give this carry and then say, like, say, search. I at the moment, I said the top k is three and the score is a case similarity score realized by the Elastic Search. But you can see here the performance is not very good, because I asked, I want to search about the data the first data,

[#P094] [18:46] **Ankit:** but it gives me the data

[#P095] [18:50] **Hongye Qian:** seven and data six and data one. This is the first thing as the first three. I think it's not very good, but to be noticed that

[#P096] [19:04] **Ankit:** in this test, we do not filter,

[#P097] [19:07] **Hongye Qian:** maybe we go, GO filter first, first, The answer will be more will be more good? I

[#P098] [19:23] **Ankit:** uh, so you can see I get gives the

[#P099] [19:27] **Ankit:** second, the second

[#P100] [19:31] **Ankit:** test I want to ask it the action,

[#P101] [19:35] **Ankit:** the action items, which is,

[#P102] [19:41] **Ankit:** I want to ask it,

[#P103] [19:48] **Ankit:** the action items,

[#P104] [19:56] **Hongye Qian:** which which is this? Which is this one? I want to ask about, how is this going? But, but it it truly identifies action items, but it will also give action items from other

[#P105] [20:18] **Ankit:** other meetings. So uh.

[#P106] [20:24] **Ankit:** So I think the

[#P107] [20:25] **Ankit:** result may not very good,

[#P108] [20:31] **Ankit:** but, but if we, if we add the filter first,

[#P109] [20:41] **Ankit:** if we add as a filter first, we please wait.

[#P110] [20:51] **Hongye Qian:** But if we add the filter first, I think it will be more helpful. And at the moment, I give it a filter so it will focus focus on my filters and find the related meetings. It looks like this. You do not need to notice this. This one, because, in this law, because in this logic, I earn test the filter works correctly. I only want to look about my code if we add a filter, if we add a filter, it will give me the right it will give me the right meeting. I want to, only want to check this

[#P111] [22:02] **Hongye Qian:** and this, and this test is for the filter action test,

[#P112] [22:12] **Ankit:** because in this morning i

[#P113] [22:17] **Hongye Qian:** i Find an arrow in the actions. And this, this test is particular for the actions.

[#P114] [22:26] **Ankit:** It's also the filters

[#P115] [22:29] **Hongye Qian:** I want to I want to find the correct meeting back for the filters.

[#P116] [22:41] **Ankit:** That's what I do in the weekend. Sorry.

[#P117] [23:01] **Ankit:** Pass, means this one,

[#P118] [23:09] **Ankit:** since this one means that

[#P119] [23:13] **Hongye Qian:** my test is correct, the code as a test code if i Okay,

[#P120] [23:22] **Ankit:** can you ask? I'm actually bit confused about the question. Okay, the questions which

[#P121] [23:38] **Ankit:** you're asking. Where is the question?

[#P122] [23:41] **Ankit:** Sorry, I as this question is not,

[#P123] [23:47] **Hongye Qian:** this question is not the final question. The query is asked. It's for the

[#P124] [23:55] **Ankit:** testing purpose. No, I understand, but I understand, but

[#P125] [24:06] **Ankit:** what is the kind of question the user

[#P126] [24:19] **Hongye Qian:** is expected to us? I think so. User maybe ask almost everything, but

[#P127] [24:30] **Ankit:** my first test purpose here

[#P128] [24:36] **Hongye Qian:** is for the is for the filter purpose, because Because, because, if we look at this question, will we look at this question? I want my code to generate a filter, for example, to to generate a filter to filter the meeting tab in our metadata. But I have not all but, but I have not realized the LLM, because at this step, or we filter the metadata at the end, we'll give it to the LLM to to identify the filters to filter rameta data. But now I'm not generated yet, so I need to write the filter, manual, manually. LLM will LLM will write you. LLM will write the filter. And picture. Yes, I think the logic may be like this. So when this, when the

[#P129] [25:48] **Ankit:** user asks questions, then,

[#P130] [25:55] **Ankit:** but the LLM will extract topics out of it, right?

[#P131] [26:06] **Ankit:** But the LLM will extract topics out of it,

[#P132] [26:10] **Ankit:** right? I say, if the user asks questions,

[#P133] [26:15] **Ankit:** we should first go through

[#P134] [26:19] **Hongye Qian:** a logic to make the user's question more clearly, and then we should give it to the LM, and the LM should decide

[#P135] [26:34] **Ankit:** which filters it

[#P136] [26:38] **Ankit:** should goes. What do you mean by

[#P137] [26:42] **Ankit:** which? By which builder, for example,

[#P138] [26:47] **Ankit:** let's say that

[#P139] [26:49] **Ankit:** if the user asks these questions,

[#P140] [26:55] **Ankit:** the LLM will to

[#P141] [26:58] **Ankit:** go through the JSON files,

[#P142] [27:03] **Ankit:** yeah, and

[#P143] [27:09] **Ankit:** yes, yes. Json file is look like this, and please with me.

[#P144] [27:19] **Ankit:** And here we already,

[#P145] [27:23] **Ankit:** you can see here, we already

[#P146] [27:27] **Hongye Qian:** realize filter logic. We should, we should tell the LM to design. We design which filters it should mate and for these particular questions that the filters need to focus on the meeting tab. So the LLM should this filter, this filter, the LLM should extract from the April

[#P147] [28:04] **Hongye Qian:** from the question, right? Yes, is the

[#P148] [28:08] **Ankit:** meeting tab should be

[#P149] [28:11] **Hongye Qian:** extracted from the crescent. But now I do not realize the meeting tab. Do not write the ln, so I need to give it the also meeting content, also, right,

[#P150] [28:29] **Hongye Qian:** yes, yes, yes. I think for this question, we should extract the

[#P151] [28:38] **Ankit:** meeting tab. I

[#P152] [28:44] **Hongye Qian:** I want to take test, test, test this, if, if the logic will give me back the correct data because we use the inverted index. I want to, I want to test it. Will give me the right,

[#P153] [29:07] **Ankit:** right data.

[#P154] [29:09] **Hongye Qian:** I only want to test this one. I only want to test this.

[#P155] [29:21] **Ankit:** Do you understand? Okay,

[#P156] [29:32] **Hongye Qian:** so these are the test cases for metadata Correct, yes. This is for the all fours level.

[#P157] [29:44] **Ankit:** Say the test case

[#P158] [29:46] **Ankit:** for the meeting level is here,

[#P159] [29:50] **Hongye Qian:** but, but, but in this test, we do not filter first. We directly give it to the search method so.

[#P160] [00:00] **Hongye Qian:** Okay, you can see the logic here, this question for the meeting level and for the test logic, we

[#P161] [00:14] **Ankit:** use a search method,

[#P162] [00:19] **Ankit:** but you can see here, we directly

[#P163] [00:23] **Hongye Qian:** give it to give it a meeting level to the search method. But we do not filter without we just to compile it use we can. We just to directly embedding the Curie and compel the similarities, but, but as the result is not very good here, what

[#P164] [00:57] **Ankit:** is the question you asked this one? I

[#P165] [01:07] **Hongye Qian:** but code now gives me the top, top three will from the other will from the other meetings. I think

[#P166] [01:21] **Ankit:** why this happens? Because we do not felt first

[#P167] [01:26] **Ankit:** but, but I have not.

[#P168] [01:31] **Hongye Qian:** I have not called the logic for the filter first and similarity second.

[#P169] [01:38] **Ankit:** I if you but if you have,

[#P170] [01:50] **Ankit:** if you do it like better first, yeah, similarity later, yeah,

[#P171] [02:00] **Ankit:** then it can be better,

[#P172] [02:01] **Ankit:** right? Yes, when

[#P173] [02:09] **Hongye Qian:** you only tested test cases or filtering, filtering test cases passed, yes, yeah, I check it. It gives me the right data, the embedding

[#P174] [02:21] **Ankit:** comparison failed.

[#P175] [02:24] **Ankit:** Yes, yes, the embedding comparison failed

[#P176] [02:32] **Ankit:** for using Elasticsearch. I think if we,

[#P177] [02:35] **Ankit:** if I directly use vector search, will the

[#P178] [02:40] **Hongye Qian:** comparison be okay? I um, I think we should not say it fails, because at the moment I says the top, top, top k, k is three. I think it is very limited, because in the rank we have another, another step is re rank. I have not used a re rank method, so I think it actually we cannot say it is failed. Maybe we said, we says Top K is five or 10. It will also identify the data I want.

[#P179] [03:30] **Ankit:** Okay, so can you try two things? One is first filter,

[#P180] [03:33] **Ankit:** then embedding.

[#P181] [03:40] **Ankit:** Yeah, yeah, I will track first

[#P182] [03:45] **Ankit:** embedding for top 10. Okay, okay, I will try these two things.

[#P183] [03:57] **Hongye Qian:** But can I ask some questions because, because you said, which we may use Elastic Search. I read some I read some blogs

[#P184] [04:21] **Ankit:** on the sorry,

[#P185] [04:26] **Hongye Qian:** I read some elective search on The official,

[#P186] [04:32] **Ankit:** official documents.

[#P187] [04:36] **Hongye Qian:** For example, this elastic search I uh, this is not,

[#P188] [04:48] **Ankit:** please, await me.

[#P189] [04:50] **Hongye Qian:** I was, I will say it quickly, because I think your time is not very enough. Because I, I look, I look it's I look it's videos and say, sadly, likes Dick search, offer easier, elects search, they already offers a factor, factor search. I'm wondering that, do we need to use their methods for the factor search, or we should use our own methods? Sorry, I will so I don't know, actually, how good the

[#P190] [05:40] **Ankit:** uh, so I don't know actually how

[#P191] [05:46] **Hongye Qian:** good, maybe I will check it. I already use. I already is here. I already, I already checked their official documents. You can see here, there, there there gives a method called dense factors. And there, how can they find the similarities? As they do not calculate the cosine, they use km methods to find the similarities. The this. This is tells that if we want to use their factors, we need, we need, we necessarily need to give each factors an index, and we should write like this. I also write like like mappings because their documents said, if we need to use their factors that we need to like write like this. I also write, please. You can say I also write this because we, I need, I'm now, I use the factor search in the inside electric search. So I'm wondering that we do we need to use the methods in the official document, gift, in the electric search. We can use

[#P192] [07:24] **Ankit:** our own also, but vector search will be done

[#P193] [07:32] **Ankit:** after the elastic search, right? So once vector

[#P194] [07:40] **Hongye Qian:** search, once you are elastic search, but I have a I have a confusion, because the electric search is already done the factor search in their logic.

[#P195] [07:55] **Ankit:** Do we need to done the factor search again?

[#P196] [08:02] **Hongye Qian:** Because to figure out actually, because my my confusion, is that what the electric search is used is used for our logic, because you recommend me to use it, and I find that it offers a denser factor methods. Yeah, I understand. I understand. So I'm also kind of looking that if we use, like, a

[#P197] [08:40] **Ankit:** is Like, okay,

[#P198] [08:57] **Hongye Qian:** so maybe I will realize factor, traditional factor search methods and tomorrow, if You okay?

[#P199] [09:18] **Hongye Qian:** I think so electric search advantages is for for the larger, larger data,

[#P200] [09:32] **Ankit:** I think because I also check on like because, say,

[#P201] [09:47] **Hongye Qian:** because, for example, if we want to find something, if we,

[#P202] [09:53] **Ankit:** if we, if we want to Find,

[#P203] [09:58] **Hongye Qian:** if the user want to find a particular, particular, particular name, for example, if one meeting includes a concept called deep learning. For example, deep learning. So for example, a particular word, ResNet 50 in the deep learning conversation, if the user want to find only want to find service net 50, I think Elastic Search is efficient for This, for this user, carry because, because I check online, they said they check this very efficient. I think they realized algorithm, but I do not know what I present is like this. Okay, so your recommendation

[#P204] [10:58] **Ankit:** is that we still

[#P205] [11:02] **Ankit:** do

[#P206] [11:02] **Ankit:** recommendation is

[#P207] [11:09] **Hongye Qian:** that we still do Elastic Search first, right? Yes, but, but maybe we do not use a factor method in the electric search. They have other they have other electric search I find on the official document,

[#P208] [11:28] **Hongye Qian:** for example, say, use infer, Inferred index concept and another algorithm to

[#P209] [11:38] **Ankit:** fight a particular words in the raw data.

[#P210] [11:48] **Ankit:** I think that may be useful, yeah,

[#P211] [11:52] **Hongye Qian:** but I'm not very sure we should check on the can you we

[#P212] [11:58] **Ankit:** need to test carefully,

[#P213] [12:07] **Hongye Qian:** right? Okay, so my next step is to use traditional factors stone to check out the results. Is that correct? Okay? Traditional vectors with elastic subs, right? Yeah, traditionally the No, no, no, not. Traditional factor with like, any factor,

[#P214] [12:31] **Ankit:** yeah, I will try this.

[#P215] [12:40] **Ankit:** Okay, and then like, Can you, can you, kind

[#P216] [12:48] **Ankit:** of, can you remind me to kind of give you, somehow the real conversations from the project, so that we include that in

[#P217] [13:01] **Ankit:** our data, rather than only Yeah,

[#P218] [13:04] **Hongye Qian:** I think we need it Yeah, because real conversation.

[#P219] [13:09] **Ankit:** And then yeah, so that that will be also and then, like,

[#P220] [13:15] **Ankit:** we need to prepare the question answers. Question answers. We need to prepare the right question as a part of our test data,

[#P221] [13:23] **Ankit:** yes. Otherwise, we all know that

[#P222] [13:27] **Ankit:** whether it is kind of

[#P223] [13:32] **Ankit:** going to the right pointing to the right data,

[#P224] [13:36] **Ankit:** or it is like not able to figure it out. Okay, okay, I guess a consequence.

[#P225] [13:47] **Ankit:** I think one model may be useful.

[#P226] [13:51] **Hongye Qian:** I think open AI has a one model called whisper.

[#P227] [14:00] **Hongye Qian:** Yes, I'm wondering that could, could a trans translate songs into words so we can get the real data more easily? Is that,

[#P228] [14:18] **Ankit:** is there any possibilities of it? Can you

[#P229] [14:19] **Ankit:** come again? Because,

[#P230] [14:26] **Hongye Qian:** could we use whisper to translate sounds into the

[#P231] [14:33] **Ankit:** conversation, and then we can make some data we

[#P232] [14:42] **Ankit:** have I actually, we have, I think we have

[#P233] [14:47] **Hongye Qian:** conversations which can be used. Okay, okay, so you can give me, yeah, I can give you some transcripts. Okay, yeah, okay, do

[#P234] [15:07] **Hongye Qian:** Okay, so do I need to come to the A star? Because I talked to I also discussed with my friends and say, Give, say, give me another idea. I think that will be useful. But in the meetings, I think I cannot carry the clearly, okay, if you want to come tomorrow, I at

[#P235] [15:54] **Ankit:** maybe Friday? Friday is okay, not tomorrow. Yeah, okay.

[#P236] [15:59] **Ankit:** If you want to come tomorrow,

[#P237] [16:06] **Ankit:** then tomorrow morning is okay, but not so

[#P238] [16:10] **Hongye Qian:** can we Friday? Yeah, so it's all okay tomorrow. Let's catch up in teams. Okay, thank you. I.
