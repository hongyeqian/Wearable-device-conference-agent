# data012

[#P001] [00:00] **Ankit:** Yes, yes. And what is the high level topic? We asked LLM to give us a high level topic, right? Yes, high level topic is look like this, yeah, okay, we want, we want to build this for rag, right? Yes, yes, you.

[#P002] [00:49] **Ankit:** Yeah, so sorry. So can you have another summary? Also, like, this kind of summary, you know, like, Wait, can you show me? Yeah. So, this kind of summary, very you can see the summary abstract.

[#P003] [01:21] **Hongye Qian:** I designed. It in the meta data summary is different from this summary, as metadata is I can use you can all see, right? I metadata has a has a very, very, very, very, very abstract summary, okay, but you can have, like, a bit more still in the similar manner, as in, like, very words, kind of but you can have this kind of summary, right? Yes, but if, sorry, if we design another summary layer. I think our chain is going to be very long. I think, I think, I'm not very sure it will perform. Well, okay, okay. What I'm saying is, like the metadata summary can be slightly okay, I know it because.

[#P004] [02:45] **Ankit:** Because imagine, like you have a three hour conversation, yes, yes. So then you cannot summarize, really summarize that in two times, right? It might be, but that okay, that will play with the prompt, okay, if we do this, we can change, we can change the summary level. I think we can, yeah, we can change these. We can change data. I think it's okay, okay.

[#P005] [03:20] **Ankit:** I suggest one more thing is, is the terminology? 

[#P006] [03:29] **Ankit:** Yeah, is that? Is the terminology? Terminology You understand, right? What is terminology like? Let's say you are calling something summary, so, yeah, somewhere you call it something else, okay.

[#P007] [03:45] **Ankit:** So, because when you say something, sometimes to me, summary, I understand something else, okay, okay, you can have the different nomenclature for different things, okay, I understand when we call summary. So, okay, maybe I rephrase myself. So when you say topic, topic should mean one thing, then you say summary, I'll because if we keep calling everything topic, if we keep calling everything summary, I think we are going to be confused. Okay, okay, I get your idea.

[#P008] [04:16] **Hongye Qian:** Okay, but right now it's good. And then, okay, so what are the next? So what are the next steps?

[#P009] [04:25] **Hongye Qian:** I have some questions. Yeah, sure, because I remember that in the yesterday, before yesterday meetings that you tell me that the user asks a question and there is a semantic data before metadata, and then it goes to the metadata, and after that, we will choose, we will choose the top k meetings. So how we choose the top k meetings that when you do the comparison, it will the rack will throw you top, because, okay, when you how you how you select. Let me complete so, so there is this question, right? Yes, yes. Then you are going to get the embedding of the question, okay, then you are going to have this lot of summaries, right?

[#P010] [05:30] **Ankit:** Everything is stored as an embedding.

[#P011] [05:31] **Hongye Qian:** Oh, I cause idea now the similarity, right? So then it's basically similarity percentage of some metric, right? Yes. You basically filter everything which is greater than 85% or something like that. So I know the concept, but I want to know that, because if we want to do these, we need to use, to we use, we need to embed in our metadata. Then we say we, then we can do this, but, but metadata is looks like this. Is it worth for embedding? I, I think it may not be worth for embedding. I think the metadata is used for the filter. So I'm wondering that if it really filters on like different part of metadata, we have to use differently.

[#P012] [06:41] **Hongye Qian:** Okay, feature in your metadata definitely have to do embedding. Okay, okay, I guess idea. So it's clear. Yes, it's clear now, so I have Get, get one more problem is that you look like you see that here is our road, raw data, if we give them the index, because mid level needs, needs our index for our raw Data. Is that right? So it If the meeting is 24 hours, so the index can be very long, I think it will be 1000s, or maybe it will be 1000s. So is it worth to is there any more efficient way to do this? Because if, if, if, for the raw data, if one graph, I give it an index, it will be many, many, many. So I I want to know if there is a more efficient way. So yeah.

[#P013] [08:02] **Ankit:** But yeah, maybe okay, because I am not trying it and you are writing it so, so, so you so, I'm still not very sure that is there any other possible way other than index, maybe we explore at least, yeah. I mean, that is that can really be problem. I mean, yes, worried about, about.

[#P014] [08:23] **Hongye Qian:** About the index that okay, if it is 24 hours, yes. So, yeah, do you have some ideas for more efficient design? So I think I have one year, but I'm not sure it will be useful. So if we have the raw data, can we do a.

[#P015] [08:55] **Hongye Qian:** Can we apply an unsupervised method, for example, doing clustering on the on yet we can cluster the same topic sentence in one cluster and give this cluster an index. I think that will reduce the index number. I or so so it will, I think it will build another level. I think it just, but I think the sentence will be same. It just to change the sentence.

[#P016] [09:49] **Ankit:** Let me think how to say let me think how to explain it. Can you.

[#P017] [10:11] **Hongye Qian:** Can you explore? Can you like search on internet as the internet do not have this idea come from my own. It means that it will change the order of sentence. For example, if this sentence, if this sentence, idea is same with is is same with this sentence, the algorithm may close the same into more closer into one cluster. Does you get my idea? Yeah, that is okay.

[#P018] [10:52] **Ankit:** So clustering, Clustering is, is okay as an idea. It's just, I'm thinking that does it? You have a.

[#P019] [11:23] **Hongye Qian:** Uh, I'm not very sure it will be reliable. Because, yeah, that is, that is what I'm worried on. Because.

[#P020] [11:31] **Ankit:** This is again, another level of uncertainty, right?

[#P021] [11:36] **Hongye Qian:** If you kind of cluster, if the clustering doesn't work, well, then yeah, we should train a cluster, unsupervised. Yeah, classify. Unsupervised methods are not that good, so, but don't reject the idea. Definitely there is a possibility.

[#P022] [12:00] **Ankit:** Yeah, but the need to be certain of what we are able to repeat, right? So if, like, we retrieve the wrong thing because our trusting algorithm is wrong, then, yes.

[#P023] [12:20] **Hongye Qian:** So as that, okay.

[#P024] [12:41] **Hongye Qian:** Let's think about it, about tomorrow. Okay, tomorrow. Maybe I will, maybe I find out a new structure that maybe, maybe useful. I will talk it tomorrow, because I have not take a close look about yet. Okay, but.

[#P025] [13:03] **Ankit:** One thing I want you to do is whatever you are meeting, yeah, you should be, like, very certain about the code and things like that, so there should not be any loose ends. Okay, okay, yeah, so, so it's not like you just do ask it. It's not like you just ask Karzai and whatever says, Copy, Paste. Don't do that.

[#P026] [13:33] **Hongye Qian:** Because when, when, when our code becomes very big, then it will be very difficult to maintain. Okay, yes, I do not let it generate. I it because there is going to be lot of different features right to be kind of very careful about what we are building. Yeah. I test it function by function, actually. Okay, yeah. So tomorrow will Yes, tomorrow will meet you. Thanks. Sorry for today. I could have told you, okay, just back back to back meetings.

[#P027] [14:13] **Ankit:** I didn't have chance to have no back meetings. I didn't have chance to have lunch today, so I will have to have lunch now. I do not have lunch also because.

[#P028] [14:40] **Hongye Qian:** I'm coding the at Jerome point. Yes.

[#P029] [14:51] **Hongye Qian:** What I just does six the nearby or from the June point, it's very close. No, I, I rent a HDB with my friends. I'm I'm now watching, watching the lesson you give me, and actually the progress is very slow because I have another project that teacher led me to read paper very in detail, so I also spend much time which project you may know it's also from a star. It's from a star bill, yeah, yes. So, yeah, I'm learning every day I plan that in the daytime I'm coding for this project, and the night for that one, I'm playing like this, So maybe I only get two hour for for my how many hours you are working right now? I work six hour a day, yeah, because I use this this. I use this for for control my starting time. It can five minutes break and 40 minutes working. I use this to control my time so I can record how, how many times I'm working. It's about six hour, I think, and then the rest is basically school. Is it? Yes, I have three hour a day for course, but as a course, only part of the course, only have three days, Wednesday to Friday. So I think I have have many times for for myself project. But in.

[#P030] [17:15] **Ankit:** The course, also you will have projects, right? Assignments, yeah, but I think.

[#P031] [17:24] **Hongye Qian:** It is more easier compared with these projects. It is more easy, more easy. Or how is.

[#P032] [17:38] **Ankit:** It is not beginning yet.

[#P033] [17:42] **Hongye Qian:** So I'm we are, we are reading papers now, not beginning coding.

[#P034] [17:51] **Ankit:** So I think said paper is.

[#P035] [17:55] **Hongye Qian:** Also worth to read. I think it's good the what is the topic? It is self self supervised learning. I talked to you before, but as a data, if the teacher do not give us answer, he do not tell us also, he does let us read the papers first, because that said project is for for one year. I think it can be very long, so it have not begin yet. Do okay.

[#P036] [18:47] **Ankit:** Okay, all the best. Work hard. I think this is the time to work hard. Yes, yes. I think it's very viable to me, because reg is the necessary things for finding some job for mle, I think, yes, so yeah, so I think if I want you to do a PhD, I also need to learn about yet.

[#P037] [19:16] **Hongye Qian:** So I think it's okay for me. Yeah.

[#P038] [19:21] **Ankit:** Thank you. Bye. You.