from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.http import JsonResponse, StreamingHttpResponse
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain.docstore.document import Document

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, stem_text, STOPWORDS
import re
import json
import os
import openai
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from apify_client import ApifyClient
from dotenv import load_dotenv

import uuid 


load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-ada-002"
PINECONE_CREDENTIALS = os.getenv('PINECONE_API_KEY')
HUGGING_FACE = os.getenv("HUGGING_FACE_KEY")
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")
pc = Pinecone(api_key=PINECONE_CREDENTIALS)
index = pc.Index("storia")

# Langchain pinecone vector store:

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)



vectorstore = PineconeVectorStore(index_name="storia", embedding=embeddings)




from openai import OpenAI

client = OpenAI(
	base_url="https://hnkyim7dr0wn0en3.us-east-1.aws.endpoints.huggingface.cloud/v1/", 
	api_key= HUGGING_FACE 
)


"""
Embedding / Pinecone related
"""
def get_embedding(text, model=EMBEDDING_MODEL):
    result = openai.embeddings.create(
      model=model,
      input=text
    )
    print("gone through")
    # return result["data"][0]["embedding"]
    return result.data[0].embedding

@api_view(['POST'])
def upsert_tweets(request):
    # Assuming the body of the request is the raw JSON data you need to process
    data_json = json.loads(request.body)
    response_data = []

    data_json = data_json['data']
    
    for tweet_data in data_json:
        metadata = {}
        tweet = tweet_data.get('full_text', '')

        # Extracting the required data
        user_data = tweet_data.get('user', {})
        metadata['user'] = user_data.get('name', '')
        metadata['screen_name'] = user_data.get('screen_name', '')
        metadata['full_text'] = tweet
        metadata['url'] = f"https://twitter.com/{user_data.get('screen_name', '')}/status/{tweet_data.get('id_str', '')}"
        metadata['created_at'] = tweet_data.get('created_at', '')
        # An id tying the tweet to the request that created it. 
        metadata['request_id'] = uuid.uuid4()


        tweet_embedding = get_embedding(tweet)
        tweet_id = tweet_data.get('id_str', '')

        tweet_data = [{
            "id": f"{tweet_id}",
            "values": tweet_embedding,
            "metadata": {
                "screen_name": metadata['screen_name'],
                "tweet_text": metadata['full_text'],
                "created_at": metadata['created_at'],
                "url": metadata['url'],
                "user": metadata['user']
            }
        }]
        print("Upserting title embedding data")
        index.upsert(tweet_data)

        # # Assuming you need to do something with metadata and tweet here,
        # # for demonstration, we're just appending it to a response list.
        response_data.append({'metadata': metadata, 'tweet': tweet})
    
    # Returning the processed data as JSON response
    return JsonResponse(response_data, safe=False)

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)

def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub('', text)

additional_stopwords = {"medtwitt", "s", "medx", "type", "common","medtwitter","⁉","🩸","heart🥰","isn","t"}
all_stopwords = STOPWORDS.union(additional_stopwords)

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in all_stopwords])

def preprocess(post):
    post = remove_urls(post)
    post = remove_emojis(post)
    post = ' '.join(preprocess_string(post, [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric]))
    post = remove_stopwords(post)
    return post.split()

def postTopics(posts):
    processed_posts = [preprocess(post) for post in posts]
    dictionary = Dictionary(processed_posts)
    corpus = [dictionary.doc2bow(text) for text in processed_posts]
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=10)

    topics_for_posts = []
    for text in corpus:
        post_topics = lda_model.get_document_topics(text)
        actual_topics = [(lda_model.show_topic(topic[0], topn=5), topic[1]) for topic in post_topics]
        topics_for_posts.append(actual_topics)

    return topics_for_posts


def query_pinecone(query_embedding, str_handles):

    if (len(str_handles) == 0):
        results = index.query(
                    vector=query_embedding, 
                    top_k=10, 
                    include_metadata=True
        )  
    else:
        # handles = [str_handles]
        handles_list = str_handles.split(',')

        # Strip whitespace from each handle
        handles_list = [handle.strip() for handle in handles_list]

        print("handles", handles_list)
        results = index.query(
                vector=query_embedding, 
                namespace="",
                top_k=10, 
                filter = {
                    "screen_name": {"$in": handles_list}
                },
                include_metadata=True
            )  
    return [(match["id"], match["score"], match["metadata"]) for match in results["matches"]]

# @api_retrieve(['GET'])
def retrieve_tweet(user_query):
    query = user_query["user_query"]
    query_embedding = get_embedding(query)

    try:
        get_tweets = query_pinecone(query_embedding, user_query["handles_mentioned"])

        print("\n\n")
        print("______________________________________________________\n") 
        print(f"Successfully retreived: {len(get_tweets)} sources from search")

        scores = []
        chosen_tweets = []

        for tweet_item in get_tweets:

            chunk_identifier, score, metadata = tweet_item[0], tweet_item[1], tweet_item[2]
            print(f"Chunk Identifier: {chunk_identifier}")
            print(f"Score: {score}")
            # print(f"Metadata: {metadata}")

            tweet = metadata["full_text"]
            handle = metadata["screen_name"]
            created_at = metadata["created_at"]
            #  = metadata["transcript_chunk"]
        
            scores.append(score)
            chosen_tweets.append(["tweet: " + tweet, "handle: " + handle, "created_at: " + created_at])

        max_similarity = max(scores)
        print("MAX SIMILIARITY:", max_similarity)

        return chosen_tweets
    except Exception as e: print(e)
    # except:

    #     print("No relevant tweets")

    #     return "failure"

@api_view(['POST'])
def answer_query(request):
    user_query = json.loads(request.body)
    print("User Query: ", user_query) #EX: {'user_query': 'tell me about tumors', 'handles_mentioned': ['sama', elonmusk]}
    chosen_tweets = retrieve_tweet(user_query)
    print(chosen_tweets)

    if (chosen_tweets == "failure"):
        return JsonResponse("No relevant tweets found. Please try another query!", safe=False)

    # Example structure of chosen_tweets: [["tweet1", "@handle1", "created_at:"], ["tweet2", "@handle2", "created_at:"]]
    # chosen_tweets_str = "\n".join([f'{tweet}: {handle}' for tweet, handle in chosen_tweets])
    if chosen_tweets is not None:
        chosen_tweets_str = "\n".join([f'Tweet: {tweet}, Handle: {handle}, Created At: {created_at}' for tweet, handle, created_at in chosen_tweets])
    else:
        chosen_tweets_str = "No tweets selected"
    prompt = """
    Please state the handle first. Then say 'According to the top ten relevant tweets from' [Handle Name]
    then summarize them concisely. Please answer in the form of one short summary, no bullet points. 
    """ 

    # LLM call (Mistral 7B model)
    chat_completion = client.chat.completions.create(
        model="tgi",
        messages=[
        {
            "role": "user",
            "content": user_query["user_query"] 
        },
        {
            "role": "assistant",
            "content": prompt + chosen_tweets_str
        },
    ],
        stream=True,
        max_tokens=500
    )

    # for message in chat_completion:
    #     print(message.choices[0].delta.content, end="")

    final_response_content = ""
    for message in chat_completion:
        final_response_content += message.choices[0].delta.content
        print(message.choices[0].delta.content, end="")

    return JsonResponse(final_response_content, safe=False)

def fetchPostsByHandles(handle_list):
	apify_client = ApifyClient(APIFY_API_TOKEN)
	run_input = {
	    "handles": handle_list,
	    "tweetsDesired": 100,
	    "addUserInfo": True,
	    "startUrls": [],
	    "proxyConfig": { "useApifyProxy": True },
	}


	# Start an actor and wait for it to finish
	actor_call = apify_client.actor('quacker/twitter-scraper').call(run_input=run_input)
	items = []
	for item in apify_client.dataset(actor_call["defaultDatasetId"]).iterate_items():
		metadata = {}
		metadata['screen_name'] = item['user']['screen_name']
		metadata['full_text'] = item['full_text']
		metadata['url'] = item['url']
		metadata['created_at'] = item['created_at']
		# An id tying the tweet to the request that created it. 
		metadata['request_id'] = str(uuid.uuid4())
		items.append(Document(page_content=item["full_text"], metadata=json.loads(json.dumps(metadata))))
	print(items)
	vectorstore.from_documents(index_name="storia", documents=items, embedding=embeddings)
	return items



def huggingface_chat_with_system_prompt(system_prompt_str, human_prompt_str):

	default_system_prompt_str = """You are a journalist and must use the following tweets 
	to answer the question at the end by constructing a narrative about the timeline of the tweets. 
	use tweet metadata like created_at to establish the timeline, and in each step of the timeline, 
	specify two or three sentences explaining that trending tweet. 

	{context}

	Question: {question}

	Helpful Answer:"""
	custom_rag_prompt = PromptTemplate.from_template(default_system_prompt_str)

	retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
	rag_chain = (
	    {"context": retriever | format_docs, "question": RunnablePassthrough()}
	    | custom_rag_prompt
	    | llm
	    | StrOutputParser()
	)

	res = rag_chain.invoke(human_prompt_str)
	print(res.content)
	return res.content

# @api_retrieve(['GET'])
@api_view(['POST'])
def get_tweets_by_handles(request):
    handle_list = json.loads(request.body)

    hundredPosts = fetchPostsByHandles(handle_list["handles"])

    return JsonResponse(hundredPosts, safe=False)



